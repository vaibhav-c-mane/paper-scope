"""The web API, with a fake internet behind it."""
import logging
import os
import tempfile
import unittest

from werkzeug.security import generate_password_hash

from admin import MAX_TRIES, LoginGuard
from app import create_app, for_deployment
from engine.pipeline import SearchPipeline
from engine.ranker import HeuristicRanker
from scripts.set_admin import admin_values, read_env, update_env
from tests.fixtures import FakeClock, client_with, lexicon, ok, openalex_work

ADMIN = {"path": "test-door-4821", "user": "owner", "secret_key": "test-key",
         "password_hash": generate_password_hash("correct horse battery", method="pbkdf2:sha256:1000")}
logging.getLogger("paperscope.admin").addHandler(logging.NullHandler())     # sign-in lines stay out of test output


def page(web, url="/"):
    with web.get(url) as r:                       # closes the file behind a static page
        return r.get_data(as_text=True)


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


class AppTests(unittest.TestCase):
    def setUp(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(6)]
        client, self.transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        pipe = SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), capacity=100)
        self.web = create_app(pipe, admin_settings=False).test_client()

    def test_search_then_open_a_paper(self):
        r = self.web.get("/api/search?q=protein+folding+structure").get_json()
        self.assertEqual(r["total"], 6)
        self.assertIn("protein", r["words"])
        key = r["results"][0]["key"]
        p = self.web.get("/api/paper", query_string={"key": key}).get_json()
        self.assertEqual(p["title"], r["results"][0]["title"])
        self.assertNotIn("_keys", p)

    def test_visitors_get_no_engine_internals(self):
        """The trace (sources, timings, budget) and the ranking numbers are only for the admin."""
        r = self.web.get("/api/search?q=please+find+protein+folding+structure").get_json()
        self.assertNotIn("trace", r)
        for k in ("score", "relevance", "features"):
            self.assertNotIn(k, r["results"][0])
        self.assertEqual(set(r["understood"]), {"changed", "query", "original", "corrections", "recent_from"})
        self.assertEqual(set(self.web.get("/api/state").get_json()), {"fields"})
        html = page(self.web)
        self.assertNotIn("How this search worked", html)
        self.assertNotIn("admin", html.lower())

    def test_visitors_cannot_change_the_test_settings(self):
        """online=never would answer from an empty memory; for a visitor it is ignored, so the search
        still goes online, and a value that is not allowed is not an error either."""
        r = self.web.get("/api/search?q=protein+folding&online=never&mode=bm25").get_json()
        self.assertEqual(r["total"], 6)
        self.assertEqual(self.web.get("/api/search?q=x&online=sometimes").status_code, 200)

    def test_safety_headers(self):
        with self.web.get("/") as r:
            self.assertIn("script-src 'self'", r.headers["Content-Security-Policy"])
            self.assertIn("frame-ancestors 'none'", r.headers["Content-Security-Policy"])
            self.assertEqual(r.headers["X-Content-Type-Options"], "nosniff")
            self.assertNotIn("Strict-Transport-Security", r.headers)               # not over http on your computer
        self.assertIn("Content-Security-Policy", self.web.get("/api/state").headers)

    def test_related_papers_follow_citation_links(self):
        """A chain of five papers, each citing the one before; the drawer lists the citation links
        in BFS order, and related papers (Dijkstra) start with the direct citations."""
        titles = [f"Protein folding chain paper {i}" for i in range(5)]
        num = [abs(hash(t)) % 10 ** 8 for t in titles]
        works = [openalex_work(t, refs=[num[i - 1]] if i else []) for i, t in enumerate(titles)]
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        web = create_app(SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker()), admin_settings=False).test_client()
        web.get("/api/search?q=protein+folding+chain")
        self.assertIn("referenced_works", transport.urls("api.openalex.org")[0])
        p = web.get("/api/paper", query_string={"key": f"openalex:w{num[2]}"}).get_json()
        got = [(r["title"][-1], r["hops"], r["relation"]) for r in p["citation_links"]]
        self.assertEqual(got, [("1", 1, "reference"), ("3", 1, "citing"), ("0", 2, None), ("4", 2, None)])
        self.assertEqual(p["citation_links"][2]["via"], titles[1])
        rel = p["related"]
        self.assertEqual({r["title"][-1] for r in rel[:2]}, {"1", "3"})      # citation links are the shortest
        self.assertEqual({r["link"] for r in rel[:2]}, {"cites", "cited by"})
        self.assertTrue(all(0 < r["closeness"] <= 1 for r in rel))
        self.assertTrue(any("cites%3AW" in u for u in transport.urls("api.openalex.org")))   # papers around it
        self.assertTrue(p["citation_data"])
        self.assertNotIn("_refs", p)
        self.assertIsInstance(p["citations"], int)                          # the count, not the link list

    def test_suggest(self):
        self.assertIn("suggestions", self.web.get("/api/suggest?prefix=gra").get_json())
        title = self.web.get("/api/search?q=protein+folding").get_json()["results"][0]["title"]
        got = self.web.get("/api/suggest", query_string={"prefix": "x", "q": title[:12].upper()}).get_json()["titles"]
        self.assertIn(title, [t["title"] for t in got])

    def test_bad_input_is_rejected(self):
        self.assertEqual(self.web.get("/api/search?q=x&field=Astrology").status_code, 400)
        self.assertEqual(self.web.get("/api/search?q=x&sort=alphabetical").status_code, 400)
        r = self.web.get("/api/search?q=protein+AND+(folding").get_json()       # unreadable: words searched instead
        self.assertGreater(r["total"], 0)
        self.assertEqual(self.web.get("/api/paper?key=doi:10.9/missing").status_code, 404)


class AdminTests(unittest.TestCase):
    def setUp(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(6)]
        client, self.transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        self.pipe = SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), capacity=100)
        self.web = create_app(self.pipe, admin_settings=ADMIN).test_client()

    def sign_in(self, password="correct horse battery", user="owner"):
        return self.web.post("/test-door-4821/", data={"username": user, "password": password})

    def test_switched_off_without_settings(self):
        web = create_app(self.pipe, admin_settings=False).test_client()
        self.assertEqual(web.get("/test-door-4821/").status_code, 404)
        self.assertEqual(web.get("/admin").status_code, 404)

    def test_login_page_is_only_at_the_secret_address(self):
        self.assertEqual(self.web.get("/admin").status_code, 404)
        self.assertEqual(self.web.get("/admin/").status_code, 404)
        page = self.web.get("/test-door-4821/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("noindex", page.headers["X-Robots-Tag"])
        self.assertEqual(page.headers["Cache-Control"], "no-store")
        self.assertEqual(self.web.get("/test-door-4821/admin.js").status_code, 404)    # not before signing in

    def test_wrong_password_then_sign_in_and_out(self):
        with self.assertLogs("paperscope.admin") as logs:
            self.assertEqual(self.sign_in("wrong password!!").status_code, 401)
            self.assertEqual(self.sign_in(user="someone").status_code, 401)
            r = self.sign_in()
        self.assertEqual(len(logs.output), 3)
        self.assertIn("admin signed in from 127.0.0.1", logs.output[2])
        self.assertNotIn("wrong password!!", " ".join(logs.output))          # what was typed is never logged
        self.assertNotIn("someone", " ".join(logs.output))
        self.assertEqual((r.status_code, r.headers["Location"]), (302, "/"))
        cookie = r.headers["Set-Cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

        self.assertIn("/test-door-4821/admin.js", page(self.web))
        self.assertIn("How this search worked", page(self.web, "/test-door-4821/admin.js"))
        r = self.web.get("/api/search?q=protein+folding+structure").get_json()
        self.assertTrue(r["trace"]["online"]["used"])
        self.assertIn("features", r["results"][0])
        self.assertEqual(self.web.get("/api/state").get_json()["papers_in_memory"], 6)

        self.web.post("/test-door-4821/logout")
        self.assertNotIn("admin.js", page(self.web))
        self.assertNotIn("trace", self.web.get("/api/search?q=protein+folding").get_json())

    def test_admin_test_settings_are_checked(self):
        self.sign_in()
        self.assertEqual(self.web.get("/api/search?q=x&online=sometimes").status_code, 400)
        self.assertEqual(self.web.get("/api/search?q=x&mode=magic").status_code, 400)
        r = self.web.get("/api/search?q=protein+folding&online=never").get_json()
        self.assertEqual(r["total"], 0)                                     # memory only, and memory is empty
        r = self.web.get("/api/search?q=protein+AND+(folding").get_json()
        self.assertIn("error", r["trace"]["boolean"])

    def test_too_many_wrong_passwords_lock_the_address(self):
        for _ in range(MAX_TRIES):
            self.assertEqual(self.sign_in("wrong password!!").status_code, 401)
        with self.assertLogs("paperscope.admin") as logs:
            r = self.sign_in()                                               # even the right one now waits
        self.assertEqual(r.status_code, 429)
        self.assertIn("refused", logs.output[0])
        self.assertNotIn("Set-Cookie", r.headers)
        self.assertNotIn("trace", self.web.get("/api/search?q=protein").get_json())

    def test_lock_ends_and_a_right_password_clears_the_count(self):
        clock = FakeClock()
        g = LoginGuard(tries=3, lock_s=60, clock=clock)
        for _ in range(3):
            g.wrong("1.2.3.4")
        self.assertAlmostEqual(g.wait_s("1.2.3.4"), 60)
        self.assertEqual(g.wait_s("5.6.7.8"), 0)
        clock.t += 61
        self.assertEqual(g.wait_s("1.2.3.4"), 0)
        g.wrong("1.2.3.4")
        g.right("1.2.3.4")
        g.wrong("1.2.3.4")
        g.wrong("1.2.3.4")
        self.assertEqual(g.wait_s("1.2.3.4"), 0)                             # two in a row, not three

    def test_a_new_password_signs_everyone_out(self):
        self.sign_in()
        changed = dict(ADMIN, password_hash=generate_password_hash("another long password", method="pbkdf2:sha256:1000"))
        fresh = create_app(self.pipe, admin_settings=changed).test_client()
        fresh.set_cookie("session", self.web.get_cookie("session").value)          # same cookie key, old password
        self.assertNotIn("admin.js", page(fresh))


class DeploymentTests(unittest.TestCase):
    """wsgi.py's settings for a public site (app.for_deployment): one proxy in front, HTTPS."""
    def setUp(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(6)]
        client, _, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        pipe = SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), capacity=100)
        self.web = for_deployment(create_app(pipe, admin_settings=ADMIN), proxies=1, https=True).test_client()

    def via(self, forwarded):
        """Headers as the host's proxy sends them: it adds the address it got the request from."""
        return {"X-Forwarded-For": forwarded, "X-Forwarded-Proto": "https"}

    def sign_in(self, forwarded, password="correct horse battery"):
        return self.web.post("/test-door-4821/", data={"username": "owner", "password": password}, headers=self.via(forwarded))

    def test_wrong_passwords_count_per_visitor_behind_the_proxy(self):
        """Every request comes from the proxy; the lock is still per visitor, and a visitor cannot
        get round it by putting made-up addresses in front of the one the proxy adds."""
        with self.assertLogs("paperscope.admin") as logs:
            for i in range(MAX_TRIES):
                self.assertEqual(self.sign_in(f"10.0.0.{i}, 203.0.113.7", "wrong password!!").status_code, 401)
            self.assertEqual(self.sign_in("10.9.9.9, 203.0.113.7").status_code, 429)
            self.assertEqual(self.sign_in("198.51.100.4").status_code, 302)          # someone else is not locked out
        self.assertIn("from 203.0.113.7", logs.output[0])
        self.assertIn("admin signed in from 198.51.100.4", logs.output[-1])

    def test_sign_in_cookie_only_travels_over_https(self):
        with self.assertLogs("paperscope.admin"):
            r = self.sign_in("198.51.100.4")
        cookie = r.headers["Set-Cookie"]
        self.assertTrue(cookie.startswith("__Host-session="))
        self.assertIn("Secure", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertEqual(self.web.get("/api/state", headers=self.via("198.51.100.4")).get_json()["papers_in_memory"], 0)
        with self.web.get("/", headers=self.via("198.51.100.4")) as home:
            self.assertIn("max-age=", home.headers["Strict-Transport-Security"])
            self.assertIn("/test-door-4821/admin.js", home.get_data(as_text=True))


class SetAdminTests(unittest.TestCase):
    def test_env_lines_are_added_and_replaced_and_the_rest_kept(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ".env")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("# my keys\r\nOPENALEX_API_KEY=abc\r\nPAPERSCOPE_ADMIN_USER=old\r\n")
            first = admin_values("owner", "a long enough password", read_env(path))
            update_env(path, first)
            text = read(path)
            self.assertTrue(text.startswith("# my keys\r\nOPENALEX_API_KEY=abc\r\nPAPERSCOPE_ADMIN_USER=owner\r\n"))
            env = read_env(path)
            self.assertNotIn("a long enough password", text)
            self.assertTrue(env["PAPERSCOPE_ADMIN_PASSWORD_HASH"].startswith("scrypt:"))
            self.assertGreaterEqual(len(env["PAPERSCOPE_ADMIN_PATH"]), 12)

            update_env(path, admin_values("owner", "a different password", env))    # address and key kept
            again = read_env(path)
            self.assertEqual(again["PAPERSCOPE_ADMIN_PATH"], env["PAPERSCOPE_ADMIN_PATH"])
            self.assertEqual(again["PAPERSCOPE_SECRET_KEY"], env["PAPERSCOPE_SECRET_KEY"])
            self.assertNotEqual(again["PAPERSCOPE_ADMIN_PASSWORD_HASH"], env["PAPERSCOPE_ADMIN_PASSWORD_HASH"])
            self.assertEqual(read(path).count("PAPERSCOPE_ADMIN_PATH="), 1)
            moved = admin_values("owner", "a different password", again, new_address=True)
            self.assertNotEqual(moved["PAPERSCOPE_ADMIN_PATH"], again["PAPERSCOPE_ADMIN_PATH"])


if __name__ == "__main__":
    unittest.main()
