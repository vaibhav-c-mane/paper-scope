"""The web API, with a fake internet behind it."""
import unittest

from app import create_app
from engine.pipeline import SearchPipeline
from engine.ranker import HeuristicRanker
from tests.fixtures import client_with, lexicon, ok, openalex_work


class AppTests(unittest.TestCase):
    def setUp(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(6)]
        client, self.transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        pipe = SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), capacity=100)
        self.web = create_app(pipe).test_client()

    def test_search_then_open_a_paper(self):
        r = self.web.get("/api/search?q=protein+folding+structure").get_json()
        self.assertEqual(r["total"], 6)
        self.assertTrue(r["trace"]["online"]["used"])
        key = r["results"][0]["key"]
        p = self.web.get("/api/paper", query_string={"key": key}).get_json()
        self.assertEqual(p["title"], r["results"][0]["title"])
        self.assertNotIn("_keys", p)

    def test_related_papers_follow_citation_links(self):
        """A chain of five papers, each citing the one before; the drawer lists them by BFS order."""
        titles = [f"Protein folding chain paper {i}" for i in range(5)]
        num = [abs(hash(t)) % 10 ** 8 for t in titles]
        works = [openalex_work(t, refs=[num[i - 1]] if i else []) for i, t in enumerate(titles)]
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        web = create_app(SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker())).test_client()
        web.get("/api/search?q=protein+folding+chain")
        self.assertIn("referenced_works", transport.urls("api.openalex.org")[0])
        p = web.get("/api/paper", query_string={"key": f"openalex:w{num[2]}"}).get_json()
        got = [(r["title"][-1], r["hops"], r["relation"]) for r in p["related"]]
        self.assertEqual(got, [("1", 1, "reference"), ("3", 1, "citing"), ("0", 2, None), ("4", 2, None)])
        self.assertEqual(p["related"][2]["via"], titles[1])
        self.assertTrue(p["citation_data"])
        self.assertNotIn("_refs", p)

    def test_state_and_suggest(self):
        self.web.get("/api/search?q=protein+folding")
        s = self.web.get("/api/state").get_json()
        self.assertEqual(s["papers_in_memory"], 6)
        self.assertEqual(s["ranker"]["kind"], "heuristic")
        self.assertIn("suggestions", self.web.get("/api/suggest?prefix=gra").get_json())

    def test_bad_input_is_rejected(self):
        self.assertEqual(self.web.get("/api/search?q=x&online=sometimes").status_code, 400)
        self.assertEqual(self.web.get("/api/search?q=x&field=Astrology").status_code, 400)
        self.assertEqual(self.web.get("/api/search?q=x&sort=alphabetical").status_code, 400)
        r = self.web.get("/api/search?q=protein+AND+(folding").get_json()       # unreadable: words searched instead
        self.assertIn("error", r["trace"]["boolean"])
        self.assertGreater(r["total"], 0)
        self.assertEqual(self.web.get("/api/paper?key=doi:10.9/missing").status_code, 404)


if __name__ == "__main__":
    unittest.main()
