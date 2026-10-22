"""Online sources: cascade, in-memory cache, cool-down, deadline, batching, normalisation.
No test touches the network: every HTTP call goes to a FakeTransport."""
import unittest
import urllib.parse

from engine.papers import paper_from_openalex
from engine.sources import RateLimiter, api_query, openalex_number
from tests.fixtures import FakeClock, client_with, ok, openalex_work, s2_paper, timeout


def query_of(url):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))


class ClientTests(unittest.TestCase):
    def test_openalex_answers_and_cascade_stops(self):
        works = [openalex_work(f"Protein folding paper {i}", doi=f"10.1/pf{i}") for i in range(8)]
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})})
        papers, report = client.search("protein folding", limit=10)
        self.assertEqual(len(papers), 8)
        self.assertEqual(transport.urls("semanticscholar"), [])
        self.assertEqual(report[0]["provider"], "OpenAlex")
        self.assertEqual(query_of(transport.calls[0]["url"])["search"], "protein folding")

    def test_repeated_search_is_served_from_memory(self):
        works = [openalex_work(f"Protein folding paper {i}") for i in range(8)]
        client, transport, clock = client_with({"api.openalex.org/works": ok({"results": works})})
        client.search("protein folding", limit=10)
        _, report = client.search("Protein   folding", limit=10)
        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(report[0]["cached"])
        clock.t += client.search_ttl + 1                                   # expired: asked again
        client.search("protein folding", limit=10)
        self.assertEqual(len(transport.calls), 2)

    def test_cache_is_bounded(self):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": [openalex_work("A b c")] * 3})},
                                           order=("openalex",), cache_size=2)
        for q in ("one", "two", "three", "one"):
            client.search(q)
        self.assertEqual(len(client.cache), 2)
        self.assertEqual(len(transport.calls), 4)                          # "one" had been evicted

    def test_not_found_is_cached_for_a_shorter_time(self):
        client, transport, clock = client_with({
            "api.openalex.org/works": ok({"results": []}),
            "semanticscholar.org": ok({"data": []}),
            "api.crossref.org": ok({"message": {"items": []}}),
        })
        client.search("nothing here")
        client.search("nothing here")
        self.assertEqual(len(transport.calls), 3)
        clock.t += client.negative_ttl + 1
        client.search("nothing here")
        self.assertEqual(len(transport.calls), 6)

    def test_timeout_falls_through_with_retry_and_backoff(self):
        papers = [s2_paper(f"Protein structure {i}", doi=f"10.2/ps{i}") for i in range(8)]
        client, transport, clock = client_with({
            "api.openalex.org": timeout,
            "semanticscholar.org/graph/v1/paper/search": ok({"data": papers}),
        })
        found, report = client.search("protein folding", limit=10)
        self.assertEqual([r["provider"] for r in report], ["OpenAlex", "Semantic Scholar"])
        self.assertEqual([r["ok"] for r in report], [False, True])
        self.assertEqual(len(transport.urls("openalex")), 2)                # one retry
        self.assertIn(0.5, clock.slept)
        self.assertEqual(found[0]["published"], True)
        self.assertEqual(found[0]["venue"], "Nature")

    def test_failed_provider_cools_down(self):
        client, transport, clock = client_with({"api.openalex.org": timeout,
                                                "semanticscholar.org": ok({"data": [s2_paper("X y z")]}),
                                                "crossref.org": ok({"message": {"items": []}})})
        client.search("first")
        before = len(transport.urls("openalex"))
        client.search("second")
        self.assertEqual(len(transport.urls("openalex")), before)
        clock.t += client.COOLDOWN + 1
        client.search("third")
        self.assertGreater(len(transport.urls("openalex")), before)

    def test_used_up_budget_skips_the_provider_without_waiting(self):
        # what api.openalex.org answers when the free daily budget is used up (seen live, 2026-10-19)
        budget = lambda method, url, body: (429, {"Retry-After": "59585"}, {"message": "Insufficient budget."})
        items = [{"DOI": f"10.3/x{i}", "title": [f"Surface codes {i}"], "type": "journal-article"} for i in range(6)]
        client, transport, clock = client_with({"api.openalex.org": budget, "semanticscholar.org": budget,
                                                "crossref.org": ok({"message": {"items": items}})})
        papers, report = client.search("surface codes", limit=10)
        self.assertEqual(len(papers), 6)                                    # Crossref still answered
        self.assertEqual(len(transport.urls("openalex")), 1)               # no retries, no sleeping
        self.assertEqual(clock.slept, [])
        self.assertIn("Insufficient budget", report[0]["error"])
        clock.t += client.COOLDOWN + 1                                     # a minute is not enough: still skipped
        client.search("another query", limit=10)
        self.assertEqual(len(transport.urls("openalex")), 1)

    def test_doi_lookup_accepts_a_work_whose_main_doi_differs(self):
        # live OpenAlex: filter doi:10.48550/arxiv.1706.03762 returns the work, whose own doi is another one
        work = openalex_work("Attention Is All You Need", doi="10.65215/2q58a426")
        client, _, _ = client_with({"api.openalex.org/works": ok({"results": [work]})}, order=("openalex",))
        p, _ = client.get_paper("10.48550/arXiv.1706.03762")
        self.assertEqual(p["title"], "Attention Is All You Need")

    def test_crossref_drops_non_papers_and_looks_up_dois(self):
        items = [{"DOI": "10.3/fig", "title": ["Figure 2"], "type": "component"},
                 {"DOI": "10.3/a", "title": ["A real article"], "type": "journal-article",
                  "issued": {"date-parts": [[None]]}, "created": {"date-parts": [[2025, 8, 20]]}}]
        client, transport, _ = client_with({"api.crossref.org/works?": ok({"message": {"items": items}}),
                                            "api.crossref.org/works/10.3/a": ok({"message": items[1]})},
                                           order=("crossref",))
        papers, _ = client.search("anything")
        self.assertEqual([p["title"] for p in papers], ["A real article"])
        self.assertEqual(papers[0]["year"], 2025)                          # no issue date: DOI registration year
        p, _ = client.get_paper("10.3/a")
        self.assertEqual(p["title"], "A real article")

    def test_deadline_limits_the_wait(self):
        def slow(method, url, body):
            clock.t += 5
            raise TimeoutError("slow")
        client, transport, clock = client_with({"api.openalex.org": slow, "semanticscholar.org": slow,
                                                "crossref.org": slow})
        start = clock.t
        papers, report = client.search("anything", deadline_s=6)
        self.assertEqual(papers, [])
        self.assertLess(clock.t - start, 12)

    def test_year_filters_and_query_syntax_reach_the_api(self):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": []})}, order=("openalex",))
        client.search("deep AND learn* -survey", year_from=2018, year_to=2020)
        q = query_of(transport.calls[0]["url"])
        self.assertEqual(q["filter"], "publication_year:2018-2020")
        self.assertEqual(q["search"], "deep AND learn NOT survey")
        self.assertEqual(api_query("graph -review"), "graph NOT review")

    def test_get_paper_by_doi_arxiv_and_openalex_id(self):
        work = openalex_work("A paper found by its DOI", doi="10.5555/abc")
        found = lambda method, url, body: (200, {}, {"results": [] if "arxiv" in url.lower() else [work]})
        client, transport, _ = client_with({"api.openalex.org/works": found})
        p, report = client.get_paper("10.5555/abc")
        self.assertEqual(p["title"], "A paper found by its DOI")
        self.assertIn("doi:10.5555/abc", query_of(transport.calls[0]["url"])["filter"])
        client.get_paper("1706.03762")                                     # arXiv id -> arXiv DOI
        self.assertIn("doi:10.48550/arxiv.1706.03762", query_of(transport.calls[1]["url"])["filter"])
        self.assertIn("ARXIV:1706.03762", transport.calls[2]["body"]["ids"])   # not in OpenAlex: Semantic Scholar
        client.get_paper("W12345")
        self.assertEqual(query_of(transport.calls[3]["url"])["filter"], "ids.openalex:W12345")
        _, report = client.get_paper("10.5555/abc")
        self.assertTrue(report["cached"])
        self.assertEqual(len(transport.calls), 4)

    def test_by_ids_batches_fifty_per_request(self):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": []})}, order=("openalex",))
        client.providers["openalex"].by_ids(list(range(1, 121)))
        self.assertEqual(len(transport.calls), 3)

    def test_rate_limiter_spaces_requests(self):
        clock = FakeClock()
        rl = RateLimiter(2, burst=1, clock=clock, sleep=clock.sleep)
        for _ in range(3):
            rl.acquire()
        self.assertEqual(clock.slept, [0.5, 0.5])
        self.assertFalse(rl.acquire(deadline=clock.t + 0.1))


class NormaliseTests(unittest.TestCase):
    def test_openalex_work_fields(self):
        w = openalex_work("Graph attention networks", doi="10.1/gat", venue="Journal of ML", listed_in=["norway-2"],
                          refs=[11, 22])
        p = paper_from_openalex(w)
        self.assertEqual((p["published"], p["venue_quality"], p["venue"]), (True, 1.0, "Journal of ML"))
        self.assertEqual(p["abstract"].split()[0], "protein")
        self.assertIn("Molecular Biology", p["topic"])
        pre = paper_from_openalex(openalex_work("A preprint", work_type="preprint"))
        self.assertEqual((pre["published"], pre["venue_quality"]), (False, 0.0))
        self.assertIsNone(paper_from_openalex({"id": "W1"}))
        self.assertEqual(openalex_number("https://openalex.org/W2741809807"), 2741809807)


if __name__ == "__main__":
    unittest.main()
