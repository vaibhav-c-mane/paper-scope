"""Online sources: cascade, in-memory cache, cool-down, deadline, batching, normalisation.
No test touches the network: every HTTP call goes to a FakeTransport."""
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from engine.papers import paper_from_openalex
from engine.sources import ArXiv, Budget, Http, RateLimiter, api_query, openalex_number
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

    def test_search_by_meaning_or_title_fails_alone(self):
        works = [openalex_work(f"Surface codes {i}") for i in range(4)]
        broken = lambda method, url, body: (500, {}, {})
        route = lambda method, url, body: broken(method, url, body) if "search.semantic" in url else (200, {}, {"results": works})
        client, transport, clock = client_with({"api.openalex.org/works": route}, order=("openalex",))
        found, entry = client.search_openalex("semantic", "how surface codes protect qubits")
        self.assertEqual((found, entry["ok"], entry["provider"]), ([], False, "OpenAlex (by meaning)"))
        papers, report = client.search("surface codes", limit=10)          # the keyword search is not paused
        self.assertEqual((len(papers), report[0]["ok"]), (4, True))
        found, entry = client.search_openalex("titles", "surface codes")
        self.assertEqual((len(found), entry["ok"]), (4, True))
        self.assertIn("sort=cited_by_count%3Adesc", transport.urls("title.search")[0])
        found, entry = client.search_openalex("semantic", "how surface codes protect qubits")
        self.assertIn("skipped", entry["error"])                           # cooling down after the failure

    def test_used_up_budget_on_any_openalex_search_pauses_all_of_them(self):
        budget = lambda method, url, body: (429, {"Retry-After": "59585"}, {"message": "Insufficient budget."})
        client, transport, clock = client_with({"api.openalex.org": budget}, order=("openalex",))
        client.search_openalex("semantic", "how surface codes protect qubits")
        papers, report = client.search("surface codes", limit=10)
        self.assertIn("skipped", report[0]["error"])
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


# what export.arxiv.org answers (Atom XML), shortened from a real response
ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2104.01234v2</id>
    <published>2021-04-03T10:00:00Z</published>
    <title>Deep Reinforcement Learning for
      Autonomous Navigation</title>
    <summary>  We train a navigation policy with deep reinforcement learning.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:doi>10.1109/LRA.2021.1</arxiv:doi>
    <arxiv:journal_ref>IEEE RA-L 6 (2021)</arxiv:journal_ref>
    <link href="http://arxiv.org/abs/2104.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2104.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.RO"/>
    <category term="cs.LG"/>
    <category term="cs.RO"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2201.00002v1</id>
    <published>2022-01-01T00:00:00Z</published>
    <title>A preprint without a journal</title>
    <summary>Abstract.</summary>
    <author><name>Grace Hopper</name></author>
    <link href="http://arxiv.org/abs/2201.00002v1" rel="alternate" type="text/html"/>
    <category term="cs.LG"/>
  </entry>
</feed>"""

EPMC = {"hitCount": 2, "resultList": {"result": [
    {"id": "12345", "source": "MED", "pmid": "12345", "doi": "10.1038/S41586-1", "title": "CRISPR off-target effects.",
     "authorString": "Doudna J, Charpentier E.", "pubYear": "2019", "abstractText": "We measure <i>off-target</i> edits.",
     "journalInfo": {"journal": {"title": "Nature"}}, "citedByCount": 321,
     "fullTextUrlList": {"fullTextUrl": [{"availabilityCode": "OA", "documentStyle": "html", "url": "https://x/html"},
                                         {"availabilityCode": "OA", "documentStyle": "pdf", "url": "https://x/pdf"}]}},
    {"id": "PPR1", "source": "PPR", "title": "A &lt;i&gt;bioRxiv&lt;/i&gt; preprint", "pubYear": "2024", "citedByCount": 0},
]}}


def budget_headers(left, limit=1, cost=0.001, reset=3600):
    return {"X-RateLimit-Remaining-USD": str(left), "X-RateLimit-Limit-USD": str(limit),
            "X-RateLimit-Cost-USD": str(cost), "X-RateLimit-Reset": str(reset)}


class FreeSourceTests(unittest.TestCase):
    def test_arxiv_atom_is_parsed(self):
        papers = ArXiv(Http()).parse(ARXIV_FEED)
        a, b = papers
        self.assertEqual(a["arxiv_id"], "2104.01234")
        self.assertEqual(a["title"], "Deep Reinforcement Learning for Autonomous Navigation")
        self.assertEqual((a["authors"], a["year"], a["categories"]), ("Ada Lovelace, Alan Turing", 2021, "cs.RO cs.LG"))
        self.assertEqual((a["doi"], a["venue"], a["published"]), ("10.1109/lra.2021.1", "", True))
        self.assertEqual((a["oa_url"], a["citations"]), ("http://arxiv.org/pdf/2104.01234v2", None))
        self.assertEqual((b["published"], b["doi"]), (False, None))

    def test_arxiv_query_syntax(self):
        self.assertEqual(ArXiv.search_query("the protein folding -review"),
                         "(all:protein AND all:folding) ANDNOT all:review")
        self.assertEqual(ArXiv.search_query("qubits", 2020, 2022),
                         "all:qubits AND submittedDate:[202001010000 TO 202212312359]")
        self.assertEqual(ArXiv.search_query("the of"), "")
        self.assertEqual(ArXiv.search_query("teaching computers to recognise emotions -survey"),
                         "(all:teaching OR all:computers OR all:recognise OR all:emotions) ANDNOT all:survey")

    def test_europe_pmc_results(self):
        client, transport, _ = client_with({"ebi.ac.uk/europepmc": ok(EPMC)}, order=(), field_sources=("europepmc",))
        papers, report = client.search("crispr off-target", limit=10, field="Biology & Medicine", year_from=2018)
        a, b = papers
        self.assertEqual((a["title"], a["doi"], a["venue"], a["citations"]), ("CRISPR off-target effects", "10.1038/s41586-1", "Nature", 321))
        self.assertEqual((a["abstract"], a["oa_url"], a["published"]), ("We measure off-target edits.", "https://x/pdf", True))
        self.assertEqual((b["published"], b["venue"], b["title"]), (False, "", "A bioRxiv preprint"))
        self.assertEqual(report[0]["provider"], "Europe PMC")
        self.assertEqual(query_of(transport.calls[0]["url"])["query"], "TITLE_ABS:(crispr AND off-target) AND PUB_YEAR:[2018 TO 2100]")

    def test_the_field_picks_the_free_source(self):
        client, _, _ = client_with({}, field_sources=("arxiv", "europepmc"))
        self.assertEqual(client.plan("Robotics & Control"), ["openalex", "arxiv", "semanticscholar", "crossref"])
        self.assertEqual(client.plan("Biology & Medicine"), ["openalex", "europepmc", "semanticscholar", "crossref"])
        self.assertEqual(client.plan("Economics & Finance"), ["openalex", "semanticscholar", "crossref"])
        self.assertEqual(client.plan(None), ["openalex", "semanticscholar", "crossref"])
        self.assertEqual(client.plan("Biology & Medicine", saver=True), ["europepmc", "semanticscholar", "openalex", "crossref"])

    def test_arxiv_answers_when_openalex_is_out_of_budget(self):
        used_up = lambda method, url, body: (429, {"Retry-After": "40000"}, {"message": "Insufficient budget."})
        client, transport, _ = client_with({"api.openalex.org": used_up, "export.arxiv.org": lambda m, u, b: (200, {}, ARXIV_FEED)},
                                           order=("openalex",), field_sources=("arxiv",))
        papers, report = client.search("reinforcement navigation", limit=4, field="Robotics & Control")
        self.assertEqual([r["provider"] for r in report], ["OpenAlex", "arXiv"])
        self.assertEqual(len(papers), 2)
        self.assertTrue(client.saver()[0])                                  # used up: saver mode from now on
        client.search("another query", limit=4, field="Robotics & Control")
        self.assertEqual(len(transport.urls("openalex")), 1)               # free source first, OpenAlex not asked
        self.assertEqual(transport.urls("arxiv")[-1].count("all%3Aanother"), 1)


class BudgetTests(unittest.TestCase):
    def test_headers_are_read_and_saver_mode_starts_below_a_fifth(self):
        state = {"left": 0.5}
        works = [openalex_work(f"Surface codes {i}") for i in range(4)]
        route = lambda method, url, body: (200, budget_headers(state["left"]), {"results": works})
        client, _, clock = client_with({"api.openalex.org": route}, order=("openalex",))
        self.assertEqual(client.budget()["known"], False)
        client.search("surface codes")
        b = client.budget()
        self.assertEqual((b["remaining_usd"], b["limit_usd"], b["share"], b["spent_usd"], b["saver"]), (0.5, 1.0, 0.5, 0.001, False))
        state["left"] = 0.15
        client.search("toric codes")
        self.assertEqual(client.saver(), (True, "only 15% of today's OpenAlex budget is left"))
        clock.t += 3601                                                     # the day's budget is full again
        self.assertEqual((client.budget()["share"], client.saver()[0]), (1.0, False))

    def test_saver_can_be_forced_on_or_off(self):
        client, _, _ = client_with({}, order=("openalex",))
        with mock.patch.dict(os.environ, {"PAPERSCOPE_SAVER": "on"}):
            self.assertTrue(client.saver()[0])
        b = Budget()
        b.observe(budget_headers(0.01))
        client.providers["openalex"].budget = b
        with mock.patch.dict(os.environ, {"PAPERSCOPE_SAVER": "off"}):
            self.assertFalse(client.saver()[0])
        self.assertTrue(client.saver()[0])


class CacheFileTests(unittest.TestCase):
    def test_answers_survive_a_restart_and_not_found_is_not_written(self):
        works = [openalex_work(f"Protein folding paper {i}") for i in range(8)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "api-cache.jsonl")
            client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})},
                                               order=("openalex",), cache_file=path)
            client.search("protein folding", limit=10)
            client.get_paper("10.9/missing")                               # not found: memory only
            client.cache.disk.close()
            restarted, transport2, _ = client_with({}, order=("openalex",), cache_file=path)
            papers, report = restarted.search("Protein folding", limit=10)
            self.assertEqual((len(papers), report[0]["cached"], transport2.calls), (8, True, []))
            self.assertEqual(restarted.budget()["cache_file_hits"], 1)
            self.assertEqual(len(restarted.cache.disk), 1)
            restarted.cache.disk.close()


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
