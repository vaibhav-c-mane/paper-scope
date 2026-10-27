"""The search pipeline end to end: memory first, the internet when memory is not enough."""
import unittest

from engine.pipeline import SearchPipeline, parse_query
from engine.ranker import HeuristicRanker
from tests.fixtures import client_with, lexicon, make_corpus, ok, openalex_work, timeout


def protein_works(n=8, **kw):
    return [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}", year=2010 + i, **kw)
            for i in range(n)]


def pipeline(client=None, capacity=200):
    return SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), capacity=capacity,
                          now_year=2024)


class ParseTests(unittest.TestCase):
    def test_parse(self):
        p = parse_query("graph neur* -survey")
        self.assertEqual((p["words"], p["excluded"], p["boolean"]), (["graph", "neur*"], ["survey"], None))
        self.assertEqual(parse_query("10.48550/arXiv.1706.03762")["identifier"], ("doi", "10.48550/arxiv.1706.03762"))
        self.assertEqual(parse_query("1706.03762v2")["identifier"], ("arxiv", "1706.03762"))
        self.assertEqual(parse_query("W2741809807")["identifier"], ("openalex", "W2741809807"))
        self.assertIsNotNone(parse_query("a AND (b OR c)")["boolean"])


class OnlineFlowTests(unittest.TestCase):
    def test_empty_memory_asks_the_internet_and_keeps_the_papers(self):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": protein_works()})},
                                           order=("openalex",))
        pipe = pipeline(client)
        res = pipe.search("protein folding structure")
        self.assertEqual(len(transport.urls("works?search=")), 1)
        self.assertEqual(len(transport.urls("title.search")), 1)          # three words: maybe a title
        self.assertTrue(res["trace"]["online"]["used"])
        self.assertEqual(res["trace"]["online"]["new"], 8)
        self.assertEqual(res["total"], 8)
        self.assertIn("Protein folding", res["results"][0]["title"])
        self.assertEqual(len(res["results"][0]["features"]), 9)
        self.assertEqual(pipe.state()["papers_in_memory"], 8)

    def test_repeated_query_is_answered_from_memory(self):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": protein_works()})},
                                           order=("openalex",))
        pipe = pipeline(client)
        first = pipe.search("protein folding structure")
        second = pipe.search("protein folding structure")
        self.assertEqual(len(transport.calls), 2)                         # keyword + title, no second round
        self.assertFalse(second["trace"]["online"]["used"])
        self.assertEqual([r["key"] for r in first["results"]], [r["key"] for r in second["results"]])

    def test_new_topic_goes_online_again(self):
        works = {"protein": protein_works(), "qubit": [openalex_work(f"Qubit entanglement circuit {i}",
                                                                     abstract="qubit entanglement circuit photon gate", topic=None)
                                                       for i in range(6)]}
        route = lambda m, url, b: (200, {}, {"results": works["qubit" if "qubit" in url else "protein"]})
        client, transport, _ = client_with({"api.openalex.org/works": route}, order=("openalex",))
        pipe = pipeline(client)
        pipe.search("protein folding structure")
        res = pipe.search("qubit entanglement circuit")
        self.assertEqual(len(transport.urls("works?search=")), 2)
        self.assertIn("Qubit", res["results"][0]["title"])
        self.assertEqual(res["results"][0]["field"], "Quantum Physics")

    def test_internet_failure_still_returns_what_memory_has(self):
        client, _, _ = client_with({"openalex": timeout, "semanticscholar": timeout, "crossref": timeout})
        pipe = pipeline(client)
        pipe.index.add(make_corpus(per_topic=5))
        res = pipe.search("graph node embedding", external="always")
        self.assertGreater(res["total"], 0)
        self.assertTrue(all(not p["ok"] for p in res["trace"]["online"]["providers"]))

    def test_identifier_is_fetched_once_then_found_in_memory(self):
        work = openalex_work("Attention is all you need", doi="10.48550/arxiv.1706.03762")
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": [work]})}, order=("openalex",))
        pipe = pipeline(client)
        res = pipe.search("1706.03762")
        self.assertEqual(res["results"][0]["title"], "Attention is all you need")
        res = pipe.search("10.48550/arXiv.1706.03762")
        self.assertTrue(res["trace"]["identifier"]["in_memory"])
        self.assertEqual(len(transport.calls), 1)


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.pipe = pipeline(None)
        self.pipe.index.add(make_corpus(per_topic=15))

    def test_year_author_and_exclusion_filters(self):
        res = self.pipe.search("graph node embedding", year_from=2015, year_to=2018, external="never")
        self.assertTrue(res["results"] and all(2015 <= p["year"] <= 2018 for p in res["results"]))
        res = self.pipe.search("graph node", author="author3", external="never")
        self.assertTrue(all("author3" in p["authors"].lower() for p in res["results"]))
        res = self.pipe.search("graph -community", external="never")
        self.assertTrue(all("communit" not in (p["title"] + p["abstract"]).lower() for p in res["results"]))

    def test_boolean_query_uses_the_stack(self):
        res = self.pipe.search("graph AND (node OR vertex) NOT walk", external="never")
        self.assertIn("postfix", res["trace"]["boolean"])
        for p in res["results"]:
            text = (p["title"] + " " + p["abstract"]).lower()
            self.assertIn("graph", text)
            self.assertTrue("node" in text or "vertex" in text)
            self.assertNotIn("walk", text)

    def test_field_can_be_chosen_or_enforced(self):
        res = self.pipe.search("structure detection model", field="Computer Vision", external="never")
        self.assertEqual(res["trace"]["field"]["chosen"], "Computer Vision")
        res = self.pipe.search("structure detection model", field="Computer Vision", strict_field=True, external="never")
        self.assertTrue(res["results"] and all(p["field"] == "Computer Vision" for p in res["results"]))

    def test_sort_by_year_puts_on_topic_papers_first(self):
        res = self.pipe.search("graph neural network", sort_by="year", per_page=50, external="never")
        n = res["trace"]["sorted_on_topic"]
        on, rest = res["results"][:n], res["results"][n:]
        years = [p["year"] for p in on]
        self.assertEqual(years, sorted(years, reverse=True))           # stable merge sort by year
        self.assertTrue(on and rest)
        self.assertGreater(min(p["match"] for p in on), max(p["match"] for p in rest))
        best = self.pipe.search("graph neural network", per_page=50, external="never")["results"]
        self.assertEqual([p["key"] for p in rest], [p["key"] for p in best if p["match"] < min(q["match"] for q in on)])

    def test_match_counts_words_and_meaning_not_citations(self):
        res = self.pipe.search("graph neural network", per_page=50, external="never")
        for p in res["results"]:
            f = p["features"]
            want = 0.45 * f["term_coverage"] + 0.2 * f["title_coverage"] + 0.35 * f["semantic"]
            self.assertAlmostEqual(p["match"], want, places=2)
            self.assertEqual(p["match_level"], "strong" if p["match"] >= 0.75 else "good" if p["match"] >= 0.5 else "partial")

    def test_suggest(self):
        self.assertIn("graph", self.pipe.suggest("gra"))


class MemoryBoundTests(unittest.TestCase):
    def test_capacity_bounds_memory_and_evicted_papers_come_back_online(self):
        works = protein_works(8)
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        pipe = pipeline(client, capacity=5)
        res = pipe.search("protein folding structure")
        self.assertEqual(pipe.state()["papers_in_memory"], 5)
        key = res["results"][0]["key"]
        self.assertIsNotNone(pipe.paper(key))
        evicted = [f"doi:10.1/pf{i}" for i in range(8) if pipe.index.find(f"doi:10.1/pf{i}") is None]
        self.assertEqual(len(evicted), 3)
        before = len(transport.calls)
        self.assertIsNotNone(pipe.paper(evicted[0]))                     # fetched again by its DOI
        self.assertEqual(len(transport.calls), before + 1)


if __name__ == "__main__":
    unittest.main()
