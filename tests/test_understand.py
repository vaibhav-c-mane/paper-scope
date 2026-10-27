"""Query understanding: spelling, request words, recency, and how the pipeline uses them."""
import unittest

import numpy as np

from engine.pipeline import SearchPipeline, fuse
from engine.ranker import HeuristicRanker
from engine.text import normalize
from engine.understand import QueryUnderstanding
from tests.fixtures import client_with, lexicon, ok, openalex_work


class WordCounts:
    """Stands in for the Lexicon: how many papers use each word (the real one has 600,013 papers)."""

    def __init__(self, counts, n_docs=600_013):
        self.n_docs = n_docs
        words = sorted({normalize(w) for w in counts})
        self.ids = {w: i for i, w in enumerate(words)}
        self.df = np.zeros(len(words), np.int64)
        for w, c in counts.items():
            self.df[self.ids[normalize(w)]] += c

    def term_id(self, word):
        return self.ids.get(word, -1)


COUNTS = {"neural": 20554, "network": 66710, "nueral": 2, "netwrok": 2, "pruning": 931, "learning": 90000,
          "reinforcement": 9000, "teenage": 47, "teenager": 300, "graph": 30000, "protein": 40000}


class SpellingTests(unittest.TestCase):
    def setUp(self):
        self.u = QueryUnderstanding(WordCounts(COUNTS))

    def test_typos_are_corrected_to_common_words(self):
        r = self.u.understand("nueral netwrok pruning", 2026)
        self.assertEqual(r["query"], "neural network pruning")
        self.assertEqual([(c["from"], c["to"]) for c in r["corrections"]], [("nueral", "neural"), ("netwrok", "network")])
        self.assertEqual(self.u.correct("reinforcment"), "reinforcement")      # not in the vocabulary at all
        self.assertEqual(self.u.correct("lerning"), "learning")

    def test_real_words_and_acronyms_are_left_alone(self):
        self.assertIsNone(self.u.correct("teenage"))            # rare (47 papers) but a real word
        self.assertIsNone(self.u.correct("GNN"))                # acronym
        self.assertIsNone(self.u.correct("graph"))
        self.assertIsNone(self.u.correct("the"))                # stop word

    def test_no_corrections_with_a_small_corpus(self):
        u = QueryUnderstanding(WordCounts(COUNTS, n_docs=500))
        self.assertIsNone(u.correct("nueral"))


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.u = QueryUnderstanding(WordCounts(COUNTS, n_docs=500))     # spelling off: tested above

    def test_request_words_are_dropped(self):
        cases = {
            "I want research papers on how to make neural networks smaller": "how to make neural networks smaller",
            "papers about detecting fake news using AI": "detecting fake news using AI",
            "show me the best papers about crispr gene editing": "crispr gene editing",
            "i need studies showing that social media harms teenage mental health": "social media harms teenage mental health",
            "graph neural networks papers": "graph neural networks",
        }
        for q, want in cases.items():
            r = self.u.understand(q, 2026)
            self.assertEqual(r["query"], want, q)
            self.assertEqual(r["kind"], "request", q)
            self.assertTrue(r["semantic"], q)

    def test_topic_words_that_look_like_requests_stay(self):
        for q in ["attention is all you need", "best response dynamics", "operations research scheduling",
                  "paper recycling", "electric current density", "search engines ranking", "top quark mass"]:
            r = self.u.understand(q, 2026)
            self.assertEqual(r["query"], q)
            self.assertFalse(r["changed"], q)

    def test_recent_becomes_a_year_filter(self):
        r = self.u.understand("latest research on quantum computing", 2026)
        self.assertEqual((r["query"], r["recent_from"]), ("quantum computing", 2022))
        r = self.u.understand("what are the latest papers on graph learning", 2026)
        self.assertEqual((r["query"], r["recent_from"], r["kind"]), ("what are the graph learning", 2022, "question"))
        self.assertIsNone(self.u.understand("electric current density", 2026)["recent_from"])

    def test_query_type_decides_the_extra_searches(self):
        title = self.u.understand("attention is all you need", 2026)
        self.assertEqual((title["kind"], title["semantic"], title["title"]), ("keywords", False, True))
        question = self.u.understand("how do vaccines train the immune system", 2026)
        self.assertEqual((question["kind"], question["semantic"], question["title"]), ("question", True, False))
        self.assertEqual(self.u.understand("find me papers", 2026)["query"], "find me papers")   # nothing else to search


class Directions(WordCounts):
    """WordCounts with a 4-d direction per word, standing in for the lexicon's LSA vectors."""
    DIRS = {"reinforcement": (1, 0.2, 0, 0), "learning": (1, 0.1, 0, 0), "autonomous": (0.9, 0.3, 0, 0),
            "vehicle": (0.8, 0.4, 0, 0), "navigation": (0.9, 0.2, 0, 0), "improve": (-0.2, 0, 1, 0), "night": (0, 0.1, 0, 1)}

    def word_vector(self, term):
        v = next((np.asarray(d, np.float32) for w, d in self.DIRS.items() if normalize(w) == term), None)
        return None if v is None else v / np.linalg.norm(v)


class RelaxTests(unittest.TestCase):
    def test_the_least_related_words_are_dropped(self):
        u = QueryUnderstanding(Directions({w: 5000 for w in Directions.DIRS}, n_docs=500))
        r = u.understand("How does reinforcement learning improve autonomous vehicle navigation at night?", 2026)
        self.assertEqual(r["relaxed"], "reinforcement learning autonomous vehicle navigation")
        self.assertEqual(r["relax_dropped"], ["improve", "night"])
        self.assertIsNone(u.understand("reinforcement learning at night", 2026)["relaxed"])    # too short
        self.assertIsNone(QueryUnderstanding(WordCounts(COUNTS)).relax("a b c d e f")[0])       # no vectors


class PipelineTests(unittest.TestCase):
    def test_fusion_of_several_weighted_lists(self):
        rows = [(d, 0.0, 0.0, None) for d in [10, 11, 12, 13]]
        self.assertEqual([r[0] for r in fuse(rows, [[13], [13, 12]])], [13, 12, 10, 11])
        self.assertEqual([r[0] for r in fuse(rows, [[13], [13]], weights=[0.0, 0.0])], [10, 11, 12, 13])

    def test_fusion_rewards_papers_high_in_both_lists(self):
        rows = [(d, 0.0, 0.0, None) for d in [10, 11, 12, 13]]
        fused = [r[0] for r in fuse(rows, [13, 12])]
        self.assertEqual(fused, [13, 12, 10, 11])               # 13: 1/64 + 1/61 beats 12: 1/63 + 1/62
        self.assertEqual([r[0] for r in fuse(rows, [])], [10, 11, 12, 13])

    def _pipeline(self, works):
        client, transport, _ = client_with({"api.openalex.org/works": ok({"results": works})}, order=("openalex",))
        return SearchPipeline(client=client, lexicon=lexicon(), ranker=HeuristicRanker(), now_year=2026), transport

    def test_a_sentence_also_searches_by_meaning(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(6)]
        pipe, transport = self._pipeline(works)
        res = pipe.search("can you find me papers on how proteins fold into their structure")
        self.assertEqual(len(transport.urls("search.semantic=")), 1)
        self.assertEqual(res["trace"]["understood"]["query"], "how proteins fold into their structure")
        self.assertEqual(res["trace"]["fusion"]["with"], "OpenAlex (by meaning)")
        self.assertIn("by meaning", " ".join(p["provider"] for p in res["trace"]["online"]["providers"]))
        self.assertEqual(res["total"], 6)

    def test_exact_title_goes_first_when_well_cited(self):
        works = [openalex_work(f"Protein folding structure {w}", cites=500) for w in ("survey", "methods", "review")]
        works += [openalex_work("Protein folding structure", cites=900, abstract="a short note"),
                  openalex_work("Protein folding structure prediction", cites=3)]
        pipe, transport = self._pipeline(works)
        res = pipe.search("protein folding structure")
        self.assertEqual(res["results"][0]["title"], "Protein folding structure")
        self.assertEqual(res["trace"]["exact_title"], 1)
        self.assertEqual(len(transport.urls("title.search")), 1)

    def test_exact_searches_the_words_as_typed(self):
        works = [openalex_work(f"Protein folding structure paper {i}", doi=f"10.1/pf{i}") for i in range(4)]
        pipe, transport = self._pipeline(works)
        res = pipe.search("papers on protein folding", exact=True)
        self.assertNotIn("understood", res["trace"])
        self.assertEqual(transport.urls("api.openalex.org")[0].split("search=")[1].split("&")[0], "papers+on+protein+folding")
        self.assertEqual(len(transport.calls), 1)


if __name__ == "__main__":
    unittest.main()
