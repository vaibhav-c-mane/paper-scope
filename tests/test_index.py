"""The in-memory index: inverted index (BM25), vectors, deduplication, LRU eviction, compaction."""
import math
import os
import tempfile
import unittest

import numpy as np

from engine.domains import DOMAINS
from engine.lexicon import Lexicon
from engine.memindex import PaperIndex
from engine.text import terms
from tests.fixtures import lexicon, make_corpus


def reference_bm25(lex, paper, words):
    """BM25 computed directly from the definition, for comparison."""
    counts = {}
    for t in terms(paper["title"]):
        counts[t] = counts.get(t, 0) + lex.title_weight
    for t in terms(paper["abstract"]):
        counts[t] = counts.get(t, 0) + 1
    dl = sum(counts.values())
    score = 0.0
    for w in set(words):
        tf = counts.get(w, 0)
        if tf:
            i = lex.term_id(w)
            idf = float(lex.idf[i]) if i >= 0 else lex.idf_unknown
            score += tf * (lex.k1 + 1) / (tf + lex.k1 * (1 - lex.b + lex.b * dl / lex.avgdl)) * idf
    return score


class LexiconTests(unittest.TestCase):
    def test_vocabulary_is_sorted_and_searchable(self):
        lex = lexicon()
        self.assertEqual(lex.vocab, sorted(lex.vocab))
        self.assertGreaterEqual(lex.term_id("graph"), 0)
        self.assertEqual(lex.term_id("zebra"), -1)
        lo, hi = lex.prefix_range("qu")
        self.assertTrue(all(w.startswith("qu") for w in lex.vocab[lo:hi]) and hi > lo)

    def test_embeddings_and_field_classifier(self):
        lex = lexicon()
        vec, p = lex.text_features([([], terms("qubit entanglement circuit")), ([], terms("image pixel segmentation"))])
        self.assertAlmostEqual(float(np.linalg.norm(vec[0])), 1.0, places=4)
        self.assertEqual(DOMAINS[int(p[0].argmax())], "Quantum Physics")
        self.assertEqual(DOMAINS[int(p[1].argmax())], "Computer Vision")
        self.assertLess(float(vec[0] @ vec[1]), 0.5)

    def test_save_and_load_round_trip(self):
        lex = lexicon()
        folder = tempfile.mkdtemp()
        lex.save(folder)
        again = Lexicon.load(folder)
        self.assertEqual(again.vocab, lex.vocab)
        x = [([], terms("protein folding residue"))]
        np.testing.assert_allclose(again.text_features(x)[0], lex.text_features(x)[0], atol=2e-3)   # float16 LSA
        self.assertTrue(set(os.listdir(folder)) >= {"vocab.txt", "df.npy", "lsa.npy", "lexicon.json"})


class PaperIndexTests(unittest.TestCase):
    def setUp(self):
        self.lex = lexicon()
        self.papers = make_corpus(per_topic=10, seed=5)

    def test_bm25_matches_the_definition(self):
        idx = PaperIndex(self.lex, capacity=0)
        idx.add(self.papers)
        words = terms("graph node embedding protein")
        ids, w = idx.query_ids(words)
        got = idx.bm25(ids, w)
        for d, p in enumerate(self.papers):
            self.assertAlmostEqual(float(got[d]), reference_bm25(self.lex, p, words), places=3)

    def test_unknown_words_are_indexed_with_their_own_ids(self):
        idx = PaperIndex(self.lex, capacity=0)
        idx.add([{"title": "Zebrafish regeneration", "abstract": "zebrafish fins regrow"}])
        self.assertEqual(idx.docs_with("zebrafish").tolist(), [0])
        self.assertEqual(idx.docs_with("zebra*").tolist(), [0])
        ids, w = idx.query_ids(["zebrafish"])
        self.assertGreater(float(idx.bm25(ids, w)[0]), 0)

    def test_coverage(self):
        idx = PaperIndex(self.lex, capacity=0)
        idx.add([{"title": "Graph walks", "abstract": "node community"}, {"title": "Pixel", "abstract": "graph"}])
        groups = [idx.expand(w) for w in ("graph", "node", "pixel")]
        title, anywhere = idx.coverage(groups, [0, 1])
        np.testing.assert_allclose(title, [1 / 3, 1 / 3])
        np.testing.assert_allclose(anywhere, [2 / 3, 2 / 3])

    def test_vector_search_equals_brute_force(self):
        idx = PaperIndex(self.lex, capacity=0)
        idx.add(self.papers)
        q = self.lex.text_features([([], terms("quantum qubit photon"))])[0][0]
        ids, sims = idx.search_vectors(q, 5)
        best = np.argsort(-(idx.vectors.view @ q))[:5]
        self.assertEqual(sorted(ids.tolist()), sorted(best.tolist()))

    def test_duplicates_merge_and_fill_missing_metadata(self):
        idx = PaperIndex(self.lex, capacity=0)
        a = {"title": "Graph walks", "abstract": "node", "arxiv_id": "2101.00001", "citations": None}
        b = {"title": "Graph Walks!", "abstract": "a longer abstract about node embedding", "doi": "10.48550/arxiv.2101.00001",
             "citations": 42, "venue": "Journal"}
        first = idx.add([a])
        second = idx.add([b, b])
        self.assertEqual(first + [first[0]], second[:1] + [first[0]])
        self.assertEqual(len(idx), 1)
        self.assertEqual(idx.paper(0)["citations"], 42)
        self.assertEqual(idx.meta.view[0, 1], 42)                       # metadata column updated too
        self.assertIn("longer", idx.paper(0)["abstract"])

    def test_memory_is_bounded_by_lru_eviction(self):
        idx = PaperIndex(self.lex, capacity=10)
        first = idx.add(self.papers[:10])
        idx.touch(first[0])                                             # recently used: survives
        idx.add(self.papers[10:15])
        self.assertEqual(len(idx), 10)
        self.assertIsNotNone(idx.find("arxiv:" + self.papers[0]["arxiv_id"]))
        self.assertIsNone(idx.find("arxiv:" + self.papers[1]["arxiv_id"]))
        ids, w = idx.query_ids(terms(self.papers[1]["title"]))
        scores = idx.bm25(ids, w)
        self.assertEqual(float(scores[1]), 0.0)                          # evicted papers leave the inverted index

    def test_compaction_renumbers_without_losing_anything(self):
        idx = PaperIndex(self.lex, capacity=12)
        for i in range(0, 40, 4):
            idx.add(self.papers[i:i + 4])
        idx.maintain()
        idx.add([])
        self.assertEqual(len(idx), 12)
        self.assertLessEqual(idx.n, 12 + max(64, idx.n // 4))
        for d in range(idx.n):
            p = idx.paper(d)
            if p is None:
                continue
            self.assertEqual(idx.find(p["_keys"][0]), d)
            ids, w = idx.query_ids(terms(p["title"]))
            self.assertGreater(float(idx.bm25(ids, w)[d]), 0)
        self.assertEqual(len(idx.lru), len(idx))

    def test_heavy_churn_keeps_index_consistent(self):
        idx = PaperIndex(self.lex, capacity=20)
        corpus = make_corpus(per_topic=60, seed=9)
        for i in range(0, len(corpus), 7):
            idx.add(corpus[i:i + 7])
        self.assertEqual(len(idx), 20)
        self.assertLess(idx.n, 20 + 70)                                  # evicted slots get compacted away
        alive = [d for d in range(idx.n) if idx.paper(d) is not None]
        self.assertEqual(sorted(alive), sorted(idx.key_to_doc[k] for k in idx.lru.keys()))
        self.assertTrue(math.isclose(float(idx.alive.view.sum()), 20))

    def test_appended_postings_equal_a_full_rebuild(self):
        """New papers are merged into the inverted index, evicted ones skipped until compaction:
        every read must give what rebuilding the index from scratch gives."""
        idx = PaperIndex(self.lex, capacity=30)
        corpus = make_corpus(per_topic=40, seed=5)
        words = ["graph", "quantum", "protein", "imag*", "zzunseen"]
        most_skipped = 0
        for i in range(0, len(corpus), 9):
            idx.add(corpus[i:i + 9] + [{"title": f"Zzunseen word {i}", "abstract": "graph"}])
            fresh = PaperIndex(self.lex, capacity=0)                     # same papers, built in one go
            fresh.extra = dict(idx.extra)
            fresh.fwd, fresh.papers, fresh.alive = idx.fwd, idx.papers, idx.alive
            for w in words:
                self.assertEqual(idx.docs_with(w).tolist(), fresh.docs_with(w).tolist())
                ids, ws = idx.query_ids([w])
                np.testing.assert_array_equal(idx.bm25(ids, ws), fresh.bm25(ids, ws))
            np.testing.assert_array_equal(idx._df(), fresh._df())
            most_skipped = max(most_skipped, idx._dead_indexed)
        self.assertGreater(most_skipped, 0)                               # evicted papers were in the lists

    def test_year_tree_follows_adds_evictions_and_compaction(self):
        idx = PaperIndex(self.lex, capacity=25)
        corpus = make_corpus(per_topic=40, seed=4)
        for i in range(0, len(corpus), 6):
            idx.add(corpus[i:i + 6])
            years = idx.meta.view[:, 0]
            for lo, hi in [(2000, 2010), (0, 0), (2015, 10 ** 9)]:
                scan = np.flatnonzero(idx.alive.view & (years >= lo) & (years <= hi))
                self.assertEqual(sorted(idx.years_between(lo, hi)), scan.tolist())
        self.assertEqual(idx.by_year.size, len(idx))

    def test_citation_graph_survives_eviction_and_compaction(self):
        idx = PaperIndex(self.lex, capacity=12)
        chain = [{"title": f"Linked paper {i}", "abstract": "graph", "openalex_id": f"W{100 + i}",
                  "refs": [100 + i - 1] if i else []} for i in range(150)]
        compacted = False
        for i in range(0, 150, 5):
            before = idx.n
            idx.add(chain[i:i + 5])
            compacted |= idx.n < before
            for d in range(idx.n):
                p = idx.paper(d)
                if p is None:
                    continue
                i_ = int(p["title"].split()[-1])
                want = {f"Linked paper {j}" for j in (i_ - 1, i_ + 1) if idx.find(f"openalex:w{100 + j}") is not None}
                self.assertEqual({idx.paper(c)["title"] for c in idx.neighbours(d)}, want)
        self.assertEqual(len(idx), 12)
        self.assertTrue(compacted)
        last = idx.find("openalex:w249")
        self.assertEqual([(idx.paper(r)["title"], h) for r, h, _ in idx.citation_links(last, max_depth=3)],
                         [("Linked paper 148", 1), ("Linked paper 147", 2), ("Linked paper 146", 3)])
        self.assertNotIn("refs", idx.paper(last))

    def test_hash_map_and_binary_search_agree(self):
        import bisect
        for w in self.lex.vocab[::7] + ["zebra", "", "graphs"]:
            i = bisect.bisect_left(self.lex.vocab, w)
            expected = i if i < len(self.lex.vocab) and self.lex.vocab[i] == w else -1
            self.assertEqual(self.lex.term_id(w), expected)


if __name__ == "__main__":
    unittest.main()
