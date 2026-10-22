"""Learning to rank: metrics, LambdaMART, the exported trees, calibration, labelled data."""
import math
import os
import tempfile
import unittest

import numpy as np

from engine.domains import weak_label
from engine.features import FEATURES
from engine.metadata import citation_score, recency_score, venue_quality_from_source
from engine.ranker import GBDTRanker, HeuristicRanker, LinearRanker, load_ranker, save_ranker
from ml import dataset
from ml.lambdamart import confident_threshold, mean_ndcg, platt, train_lambdamart, train_linear
from ml.metrics import ndcg_at_k, query_metrics, reciprocal_rank
from tests.fixtures import client_with, lexicon, make_corpus, openalex_work


def xor_queries(n_queries, per_query=40, seed=0):
    """Relevance needs BOTH features 0 and 5 high, or both low: an interaction a linear
    model cannot represent (think "citations only matter for on-topic papers")."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_queries * per_query, len(FEATURES))).astype(np.float32)
    y = (((X[:, 0] > 0.5) == (X[:, 5] > 0.5)) & (X[:, 1] > 0.6)).astype(np.float32)
    return X, y, [per_query] * n_queries


def walk(tree, x):
    node = 0
    while tree["left"][node] >= 0:
        node = tree["left"][node] if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
    return tree["value"][node]


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(reciprocal_rank([0, 0, 1, 1]), 1 / 3)
        self.assertAlmostEqual(ndcg_at_k([1, 0, 0], 10), 1.0)
        self.assertAlmostEqual(ndcg_at_k([0, 1], 10), (1 / math.log2(3)) / 1.0)
        self.assertAlmostEqual(ndcg_at_k([1, 0], 10, n_relevant=2), 1 / (1 + 1 / math.log2(3)))
        m = query_metrics([0.9, 0.1, 0.5], [0, 1, 0], n_relevant=2)       # one relevant paper never retrieved
        self.assertEqual(m["MRR"], 1 / 3)
        self.assertEqual(m["R@50"], 0.5)


class LambdaMARTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.sizes = xor_queries(120, seed=1)
        cls.Xv, cls.yv, cls.sv = xor_queries(40, seed=2)
        cls.trees, cls.info = train_lambdamart(cls.X, cls.y, cls.sizes, cls.Xv, cls.yv, cls.sv, n_trees=120,
                                               min_samples_leaf=20, patience=40, progress=None)

    def test_learns_an_interaction_a_linear_model_cannot(self):
        gbdt = GBDTRanker(self.trees)
        w, mean, std = train_linear(self.X, self.y, self.sizes)
        linear = LinearRanker(w, mean, std)
        off = np.concatenate([[0], np.cumsum(self.sv)])
        g, l = mean_ndcg(gbdt.score(self.Xv), self.yv, off), mean_ndcg(linear.score(self.Xv), self.yv, off)
        self.assertGreater(g, 0.85)
        self.assertGreater(g, l + 0.1)
        imp = self.info["importance"]
        self.assertEqual(set(np.argsort(imp)[-3:]), {0, 1, 5})             # the features that matter

    def test_packed_scoring_equals_walking_each_tree(self):
        gbdt = GBDTRanker(self.trees, base=0.25)
        x = self.Xv[:50]
        expect = [0.25 + sum(walk(t, row) for t in self.trees) for row in x]
        np.testing.assert_allclose(gbdt.score(x), expect, rtol=1e-5, atol=1e-6)

    def test_save_and_load(self):
        gbdt = GBDTRanker(self.trees, platt=(2.0, -1.0), threshold=0.4)
        path = os.path.join(tempfile.mkdtemp(), "ranker.json")
        save_ranker(gbdt, path)
        again = load_ranker(path, fallback=False)
        np.testing.assert_allclose(again.score(self.Xv), gbdt.score(self.Xv), rtol=1e-6)
        self.assertEqual(again.threshold, 0.4)
        self.assertIsInstance(load_ranker(os.path.join(tempfile.mkdtemp(), "none.json")), HeuristicRanker)

    def test_calibration_and_threshold(self):
        gbdt = GBDTRanker(self.trees)
        gbdt.platt = platt(gbdt.score(self.Xv), self.yv)
        p = gbdt.relevance(self.Xv)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(p[self.yv > 0].mean(), p[self.yv == 0].mean() + 0.3)
        t = confident_threshold(p, self.yv, precision=0.5)
        self.assertGreaterEqual((self.yv[p >= t] > 0).mean(), 0.5)


class SignalTests(unittest.TestCase):
    def test_metadata_scores(self):
        self.assertAlmostEqual(float(recency_score([2024], 2024)[0]), 1.0)
        self.assertAlmostEqual(float(recency_score([2016], 2024)[0]), math.exp(-1), places=5)
        self.assertEqual(float(recency_score([0], 2024)[0]), -1.0)                    # unknown year
        self.assertEqual(float(citation_score([-1], [2020], 2024)[0]), -1.0)           # unknown citations
        self.assertGreater(float(citation_score([200], [2022], 2024)[0]), float(citation_score([300], [2004], 2024)[0]))
        self.assertEqual(venue_quality_from_source({"type": "journal", "listed_in": ["ccf-a"]}, True), 1.0)
        self.assertEqual(venue_quality_from_source({"type": "repository"}, False), 0.0)

    def test_weak_field_labels(self):
        self.assertEqual(weak_label({"categories": "cs.LG cs.CV"}), "Computer Vision")   # specific beats generic
        self.assertEqual(weak_label({"categories": "quant-ph"}), "Quantum Physics")
        self.assertEqual(weak_label({"topic": "Protein Structure | Molecular Biology | Biochemistry, Genetics and Molecular Biology"}),
                         "Biology & Medicine")


class DatasetTests(unittest.TestCase):
    def test_citation_labels_from_a_pool(self):
        papers = make_corpus(per_topic=10, seed=4)
        for i, p in enumerate(papers):
            p["openalex_id"] = f"https://openalex.org/W{1000 + i}"
            p["refs"] = []
        seed = papers[0]
        seed["year"] = 2023
        seed["refs"] = [1001, 1002, 1003, 99999]                        # three in the pool, one not
        pool = dataset.Pool(papers, lexicon(), progress=None)
        self.assertEqual(pool.cited(0), {1, 2, 3})
        self.assertEqual(pool.seeds(min_refs=3), [0])
        q = dataset.pool_queries(pool, [0], progress=None)[0]
        self.assertEqual(q.n_relevant, 3)
        self.assertEqual(q.x.shape[1], len(FEATURES))
        self.assertLessEqual(int(q.y.sum()), 3)
        self.assertEqual(len(q.y), len(q.bm25))

    def test_online_labels_use_the_same_search(self):
        seed = openalex_work("Protein folding structure survey", year=2022, refs=[1, 2])
        cands = [openalex_work(f"Protein folding structure {i}", year=2015) for i in range(4)]
        cands[0]["id"], cands[1]["id"] = "https://openalex.org/W1", "https://openalex.org/W5"

        def route(method, url, body):
            return 200, {}, {"results": [seed] if "sample=" in url else cands}
        client, transport, _ = client_with({"api.openalex.org/works": route}, order=("openalex",))
        qs = dataset.online_queries(client, lexicon(), n_queries=1, progress=None)
        self.assertEqual(len(qs), 1)
        self.assertEqual(int(qs[0].y.sum()), 1)                         # W1 is cited, W5 is not
        self.assertEqual(qs[0].n_relevant, 2)
        self.assertIn("publication_year%3A1900-2022", transport.calls[1]["url"])   # searched as of the seed's year

    def test_split_is_by_query(self):
        train, valid, test = dataset.split(list(range(100)), seed=1)
        self.assertEqual((len(train), len(valid), len(test)), (70, 10, 20))
        self.assertFalse(set(train) & set(test))


if __name__ == "__main__":
    unittest.main()
