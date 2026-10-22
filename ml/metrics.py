"""Ranking metrics for one query.

`grades` are the relevance labels of the results in ranked order (0 = not
relevant); `n_relevant` is how many relevant papers exist in total, found or not
(for recall, and for the ideal ranking in NDCG). Papers nobody judged count as
not relevant, the usual convention.
"""
import numpy as np


def dcg(grades, k):
    g = np.asarray(grades, np.float64)[:k]
    return float(((2.0 ** g - 1.0) / np.log2(np.arange(2, len(g) + 2))).sum())


def ndcg_at_k(grades, k=10, n_relevant=None, ideal_grades=None):
    """Discounted cumulative gain of the top k divided by the best possible: 1.0 means
    every relevant paper is at the top. The ideal list is `ideal_grades` if given,
    else `n_relevant` papers of grade 1 (else the relevant papers among `grades`)."""
    if ideal_grades is None:
        n = int((np.asarray(grades) > 0).sum()) if n_relevant is None else int(n_relevant)
        ideal_grades = np.ones(n)
    ideal = dcg(np.sort(np.asarray(ideal_grades, np.float64))[::-1], k)
    return dcg(grades, k) / ideal if ideal else 0.0


def reciprocal_rank(grades):
    """1 / rank of the first relevant result (0 if none): the MRR of one query."""
    hits = np.flatnonzero(np.asarray(grades) > 0)
    return 1.0 / (hits[0] + 1) if len(hits) else 0.0


def precision_at_k(grades, k):
    """Share of the top k that is relevant; divides by k even when fewer came back."""
    return float((np.asarray(grades)[:k] > 0).sum()) / k


def recall_at_k(grades, k, n_relevant):
    return float((np.asarray(grades)[:k] > 0).sum()) / n_relevant if n_relevant else 0.0


METRICS = ("NDCG@10", "MRR", "P@10", "R@50")


def query_metrics(scores, labels, n_relevant):
    """All metrics for one query, ranking its candidates by `scores` (ties keep input order)."""
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    g = np.asarray(labels)[order]
    n_rel = max(int(n_relevant), int((g > 0).sum()))
    ideal = np.sort(np.asarray(labels, np.float64))[::-1]
    if n_rel > (ideal > 0).sum():                       # relevant papers the candidates missed count as grade 1
        ideal = np.concatenate([ideal[ideal > 0], np.ones(n_rel - int((ideal > 0).sum()))])
    return {"NDCG@10": ndcg_at_k(g, 10, ideal_grades=ideal), "MRR": reciprocal_rank(g),
            "P@10": precision_at_k(g, 10), "R@50": recall_at_k(g, 50, n_rel)}


def mean_metrics(rows):
    if not rows:
        return dict.fromkeys(METRICS, 0.0)
    return {m: round(float(np.mean([r[m] for r in rows])), 4) for m in rows[0]}
