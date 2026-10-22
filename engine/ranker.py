"""The ranker: turns the nine features of each candidate into a score.

* GBDTRanker - gradient-boosted regression trees trained with LambdaMART
  (ml/lambdamart.py) on real relevance labels. Trees can learn interactions a
  weighted sum cannot, e.g. "citations matter only once the paper is on topic",
  or "unknown venue (-1) is not the same as a preprint (0)".
* LinearRanker - one weight per feature (pairwise logistic regression); kept as
  the baseline the trees have to beat.
* HeuristicRanker - hand-set weights, used until a model has been trained.

Every model is stored as plain JSON (no pickle): readable, safe to load, and
independent of library versions. score() gives the raw ranking score;
relevance() maps it to 0..1 with Platt scaling, and `threshold` on that scale
decides what counts as a confident local match (fewer than 3 -> search online).
"""
import json
import os

import numpy as np

from engine.features import FEATURES, RELEVANCE_FEATURES, metadata_prior


class _Calibrated:
    platt = (1.0, 0.0)
    threshold = 0.5

    def relevance(self, x, score=None):
        """0..1 from the raw score; pass `score` when it is already computed, to avoid scoring twice."""
        a, b = self.platt
        s = self.score(x) if score is None else np.asarray(score, np.float32)
        return 1.0 / (1.0 + np.exp(-np.clip(a * s + b, -50, 50)))


class HeuristicRanker(_Calibrated):
    kind = "heuristic"
    WEIGHTS = {"bm25": 0.30, "semantic": 0.30, "title_coverage": 0.15, "term_coverage": 0.10, "domain_match": 0.15}
    PRIOR_WEIGHT = 0.25

    def __init__(self, use_prior=True):
        self.features = list(FEATURES)
        self.w = np.asarray([self.WEIGHTS[f] for f in RELEVANCE_FEATURES], np.float32)
        self.use_prior = use_prior
        self.threshold = 0.45
        self.meta = {}

    def score(self, x):
        x = np.asarray(x, np.float32)
        rel = np.clip(x[:, :len(RELEVANCE_FEATURES)] @ self.w, 0.0, 1.0)
        return rel * (1.0 + self.PRIOR_WEIGHT * metadata_prior(x)) if self.use_prior else rel

    def relevance(self, x, score=None):
        return np.clip(np.asarray(x, np.float32)[:, :len(RELEVANCE_FEATURES)] @ self.w, 0.0, 1.0)

    def describe(self):
        return {"kind": self.kind, "weights": dict(zip(RELEVANCE_FEATURES, self.w.round(3).tolist())),
                "prior_weight": self.PRIOR_WEIGHT if self.use_prior else 0}


class LinearRanker(_Calibrated):
    kind = "linear"

    def __init__(self, weights, mean, std, platt=(1.0, 0.0), threshold=0.5, meta=None):
        self.features = list(FEATURES)
        self.w = np.asarray(weights, np.float32)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.where(np.asarray(std, np.float32) == 0, 1.0, np.asarray(std, np.float32))
        self.platt, self.threshold, self.meta = tuple(platt), threshold, meta or {}

    def score(self, x):
        return ((np.asarray(x, np.float32) - self.mean) / self.std) @ self.w

    def describe(self):
        return {"kind": self.kind, "weights": dict(zip(self.features, self.w.round(3).tolist()))}

    def to_json(self):
        return {"kind": self.kind, "features": self.features, "weights": self.w.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "platt": list(self.platt), "threshold": self.threshold, "meta": self.meta}

    @classmethod
    def from_json(cls, d):
        return cls(d["weights"], d["mean"], d["std"], d["platt"], d["threshold"], d.get("meta"))


class GBDTRanker(_Calibrated):
    """Sum of regression trees. All trees are packed into flat arrays (node feature,
    threshold, left child, right child, leaf value: the array representation of a
    binary tree). Scoring walks every (candidate, tree) pair down its tree at once,
    with numpy, keeping a shrinking array of the pairs that have not reached a leaf
    yet: each pair costs one step per node on its own path (about 4.5 on average),
    not one step per level of the deepest tree (up to 19)."""
    kind = "gbdt"

    def __init__(self, trees, base=0.0, platt=(1.0, 0.0), threshold=0.5, meta=None, importance=None):
        self.features = list(FEATURES)
        self.trees = trees                     # list of dicts: feature, threshold, left, right, value
        self.base = base
        self.platt, self.threshold, self.meta = tuple(platt), threshold, meta or {}
        self.importance = importance or {}
        self._pack()

    def _pack(self):
        feat, thr, left, right, val, roots = [], [], [], [], [], []
        for t in self.trees:
            off = len(feat)
            roots.append(off)
            feat += t["feature"]
            thr += t["threshold"]
            left += [c + off if c >= 0 else -1 for c in t["left"]]
            right += [c + off if c >= 0 else -1 for c in t["right"]]
            val += t["value"]
        self._feat = np.asarray(feat, np.int64)
        self._thr = np.asarray(thr, np.float64)          # float64: exactly the thresholds scikit-learn learned
        self._left = np.asarray(left, np.int64)
        self._right = np.asarray(right, np.int64)
        self._val = np.asarray(val, np.float32)
        self._roots = np.asarray(roots, np.int64)

    def score(self, x):
        x = np.asarray(x, np.float32)
        n, n_trees = len(x), len(self._roots)
        if not n_trees or not n:
            return np.full(n, self.base, np.float32)
        node = np.tile(self._roots, n)                 # current node of every (candidate, tree) pair
        cell = np.repeat(np.arange(n) * x.shape[1], n_trees)   # where each pair's candidate row starts in x
        flat = x.ravel()
        active = np.flatnonzero(self._left[node] >= 0)  # pairs not at a leaf yet
        while len(active):
            at = node[active]
            go_left = flat[cell[active] + self._feat[at]] <= self._thr[at]
            node[active] = nxt = np.where(go_left, self._left[at], self._right[at])
            active = active[self._left[nxt] >= 0]      # pairs that reached a leaf drop out
        return self.base + self._val[node].reshape(n, n_trees).sum(axis=1)

    def describe(self):
        return {"kind": self.kind, "trees": len(self.trees), "importance": self.importance,
                "trained": self.meta.get("trained_at"), "queries": self.meta.get("train_queries")}

    def to_json(self):
        return {"kind": self.kind, "features": self.features, "base": self.base, "platt": list(self.platt),
                "threshold": self.threshold, "importance": self.importance, "meta": self.meta, "trees": self.trees}

    @classmethod
    def from_json(cls, d):
        return cls(d["trees"], d.get("base", 0.0), d["platt"], d["threshold"], d.get("meta"), d.get("importance"))


KINDS = {"gbdt": GBDTRanker, "linear": LinearRanker}


def save_ranker(ranker, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranker.to_json(), f, separators=(",", ":"))


def load_ranker(path, fallback=True):
    """The model in `path`, or the hand-set HeuristicRanker if there is none (or it was
    trained on a different feature list)."""
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("features") != FEATURES:
                raise ValueError("feature list changed since this model was trained; run python -m ml.train")
            return KINDS[d["kind"]].from_json(d)
        except (ValueError, KeyError, OSError):
            if not fallback:
                raise
    return HeuristicRanker()
