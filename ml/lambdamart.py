"""LambdaMART: gradient-boosted regression trees that optimise a ranking metric (NDCG).

Each boosting round:
1. For every query, look at each pair (relevant paper i, less relevant paper j).
   If swapping them in the current ranking would change NDCG by |dNDCG|, the
   pair pushes i up and j down with strength rho * |dNDCG|, where
   rho = 1 / (1 + exp(s_i - s_j)) is large when the model currently ranks the
   pair the wrong way round. These pushes are the "lambdas" (the gradient of a
   smoothed NDCG); rho * (1 - rho) * |dNDCG| is the matching second derivative.
2. Fit a regression tree (scikit-learn) to the lambdas: it groups papers whose
   features call for the same push.
3. Each leaf gets the Newton step sum(lambda) / sum(second derivative), shrunk
   by the learning rate, and is added to every paper's score.
Training stops when NDCG@10 on the validation queries has not improved for
`patience` rounds, and keeps the best number of trees.

The trees are exported as plain lists (feature, threshold, children, value) and
scored by engine/ranker.GBDTRanker, so the search engine needs no scikit-learn
model files at run time.
"""
import time

import numpy as np

from ml.metrics import ndcg_at_k


class _Query:
    __slots__ = ("lo", "hi", "pos", "gain", "idcg")

    def __init__(self, lo, hi, y):
        self.lo, self.hi = lo, hi
        g = y[lo:hi]
        self.pos = np.flatnonzero(g > 0)
        self.gain = 2.0 ** g - 1.0
        ideal = np.sort(self.gain)[::-1]
        self.idcg = float((ideal / np.log2(np.arange(2, len(ideal) + 2))).sum())


def _lambdas(scores, y, queries):
    lam = np.zeros(len(scores))
    hess = np.zeros(len(scores))
    for q in queries:
        if not len(q.pos) or q.idcg <= 0:
            continue
        s = scores[q.lo:q.hi]
        g = y[q.lo:q.hi]
        rank = np.empty(len(s), np.int64)
        rank[np.argsort(-s, kind="stable")] = np.arange(len(s))
        disc = 1.0 / np.log2(rank + 2.0)
        p = q.pos
        better = g[p][:, None] > g[None, :]                        # pairs (i relevant, j less relevant)
        delta = np.abs(q.gain[p][:, None] - q.gain[None, :]) * np.abs(disc[p][:, None] - disc[None, :]) / q.idcg
        rho = 1.0 / (1.0 + np.exp(np.clip(s[p][:, None] - s[None, :], -50, 50)))
        push = np.where(better, rho * delta, 0.0)
        curv = np.where(better, rho * (1.0 - rho) * delta, 0.0)
        l, h = np.zeros(len(s)), np.zeros(len(s))
        l[p] += push.sum(axis=1)
        l -= push.sum(axis=0)
        h[p] += curv.sum(axis=1)
        h += curv.sum(axis=0)
        lam[q.lo:q.hi] = l
        hess[q.lo:q.hi] = h
    return lam, hess


def _groups(sizes):
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    return offsets


def mean_ndcg(scores, y, offsets, k=10):
    vals = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        g = y[lo:hi]
        if (g > 0).any():
            order = np.argsort(-scores[lo:hi], kind="stable")
            vals.append(ndcg_at_k(g[order], k, ideal_grades=g))
    return float(np.mean(vals)) if vals else 0.0


def train_lambdamart(X, y, sizes, X_val=None, y_val=None, sizes_val=None, n_trees=400, learning_rate=0.08,
                     max_leaf_nodes=31, min_samples_leaf=100, subsample=0.6, patience=60, seed=0, progress=print):
    """Returns (trees as JSON-ready dicts, info). `sizes` = candidates per query, rows grouped by query."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float64)
    offsets = _groups(sizes)
    queries = [_Query(lo, hi, y) for lo, hi in zip(offsets[:-1], offsets[1:])]
    scores = np.zeros(len(y))
    has_val = X_val is not None and len(X_val)
    if has_val:
        X_val = np.asarray(X_val, np.float32)
        y_val = np.asarray(y_val, np.float64)
        off_val = _groups(sizes_val)
        s_val = np.zeros(len(y_val))
    trees, history, best, best_n = [], [], -1.0, 0
    importance = np.zeros(X.shape[1])
    t0 = time.time()
    for it in range(n_trees):
        lam, hess = _lambdas(scores, y, queries)
        # stochastic boosting: each tree sees a random share of the queries (rows stay grouped)
        keep_q = rng.random(len(queries)) < subsample
        rows = np.concatenate([np.arange(q.lo, q.hi) for q, k in zip(queries, keep_q) if k]) \
            if keep_q.any() else np.arange(len(y))
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                     random_state=int(rng.integers(1 << 30)))
        tree.fit(X[rows], lam[rows])
        leaf_all = tree.apply(X)
        t = tree.tree_
        value = np.zeros(t.node_count)
        num = np.bincount(leaf_all[rows], weights=lam[rows], minlength=t.node_count)
        den = np.bincount(leaf_all[rows], weights=hess[rows], minlength=t.node_count)
        leaves = t.children_left < 0
        value[leaves] = learning_rate * num[leaves] / (den[leaves] + 1e-9)
        scores += value[leaf_all]
        importance += tree.feature_importances_
        trees.append({"feature": [int(f) if f >= 0 else 0 for f in t.feature],
                      "threshold": [float(v) for v in t.threshold],
                      "left": [int(c) for c in t.children_left], "right": [int(c) for c in t.children_right],
                      "value": [float(v) for v in value], "depth": int(tree.get_depth())})
        if has_val:
            s_val += value[tree.apply(X_val)]
            score = mean_ndcg(s_val, y_val, off_val)
        else:
            score = mean_ndcg(scores, y, offsets)
        history.append(round(score, 5))
        if score > best + 1e-5:
            best, best_n = score, it + 1
        if progress and (it + 1) % 25 == 0:
            progress(f"  tree {it + 1}: validation NDCG@10 {score:.4f} (best {best:.4f} at {best_n}) "
                     f"{time.time() - t0:.0f}s")
        if it + 1 - best_n >= patience:
            break
    trees = trees[:best_n]
    imp = importance / (importance.sum() or 1.0)
    return trees, {"trees": len(trees), "rounds_run": len(history), "best_validation_ndcg10": round(best, 4),
                   "history": history, "importance": imp.tolist(), "learning_rate": learning_rate,
                   "max_leaf_nodes": max_leaf_nodes, "min_samples_leaf": min_samples_leaf, "subsample": subsample}


def train_linear(X, y, sizes, negatives=50, seed=0):
    """Pairwise logistic regression (a linear RankNet): learn w so that w.(x_i - x_j) > 0
    for relevant i and non-relevant j of the same query; every query weighs the same.
    Returns (weights, mean, std) on standardised features."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float64)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    diffs, weights = [], []
    for lo, hi in zip(_groups(sizes)[:-1], _groups(sizes)[1:]):
        g = y[lo:hi]
        pos, neg = np.flatnonzero(g > 0), np.flatnonzero(g == 0)
        if not len(pos) or not len(neg):
            continue
        neg = rng.choice(neg, min(len(neg), negatives), replace=False)
        d = (Z[lo + pos][:, None, :] - Z[lo + neg][None, :, :]).reshape(-1, Z.shape[1])
        diffs.append(d)
        weights.append(np.full(len(d), 1.0 / len(d)))
    D = np.vstack(diffs)
    w = np.concatenate(weights)
    Xp = np.vstack([D, -D])
    yp = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000).fit(Xp, yp, sample_weight=np.concatenate([w, w]))
    return clf.coef_[0], mean, std


def platt(scores, labels):
    """(a, b) so that 1 / (1 + exp(-(a * score + b))) is the probability of relevance."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(scores, np.float64).reshape(-1, 1)
    clf = LogisticRegression(C=100.0, max_iter=1000).fit(s, (np.asarray(labels) > 0).astype(int))
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def confident_threshold(prob, labels, precision=0.5, min_count=50):
    """Lowest probability t such that the candidates with prob >= t are relevant at least
    `precision` of the time (on validation data). Above t a local match counts as confident."""
    prob = np.asarray(prob, np.float64)
    order = np.argsort(-prob, kind="stable")
    p = prob[order]
    rel = (np.asarray(labels)[order] > 0).astype(np.float64)
    prec = np.cumsum(rel) / np.arange(1, len(rel) + 1)
    group_end = np.append(p[1:] < p[:-1], True)          # equal probabilities stand or fall together
    ok = np.flatnonzero(group_end & (prec >= precision) & (np.arange(1, len(rel) + 1) >= min_count))
    if not len(ok):
        return float(p[0]) if len(p) else 0.5
    return float(p[ok[-1]])
