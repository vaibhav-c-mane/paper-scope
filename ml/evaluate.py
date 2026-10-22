"""Evaluation: NDCG@10, MRR, P@10 and R@50 of every ranking system on held-out queries.

    python -m ml.evaluate --pool pool.jsonl        # the same test queries ml/train.py held out
    python -m ml.evaluate --online --queries 300   # fresh queries from the OpenAlex API
    python -m ml.evaluate --qrels my_judgments.tsv # your own relevance judgments

Every system re-ranks the same candidates, so the numbers compare ranking quality:
  bm25      lexical score only (what word matching gives)
  semantic  embedding similarity only
  hybrid    hand-set weights of the relevance features + metadata prior (no training)
  linear    pairwise logistic regression on all nine features (data/models/ranker_linear.json)
  gbdt      LambdaMART gradient-boosted trees on all nine features (data/models/ranker.json)
Also reported: in-field@10, the share of the top 10 that is from the query's
research field (from the query paper's own subject labels).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from engine.domains import DOMAINS                                  # noqa: E402
from engine.pipeline import mode_scores                             # noqa: E402
from engine.ranker import HeuristicRanker, load_ranker              # noqa: E402
from ml.metrics import METRICS, mean_metrics, query_metrics         # noqa: E402


def systems(models_dir):
    """name -> function(query) giving a score per candidate."""
    out = {"bm25": lambda q: q.bm25, "semantic": lambda q: mode_scores("semantic", q.x, None),
           "hybrid": lambda q: HeuristicRanker().score(q.x)}
    for name, file in (("linear", "ranker_linear.json"), ("gbdt", "ranker.json")):
        r = load_ranker(os.path.join(models_dir, file), fallback=True)
        if r.kind == name:
            out[name] = lambda q, r=r: r.score(q.x)
    return out


def evaluate(queries, rankers):
    """{system: {metric: mean}} over the queries, plus in-field@10."""
    report = {}
    for name, score in rankers.items():
        rows, in_field = [], []
        for q in queries:
            s = np.asarray(score(q), np.float64)
            rows.append(query_metrics(s, q.y, q.n_relevant))
            if q.field in DOMAINS:
                top = np.argsort(-s, kind="stable")[:10]
                in_field.append(float((q.cand_fields[top] == DOMAINS.index(q.field)).mean()))
        m = mean_metrics(rows)
        m["in-field@10"] = round(float(np.mean(in_field)), 4) if in_field else None
        report[name] = m
    return report


def candidate_stats(queries):
    found = sum(int((q.y > 0).sum()) for q in queries)
    total = sum(max(q.n_relevant, int((q.y > 0).sum())) for q in queries)
    return {"queries": len(queries), "with_a_relevant_candidate": sum(1 for q in queries if (q.y > 0).any()),
            "candidates_per_query": round(float(np.mean([len(q.y) for q in queries])), 1) if queries else 0,
            "relevant_found_by_candidates": round(found / max(1, total), 4)}


def print_report(report, title="test queries"):
    cols = list(METRICS) + ["in-field@10"]
    print(f"\n{title}")
    print("  " + "system".ljust(10) + "".join(c.rjust(12) for c in cols))
    for name, m in report.items():
        print("  " + name.ljust(10) + "".join(
            (f"{m[c]:.4f}" if m.get(c) is not None else "-").rjust(12) for c in cols))


def main(argv=None):
    from ml import dataset
    from ml.train import add_data_args, labelled_queries, load_lexicon, make_client

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    a = ap.parse_args(argv)
    if not (a.pool or a.online or a.qrels):
        ap.error("choose the queries: --pool FILE, --online or --qrels FILE")
    lex = load_lexicon(a.models)
    if a.pool or a.online:
        test = labelled_queries(a, lex)[2]
        print_report(evaluate(test, systems(a.models)), f"held-out test queries ({len(test):,})")
        print(json.dumps(candidate_stats(test), indent=1))
    if a.qrels:
        judged = dataset.qrels_queries(make_client(), lex, a.qrels)
        print_report(evaluate(judged, systems(a.models)), f"your judgments ({len(judged)} queries)")


if __name__ == "__main__":
    main()
