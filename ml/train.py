"""Train the search engine's models on real papers and real relevance labels, and evaluate them.

    python -m ml.train --online                      # everything from the OpenAlex API
    python -m ml.train --pool pool.jsonl             # from a sample of the OpenAlex snapshot
    python -m ml.train --pool pool.jsonl --qrels my_judgments.tsv   # + evaluate on your judgments

1. Lexicon (engine/lexicon.py): word statistics (BM25 idf), TF-IDF + LSA
   embeddings and the research-field classifier, learned from real paper
   titles and abstracts. The field classifier is checked on a held-out 20%.
2. Labelled queries (ml/dataset.py): citation-based relevance. Query = a
   paper's title, relevant = the papers it cites; candidates are what the
   search retrieves; features are computed exactly as the search engine does.
   Split by query: 70% train, 10% validation, 20% test.
3. Rankers: LambdaMART gradient-boosted trees (ml/lambdamart.py), early-stopped
   on validation NDCG@10, and a linear pairwise baseline. Platt scaling on the
   validation queries turns scores into a probability of relevance; the
   "confident match" threshold (fewer than 3 confident matches in memory ->
   search online) is set where validation precision reaches 50%.
4. Evaluation on the test queries (ml/evaluate.py).

Output, all in data/models/ (the only files the search engine reads):
  lexicon/            word statistics, LSA projection, field classifier
  ranker.json         the gradient-boosted ranker the search engine loads at start-up
  ranker_linear.json  the linear baseline
  eval_report.json    every number printed below
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from engine.domains import DOMAINS, DomainClassifier                 # noqa: E402
from engine.features import FEATURES                                 # noqa: E402
from engine.lexicon import Lexicon, lexicon_dir                      # noqa: E402
from engine.ranker import GBDTRanker, LinearRanker, save_ranker      # noqa: E402
from ml import dataset                                               # noqa: E402
from ml.evaluate import candidate_stats, evaluate, print_report, systems  # noqa: E402
from ml.lambdamart import confident_threshold, platt, train_lambdamart, train_linear  # noqa: E402


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ data
def add_data_args(ap):
    ap.add_argument("--models", default=os.path.join(HERE, "data", "models"), help="where the models live")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--online", action="store_true", help="seeds and candidates from the OpenAlex API")
    src.add_argument("--pool", help="papers from scripts/fetch_openalex.py (when the API is unreachable)")
    ap.add_argument("--qrels", help="your own judgments (TSV: query, paper id, grade) for evaluation")
    ap.add_argument("--queries", type=int, default=6000, help="citation queries to label")
    ap.add_argument("--limit", type=int, help="read at most this many pool papers")
    ap.add_argument("--seed", type=int, default=7, help="random seed for sampling and the split")


def make_client():
    from engine.sources import ScholarlyClient
    return ScholarlyClient(order=("openalex",))


def load_lexicon(models):
    return Lexicon.load(lexicon_dir(models))


def pool_papers(a):
    if getattr(a, "papers", None) is None:
        log(f"loading pool {a.pool}")
        a.papers = dataset.load_pool(a.pool, a.limit)
        log(f"  {len(a.papers):,} papers")
    return a.papers


def get_pool(a, lex):
    pool = dataset.Pool(pool_papers(a), lex, progress=log)
    a.papers = None                                    # the index holds its own copy now
    return pool


def labelled_queries(a, lex):
    """(train, validation, test) lists of LabelledQuery from the chosen source."""
    if a.pool:
        pool = get_pool(a, lex)
        seeds = pool.seeds()
        log(f"{len(seeds):,} papers cite at least 3 other pool papers")
        random.Random(a.seed).shuffle(seeds)
        queries = dataset.pool_queries(pool, seeds[:a.queries], progress=log)
    elif a.online:
        queries = dataset.online_queries(make_client(), lex, a.queries, seed=a.seed, progress=log)
    else:
        return [], [], []
    return dataset.split(queries, seed=a.seed)


def stack(queries):
    x = np.vstack([q.x for q in queries]) if queries else np.zeros((0, len(FEATURES)), np.float32)
    y = np.concatenate([q.y for q in queries]) if queries else np.zeros(0, np.float32)
    return x, y, [len(q.y) for q in queries]


# ------------------------------------------------------------------ steps
def build_lexicon(a):
    if a.pool:
        papers = pool_papers(a)
        source = f"pool file {os.path.basename(a.pool)}"
    else:
        papers = make_client().providers["openalex"].iter_works("has_abstract:true,type:article|preprint", 200000)
        source = "OpenAlex API"
    log(f"learning the lexicon from {source}")
    lex = Lexicon.build(papers, progress=log)
    lex.info["source"] = source
    return lex


def evaluate_field_classifier(lex, seed=13, sample=200000):
    """Held-out accuracy and macro-F1 of the research-field classifier (80/20 split)."""
    from sklearn.metrics import accuracy_score, f1_score

    rows = np.flatnonzero(lex.train_labels >= 0)
    rng = np.random.default_rng(seed)
    rows = rng.choice(rows, min(len(rows), sample), replace=False)
    cut = int(len(rows) * 0.8)
    tr, te = rows[:cut], rows[cut:]
    y = lex.train_labels
    clf = DomainClassifier().fit(lex.train_x[tr], y[tr], seed=seed)
    pred = clf.predict_proba(lex.train_x[te]).argmax(axis=1)
    majority = np.bincount(y[tr]).argmax()
    per_field = {DOMAINS[c]: int((y[te] == c).sum()) for c in np.unique(y[te])}
    return {"test_papers": int(len(te)), "accuracy": round(float(accuracy_score(y[te], pred)), 4),
            "macro_f1": round(float(f1_score(y[te], pred, average="macro")), 4),
            "majority_baseline_accuracy": round(float((y[te] == majority).mean()), 4),
            "test_papers_per_field": per_field}


def fit_rankers(train, valid, now):
    x, y, sizes = stack(train)
    xv, yv, sv = stack(valid)
    log(f"training LambdaMART on {len(train):,} queries ({len(y):,} candidates, {int(y.sum()):,} relevant)")
    trees, info = train_lambdamart(x, y, sizes, xv, yv, sv, progress=log)
    gbdt = GBDTRanker(trees, importance=dict(zip(FEATURES, np.round(info["importance"], 4).tolist())))
    log("training the linear baseline")
    w, mean, std = train_linear(x, y, sizes)
    linear = LinearRanker(w, mean, std)
    for r in (gbdt, linear):
        s = r.score(xv)
        r.platt = platt(s, yv)
        r.threshold = round(confident_threshold(r.relevance(xv), yv), 4)
        r.meta = {"trained_at": now, "train_queries": len(train), "validation_queries": len(valid),
                  "labels": "citations (query = paper title, relevant = papers it cites)"}
    gbdt.meta["lambdamart"] = {k: v for k, v in info.items() if k != "importance"}
    return gbdt, linear


def main(argv=None):
    from engine.sources import RateLimited
    try:
        return _main(argv)
    except RateLimited as e:                 # e.g. OpenAlex's free shared budget is used up
        sys.exit(f"\nOpenAlex refused: {e}\n\nOnline training needs OpenAlex. Get a free key "
                 "(https://help.openalex.org/api/authentication/), then run\n"
                 "  OPENALEX_API_KEY=your_key python -m ml.train --online\n"
                 "or train offline with --pool FILE (see README).")


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--keep-lexicon", action="store_true", help="reuse data/models/lexicon instead of relearning it")
    a = ap.parse_args(argv)
    if not (a.pool or a.online):
        ap.error("choose where the training papers come from: --online or --pool FILE")
    os.makedirs(a.models, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    report = {"trained_at": now, "source": "pool" if a.pool else "online"}

    # 1. lexicon
    if a.keep_lexicon and os.path.isdir(lexicon_dir(a.models)):
        lex = load_lexicon(a.models)
        log("reusing the saved lexicon")
    else:
        lex = build_lexicon(a)
        report["field_classifier"] = evaluate_field_classifier(lex)
        log(f"field classifier: {report['field_classifier']}")
        lex.train_x = lex.train_labels = None
        lex.save(lexicon_dir(a.models))
    report["lexicon"] = lex.info

    # 2. labelled queries
    train, valid, test = labelled_queries(a, lex)
    train = [q for q in train if (q.y > 0).any()]          # a query without a relevant candidate teaches nothing
    valid = [q for q in valid if (q.y > 0).any()]
    report["data"] = {"train": candidate_stats(train), "validation": candidate_stats(valid), "test": candidate_stats(test)}
    log(f"queries: {json.dumps(report['data'])}")

    # 3. rankers
    gbdt, linear = fit_rankers(train, valid, now)
    save_ranker(gbdt, os.path.join(a.models, "ranker.json"))
    save_ranker(linear, os.path.join(a.models, "ranker_linear.json"))
    report["gbdt"] = gbdt.describe() | {"threshold": gbdt.threshold, "platt": gbdt.platt,
                                        "lambdamart": gbdt.meta["lambdamart"]}
    report["linear"] = linear.describe() | {"threshold": linear.threshold}

    # 4. evaluation on the held-out test queries
    report["test"] = evaluate(test, systems(a.models))
    print_report(report["test"], f"held-out test queries ({len(test):,})")
    if a.qrels:
        judged = dataset.qrels_queries(make_client(), lex, a.qrels)
        report["your_judgments"] = evaluate(judged, systems(a.models))
        print_report(report["your_judgments"], f"your judgments ({len(judged)} queries)")
    report["test_queries"] = [q.to_json() for q in test[:200]]
    with open(os.path.join(a.models, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    log(f"saved models and eval_report.json to {a.models}")


if __name__ == "__main__":
    main()
