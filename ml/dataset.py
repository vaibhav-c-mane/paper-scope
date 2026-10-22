"""Real relevance labels for training and testing the ranker. No synthetic data.

The labels come from citations: when a paper's authors wrote it, they searched
the literature and cited what was relevant to them. So for a "seed" paper,

    query     = the seed paper's title
    relevant  = the papers its reference list cites
    searched  = in the seed's own year (later papers cannot be cited, so they are
                left out, and recency is measured from that year)

Every candidate the search retrieves for that query gets label 1 if the seed
cites it and 0 otherwise. This is the standard citation-recommendation set-up;
it under-counts relevance (nobody cites everything relevant), which lowers all
systems' scores equally and so still ranks the systems fairly.

Three ways to get the labelled queries:

* online_queries  - OpenAlex API: a reproducible random sample of works with
                    their reference lists; candidates come from the same
                    OpenAlex search the app uses. Needs internet.
* pool_queries    - a real OpenAlex sample held in memory (scripts/fetch_openalex.py
                    streams one from the public OpenAlex snapshot); candidates
                    are retrieved from the pool with BM25 + vector search. Used
                    when the OpenAlex API itself cannot be reached.
* qrels_queries   - your own judgments, a TSV file: query <tab> paper id <tab> grade
                    (paper id = DOI, arXiv id or OpenAlex id); candidates come
                    from online search.

All three produce the same thing: per query, the feature matrix of its
candidates (computed exactly as the search engine computes it), their labels,
and how many relevant papers exist in total (for recall and NDCG).
"""
import json
import random
import time

import numpy as np

from engine.domains import DOMAINS, weak_label
from engine.memindex import PaperIndex
from engine.papers import paper_keys
from engine.pipeline import PreparedQuery, top_k
from engine.sources import openalex_number
from engine.text import terms


class LabelledQuery:
    __slots__ = ("qid", "text", "year", "field", "x", "y", "bm25", "cand_fields", "n_relevant")

    def __init__(self, qid, text, year, field, x, y, bm25, cand_fields, n_relevant):
        self.qid, self.text, self.year, self.field = qid, text, year, field
        self.x, self.y, self.bm25 = x, y, bm25              # (n, 9) features, (n,) labels, (n,) raw BM25
        self.cand_fields = cand_fields                      # (n,) most likely field of each candidate
        self.n_relevant = n_relevant                        # relevant papers in total (found or not)

    def to_json(self):
        return {"qid": self.qid, "text": self.text, "year": self.year, "field": self.field,
                "candidates": int(len(self.y)), "relevant_found": int((self.y > 0).sum()),
                "relevant_total": self.n_relevant}


def _label(index, prepared, docs, relevant, year, field, qid, text, n_relevant):
    p_domain = prepared.field_distribution(index, docs)
    x, raw = prepared.features(index, docs, p_domain, now_year=year)
    y = np.asarray([1.0 if int(d) in relevant else 0.0 for d in docs], np.float32)
    cand_fields = index.domains.view[docs].argmax(axis=1).astype(np.int16)
    return LabelledQuery(qid, text, year, field, x, y, raw, cand_fields, n_relevant)


# ------------------------------------------------------------------ pool (OpenAlex snapshot sample)
def load_pool(path, limit=None):
    """Papers of a pool file written by scripts/fetch_openalex.py (one JSON per line, with `refs`)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if limit and len(out) >= limit:
                break
            try:
                p = json.loads(line)
            except ValueError:
                continue
            if p.get("title") and p.get("openalex_id"):
                out.append(p)
    return out


class Pool:
    """The pool in a PaperIndex (the same inverted + vector index the search engine uses),
    plus a map from OpenAlex work numbers to index ids to resolve reference lists."""

    def __init__(self, papers, lexicon, chunk=20000, progress=print):
        self.index = PaperIndex(lexicon, capacity=0)               # capacity 0: no eviction
        self.refs = {}                                             # index id -> cited work numbers
        t0 = time.time()
        papers = [p for p in papers if p.get("title")]
        for i in range(0, len(papers), chunk):
            part = papers[i:i + chunk]
            for p, d in zip(part, self.index.add(part)):              # a duplicate merges into one paper
                self.refs.setdefault(d, []).extend(p.get("refs") or [])
            if progress:
                progress(f"  pool index: {i + len(part):,} papers ({time.time() - t0:.0f}s)")
        self.number_to_doc = {int(k[len("openalex:w"):]): d for k, d in self.index.key_to_doc.items()
                              if k.startswith("openalex:w")}
        self.years = self.index.meta.view[:, 0].astype(np.int32)

    def cited(self, d):
        """Index ids of the pool papers that paper d cites."""
        return {self.number_to_doc[r] for r in self.refs.get(d, ()) if r in self.number_to_doc} - {d}

    def seeds(self, min_refs=3, min_year=1995):
        return [d for d in range(self.index.n)
                if self.years[d] >= min_year and len(self.cited(d)) >= min_refs]

    def candidates(self, prepared, year, exclude, k_lexical=100, k_semantic=100):
        """What a search for the query, made in `year`, retrieves: BM25 top k ∪ nearest k vectors."""
        idx = self.index
        allowed = (self.years <= year) & (self.years > 0)
        allowed[exclude] = False
        lexical = top_k(idx.bm25(prepared.ids, prepared.weights), allowed, k_lexical) \
            if len(prepared.ids) else np.zeros(0, np.int64)
        semantic = idx.search_vectors(prepared.qvec, k_semantic, allowed)[0] \
            if prepared.qvec is not None else np.zeros(0, np.int64)
        return np.unique(np.concatenate([lexical, semantic]))


def pool_queries(pool, seeds, progress=print, k_lexical=100, k_semantic=100):
    out, t0 = [], time.time()
    for n, d in enumerate(seeds):
        p = pool.index.papers[d]
        text, year = p["title"], int(pool.years[d])
        prepared = PreparedQuery(pool.index, terms(text), text)
        docs = pool.candidates(prepared, year, d, k_lexical, k_semantic)
        cited = pool.cited(d)
        if len(docs):
            out.append(_label(pool.index, prepared, docs, cited, year, weak_label(p), p["openalex_id"], text, len(cited)))
        if progress and (n + 1) % 500 == 0:
            progress(f"  labelled {n + 1:,}/{len(seeds):,} queries ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ online (OpenAlex API)
def online_queries(client, lexicon, n_queries=1500, seed=0, limit=100, progress=print):
    """Seeds sampled from OpenAlex; candidates from the same online search the app uses."""
    oa = client.providers["openalex"]
    seeds = oa.sample(n_queries, seed=seed, filters="has_abstract:true,referenced_works_count:>9,publication_year:>1999")
    out = []
    for n, s in enumerate(seeds):
        year, refs = int(s.get("year") or 0), set(s.get("refs") or [])
        if not year or not refs:
            continue
        found, _ = client.search(s["title"], limit=limit, year_to=year, deadline_s=20)
        own = openalex_number(s.get("openalex_id"))
        found = [p for p in found if openalex_number(p.get("openalex_id")) != own]
        if not found:
            continue
        index = PaperIndex(lexicon, capacity=0)
        docs = np.asarray(sorted(set(index.add(found))), np.int64)
        relevant = {d for d in docs.tolist() if openalex_number(index.papers[d].get("openalex_id")) in refs}
        prepared = PreparedQuery(index, terms(s["title"]), s["title"])
        out.append(_label(index, prepared, docs, relevant, year, weak_label(s), s["openalex_id"], s["title"], len(refs)))
        if progress and (n + 1) % 100 == 0:
            progress(f"  labelled {n + 1:,}/{len(seeds):,} queries online")
    return out


# ------------------------------------------------------------------ your own judgments
def read_qrels(path):
    """TSV: query <tab> paper id <tab> grade (0, 1 or 2). Lines starting with # are comments."""
    q = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                q.setdefault(parts[0].strip(), {})[parts[1].strip().lower()] = float(parts[2])
    return q


def _id_keys(ident):
    ident = ident.lower().replace("https://doi.org/", "")
    n = openalex_number(ident)
    if n:
        return {f"openalex:w{n}"}
    if ident.startswith("10."):
        return set(paper_keys({"doi": ident}))
    return {"arxiv:" + ident.removeprefix("arxiv:")}


def qrels_queries(client, lexicon, path, limit=100, year=None):
    out = []
    for qid, (text, judged) in enumerate(read_qrels(path).items()):
        found, _ = client.search(text, limit=limit, deadline_s=20)
        if not found:
            continue
        index = PaperIndex(lexicon, capacity=0)
        docs = np.asarray(sorted(set(index.add(found))), np.int64)
        grades = {}
        for ident, g in judged.items():
            for k in _id_keys(ident):
                d = index.find(k)
                if d is not None:
                    grades[d] = max(grades.get(d, 0.0), g)
        prepared = PreparedQuery(index, terms(text), text)
        q = _label(index, prepared, docs, set(), year, None, f"q{qid}", text, sum(1 for g in judged.values() if g > 0))
        q.y = np.asarray([grades.get(int(d), 0.0) for d in docs], np.float32)
        out.append(q)
    return out


# ------------------------------------------------------------------ splits
def split(queries, seed=7, test=0.2, valid=0.1):
    """Split by query (never by candidate), so test queries are unseen in training."""
    qs = list(queries)
    random.Random(seed).shuffle(qs)
    n_test, n_valid = int(len(qs) * test), int(len(qs) * valid)
    return qs[n_test + n_valid:], qs[n_test:n_test + n_valid], qs[:n_test]


def field_index(name):
    return DOMAINS.index(name) if name in DOMAINS else -1
