"""The search pipeline: papers in memory first, the internet when they are not enough.

    query
      │  parse: words, -exclusions, prefix*, AND/OR/NOT (stack), DOI / arXiv id
      ▼
    local candidates ── BM25 top 200 (inverted index) ∪ nearest 200 vectors,
      │                 over the papers fetched earlier in this session
      ▼
    filters      ── year range, author, field, exclusions / boolean set
      ▼
    features     ── bm25, semantic, title/term coverage, domain match,
      │             citations, recency, peer reviewed, venue quality
      ▼
    rerank       ── learned ranker (gradient-boosted trees, LambdaMART)
      ▼
    fewer than 3 confident matches?
      │  yes ─► OpenAlex → Semantic Scholar → Crossref (rate-limited, cached in
      │         memory); the new papers are added to the in-memory index and the
      │         ranking is computed again over everything
      ▼
    results, ordered with merge sort (stable)

Nothing is written to disk while searching. The only files the engine reads are
the trained models in data/models/ (Lexicon + ranker), loaded once at start-up.
"""
import os
import re
import threading
import time

import numpy as np

from dsa.expression import evaluate_postfix, infix_to_postfix, tokenize_query
from dsa.heap import top_k as heap_top_k
from dsa.merge import merge_sort
from engine.domains import DOMAINS
from engine.features import FEATURES, feature_matrix
from engine.lexicon import Lexicon, lexicon_dir
from engine.memindex import PaperIndex
from engine.ranker import HeuristicRanker, load_ranker
from engine.text import normalize, normalize_query_word, terms


_OPS = {"AND", "OR", "NOT"}
_ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?$", re.I)
_DOI = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.I)
_OPENALEX = re.compile(r"^(?:https?://openalex\.org/)?(W\d{4,})$", re.I)
MODES = ("bm25", "semantic", "hybrid", "learned")


def parse_query(q):
    """Split the query box text into the parts the pipeline needs."""
    raw = " ".join((q or "").split())
    out = {"raw": raw, "identifier": None, "boolean": None, "words": [], "excluded": [], "text": ""}
    for kind, rx in (("doi", _DOI), ("arxiv", _ARXIV_ID), ("openalex", _OPENALEX)):
        m = rx.match(raw)
        if m:
            out["identifier"] = (kind, m.group(1) if kind == "openalex" else m.group(1).lower())
            return out
    toks = raw.replace("(", " ( ").replace(")", " ) ").replace('"', " ").split()
    if any(t in _OPS or t in ("(", ")") for t in toks):
        out["boolean"] = tokenize_query(raw, normalize_query_word)
    positive = []
    for i, t in enumerate(toks):
        if t in _OPS or t in ("(", ")"):
            continue
        if t.startswith("-") and len(t) > 1:
            w = normalize(t[1:])
            if w:
                out["excluded"].append(w)
            continue
        if i > 0 and toks[i - 1] == "NOT":
            continue
        positive.append(t)
    out["words"] = [w for w in (normalize_query_word(t) for t in positive) if w]
    out["text"] = " ".join(t.rstrip("*") for t in positive)
    return out


# ---------------------------------------------------------------- shared with ml/train.py
class PreparedQuery:
    """Everything about a query that does not depend on the candidates: term ids and
    weights (BM25), word groups (coverage), embedding, and P(field | query text)."""

    def __init__(self, index, words, text):
        self.words, self.text = list(words), text
        uniq = list(dict.fromkeys(words))
        self.groups = [index.expand(w) for w in uniq]
        self.ids, self.weights = index.query_ids(words)
        toks = terms(text)
        if toks:
            vec, p = index.lex.text_features([([], toks)])
            self.qvec = vec[0] if np.any(vec[0]) else None
            self.p_text = p[0]
        else:
            self.qvec, self.p_text = None, np.full(len(DOMAINS), 1.0 / len(DOMAINS), np.float32)

    def first_pass(self, index, docs):
        """Cheap score used to pick pseudo-relevant papers: scaled BM25 + cosine."""
        docs = np.asarray(docs, np.int64)
        s = np.zeros(len(docs), np.float32)
        if len(self.ids) and len(docs):
            s += np.clip(index.bm25(self.ids, self.weights)[docs] / index.ceiling(self.ids, self.weights), 0, 1)
        if self.qvec is not None and len(docs):
            s += index.vectors.view[docs] @ self.qvec
        return s

    def field_distribution(self, index, docs, field=None, feedback=10):
        """P(field | query): half from the classifier on the query text, half from the
        fields of the best first-pass candidates (pseudo-relevance feedback). A field
        chosen by the user gets most of the weight."""
        p = np.asarray(self.p_text, np.float32)
        docs = np.asarray(docs, np.int64)
        if len(docs):
            top = docs[np.argsort(-self.first_pass(index, docs), kind="stable")[:feedback]]
            p = 0.5 * p + 0.5 * index.domains.view[top].mean(axis=0)
        if field in DOMAINS:
            one = np.zeros(len(DOMAINS), np.float32)
            one[DOMAINS.index(field)] = 1.0
            p = 0.8 * one + 0.2 * p
        return (p / (p.sum() or 1.0)).astype(np.float32)

    def features(self, index, docs, p_domain, now_year=None):
        return feature_matrix(index, docs, self.groups, self.ids, self.weights, self.qvec, p_domain, now_year)


def top_k(scores, allowed, k):
    """Indices of the k best positive scores among allowed papers (size-k min-heap, dsa/heap.py)."""
    cand = np.flatnonzero(allowed & (scores > 0))
    return np.asarray(heap_top_k(scores[cand].tolist(), cand.tolist(), k), np.int64)


def mode_scores(mode, x, ranker, raw_bm25=None):
    """Ranking score of each candidate under one of the comparison modes."""
    if mode == "bm25":
        return raw_bm25 if raw_bm25 is not None else x[:, FEATURES.index("bm25")]
    if mode == "semantic":
        return x[:, FEATURES.index("semantic")]
    if mode == "hybrid":
        return HeuristicRanker().score(x)
    return ranker.score(x)


# ---------------------------------------------------------------- the service
class SearchPipeline:
    N_LEXICAL = 200
    N_SEMANTIC = 200
    MIN_CONFIDENT = 3           # fewer confident local matches than this -> search online
    ONLINE_LIMIT = 50           # papers asked from the online source per search
    N_RELATED = 8               # related papers in the paper drawer (BFS over citations)

    def __init__(self, models_dir="data/models", client=None, lexicon=None, ranker=None, capacity=20000,
                 now_year=None):
        self.models_dir = models_dir
        self.lex = lexicon if lexicon is not None else Lexicon.load(lexicon_dir(models_dir))
        self.ranker = ranker if ranker is not None else load_ranker(os.path.join(models_dir, "ranker.json"))
        self.index = PaperIndex(self.lex, capacity=capacity)
        self.client = client
        self.now_year = now_year
        self.lock = threading.RLock()        # the in-memory index is shared by all requests

    def reload_models(self):
        self.ranker = load_ranker(os.path.join(self.models_dir, "ranker.json"))

    # ------------------------------------------------------------ filters
    def _allowed(self, parsed, year_from, year_to, author, field, strict_field):
        idx = self.index
        n = idx.n
        ok = idx.alive.view.copy()
        if year_from or year_to:            # year tree (dsa/bst.py): visits only the years in range
            lo = int(year_from) if year_from else -10 ** 9
            hi = int(year_to) if year_to else 10 ** 9
            if year_to:
                lo = max(lo, 1)             # an upper bound leaves out papers of unknown year (0)
            in_range = np.zeros(n, bool)
            in_range[np.asarray(idx.years_between(lo, hi), np.int64)] = True
            ok &= in_range
        if author and author.strip():
            needle = author.lower().strip()
            ok &= np.fromiter(((p is not None and needle in p["authors"].lower()) for p in idx.papers), bool, n)
        if strict_field and field in DOMAINS and n:
            ok &= idx.domains.view.argmax(axis=1) == DOMAINS.index(field)
        for w in parsed["excluded"]:
            ok[idx.docs_with(w)] = False
        return ok

    def _boolean(self, parsed, trace):
        """AND / OR / NOT over the in-memory posting lists, with the two stacks of dsa/expression.py.
        None if the expression cannot be read (e.g. "Cats AND the", where "the" is a stop word):
        then the words are searched without the operators."""
        try:
            postfix, steps = infix_to_postfix(parsed["boolean"])
            universe = np.flatnonzero(self.index.alive.view).tolist()
            docs = evaluate_postfix(postfix, lambda w: self.index.docs_with(w).tolist(), universe)
        except (ValueError, IndexError) as e:
            trace["boolean"] = {"tokens": parsed["boolean"], "error": f"{e}; searched the words instead"}
            return None
        trace["boolean"] = {"tokens": parsed["boolean"], "postfix": postfix, "steps": steps, "matches": len(docs)}
        return docs

    # ------------------------------------------------------------ ranking over memory
    def _rank_local(self, parsed, filters, mode, ranker, forced=(), trace=None):
        """Rows (doc, score, relevance, features) over the papers in memory, best first."""
        idx = self.index
        if not idx.n:
            return []
        allowed = self._allowed(parsed, *filters)
        docs = self._boolean(parsed, trace if trace is not None else {}) if parsed["boolean"] else None
        if docs is not None:
            keep = np.zeros(idx.n, bool)
            keep[np.asarray(docs, np.int64)] = True
            allowed &= keep
        q = PreparedQuery(idx, parsed["words"], parsed["text"])
        t0 = time.perf_counter()
        lexical = top_k(idx.bm25(q.ids, q.weights), allowed, self.N_LEXICAL) if len(q.ids) else np.zeros(0, np.int64)
        t1 = time.perf_counter()
        semantic = idx.search_vectors(q.qvec, self.N_SEMANTIC, allowed)[0] if q.qvec is not None else np.zeros(0, np.int64)
        t2 = time.perf_counter()
        forced = np.asarray([d for d in forced if 0 <= d < idx.n and allowed[d]], np.int64)
        docs = np.unique(np.concatenate([lexical, semantic, forced]))
        if trace is not None:
            trace["candidates"] = {"lexical": int(len(lexical)), "semantic": int(len(semantic)),
                                   "from_online": int(len(forced)), "total": int(len(docs)),
                                   "in_memory": len(idx), "allowed_by_filters": int(allowed.sum()),
                                   "ms": {"lexical": round((t1 - t0) * 1000, 2), "semantic": round((t2 - t1) * 1000, 2)}}
        if not len(docs):
            return []
        field = filters[3]
        p_domain = q.field_distribution(idx, docs, field)
        x, raw = q.features(idx, docs, p_domain, self.now_year)
        r = ranker or self.ranker
        score = mode_scores(mode, x, r, raw)
        rel = r.relevance(x, score if mode == "learned" else None)   # learned mode: reuse the score
        if trace is not None:
            top = np.argsort(-p_domain)[:3]
            trace["field"] = {"chosen": field if field in DOMAINS else None,
                              "detected": [{"name": DOMAINS[i], "p": round(float(p_domain[i]), 3)} for i in top]}
            trace["rerank"] = {"mode": mode, "ranker": r.kind if mode == "learned" else mode,
                               "threshold": round(float(r.threshold), 3)}
        rows = [(int(d), float(s), float(v), xi) for d, s, v, xi in zip(docs, score, rel, x)]
        return merge_sort(rows, key=lambda row: row[1], reverse=True)

    # ------------------------------------------------------------ public API
    def rank(self, q, year_from=None, year_to=None, author="", field=None, strict_field=False, mode="learned",
             ranker=None, external="auto", trace=None):
        """Ranked rows (paper key, score, relevance, features). external: auto | always | never."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        trace = trace if trace is not None else {}
        parsed = parse_query(q)
        trace["query"] = {"words": parsed["words"], "excluded": parsed["excluded"], "text": parsed["text"]}
        if parsed["identifier"]:
            return self._identifier(parsed["identifier"], external, trace)
        if not parsed["words"] and not parsed["boolean"]:
            return []
        filters = (year_from, year_to, author, field, strict_field)
        r = ranker or self.ranker
        with self.lock:
            rows = self._rank_local(parsed, filters, mode, r, trace=trace)
            confident = sum(1 for row in rows if row[2] >= r.threshold)
            keys_out = [(self.index.key_of(d), s, v, x) for d, s, v, x in rows]
        go_online = external == "always" or (external == "auto" and confident < self.MIN_CONFIDENT)
        if not go_online or self.client is None:
            trace["online"] = {"used": False, "reason": "not connected" if self.client is None and go_online
                               else f"{confident} confident matches in memory" if external == "auto" else "turned off"}
            return keys_out
        papers, report, t0 = self._search_online(parsed, year_from, year_to)
        with self.lock:
            ids = self.index.add(papers)
            trace["online"] = {"used": True, "providers": report, "fetched": len(papers),
                               "new": self.index.added,
                               "reason": "requested" if external == "always" else f"only {confident} confident matches in memory",
                               "ms": round((time.perf_counter() - t0) * 1000, 1)}
            rows = self._rank_local(parsed, filters, mode, r, forced=ids, trace=trace)
            return [(self.index.key_of(d), s, v, x) for d, s, v, x in rows]

    def _search_online(self, parsed, year_from, year_to):
        t0 = time.perf_counter()
        text = parsed["raw"]
        try:
            papers, report = self.client.search(text, limit=self.ONLINE_LIMIT, year_from=year_from, year_to=year_to)
        except Exception as e:                          # an API problem never breaks the search
            papers, report = [], [{"provider": "online", "ok": False, "error": f"{type(e).__name__}: {e}"[:160]}]
        return papers, report, t0

    def _identifier(self, ident, external, trace):
        kind, value = ident
        key = {"arxiv": "arxiv:" + re.sub(r"v\d+$", "", value), "doi": "doi:" + value,
               "openalex": "openalex:" + value.lower()}[kind]
        if kind == "doi" and value.startswith("10.48550/arxiv."):
            key = "arxiv:" + value.split("arxiv.", 1)[1]
        with self.lock:
            d = self.index.find(key)
            if d is not None:
                self.index.touch(d)
                key = self.index.key_of(d)
        trace["identifier"] = {"kind": kind, "value": value, "in_memory": d is not None}
        if d is None and external != "never" and self.client is not None:
            paper, report = self.client.get_paper(value)
            trace["identifier"]["online"] = report
            if paper:
                with self.lock:
                    d = self.index.add([paper])[0]
                    key = self.index.key_of(d)
        if d is None:
            return []
        return [(key, 1.0, 1.0, np.zeros(len(FEATURES), np.float32))]

    def search(self, q, year_from=None, year_to=None, author="", field=None, strict_field=False,
               sort_by="relevance", page=1, per_page=10, external="auto", mode="learned"):
        """What app.py and the CLI call: one page of results plus a trace of what happened."""
        t0 = time.perf_counter()
        trace = {}
        rows = self.rank(q, year_from, year_to, author, field, strict_field, mode=mode, external=external, trace=trace)
        with self.lock:
            items = []
            for key, score, rel, x in rows:
                d = self.index.find(key) if key else None
                if d is None:
                    continue
                items.append((d, score, rel, x))
            if sort_by in ("year", "citations") and items:
                col = 0 if sort_by == "year" else 1                # stored metadata columns
                items = merge_sort(items, key=lambda it: float(self.index.meta.view[it[0], col]), reverse=True)
            total = len(items)
            start = max(0, (page - 1) * per_page)
            results = []
            for d, score, rel, x in items[start:start + per_page]:
                self.index.touch(d)                     # shown papers count as recently used
                p = {k: v for k, v in self.index.paper(d).items() if not k.startswith("_")}
                p.update({"key": self.index.key_of(d), "score": round(score, 4), "relevance": round(rel, 3),
                          "field": DOMAINS[int(self.index.domains.view[d].argmax())],
                          "features": {f: round(float(v), 3) for f, v in zip(FEATURES, x)}})
                results.append(p)
        trace["sort"] = sort_by
        trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"total": total, "page": page, "per_page": per_page, "results": results, "trace": trace}

    def paper(self, key):
        """One paper by its key (from memory, or online if it was evicted since)."""
        with self.lock:
            d = self.index.find(key)
            if d is not None:
                self.index.touch(d)
                return self._public(d, key)
        kind, _, value = key.partition(":")
        if self.client is None or kind not in ("doi", "arxiv", "openalex"):
            return None
        paper, _ = self.client.get_paper(value)
        if not paper:
            return None
        with self.lock:
            d = self.index.add([paper])[0]
            return self._public(d, self.index.key_of(d))

    def _public(self, d, key):
        """Paper d as the API returns it, with its related papers (BFS over citations in memory)."""
        idx = self.index
        p = {k: v for k, v in idx.paper(d).items() if not k.startswith("_")}
        p["key"] = key
        p["citation_data"] = idx.has_citation_data(d)
        cited = set(idx.cites(d))
        p["related"] = [{"key": idx.key_of(r), "title": idx.paper(r)["title"], "year": idx.paper(r)["year"],
                         "hops": hops, "via": idx.paper(via)["title"] if hops > 1 else None,
                         "relation": ("reference" if r in cited else "citing") if hops == 1 else None}
                        for r, hops, via in idx.related(d, max_depth=2, limit=self.N_RELATED)]
        return p

    def suggest(self, prefix, limit=8):
        """Words starting with `prefix`, most common first (binary search in the sorted lexicon)."""
        w = "".join(ch for ch in prefix.lower() if ch.isalnum())
        if len(w) < 2:
            return []
        lo, hi = self.lex.prefix_range(w)
        ids = np.arange(lo, hi)
        if not len(ids):
            return []
        best = ids[np.argsort(-self.lex.df[ids], kind="stable")[:limit]]
        return [self.lex.vocab[i] for i in best]

    def state(self):
        with self.lock:
            s = self.index.stats()
        s.update({"ranker": self.ranker.describe(), "lexicon": self.lex.info,
                  "online": self.client is not None, "fields": DOMAINS})
        return s
