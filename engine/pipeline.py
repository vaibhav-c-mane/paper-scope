"""The search pipeline: papers in memory first, the internet when they are not enough.

    query
      │  parse: words, -exclusions, prefix*, AND/OR/NOT (stack), DOI / arXiv id
      │  understand (engine/understand.py): fix typos, drop "find me papers on",
      │  "recent" -> year filter, sentence or title?
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
      │         memory), and at the same time OpenAlex by meaning (for sentences)
      │         and by title (for title-like queries); the new papers are added to
      │         the in-memory index and the ranking is computed again over everything
      ▼
    results, ordered with merge sort (stable)

Nothing is written to disk while searching. The only files the engine reads are
the trained models in data/models/ (Lexicon + ranker), loaded once at start-up.
"""
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
from engine.understand import QueryUnderstanding


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


FUSION_K = 60

# The match shown to the user: how much of the query a paper covers, in words and in meaning.
# Citations, recency and venue are left out on purpose: they help the ranker order papers, but
# a new or uncited paper can be exactly on topic. The ranker's own relevance (0..1) is the
# probability that a paper with this query as its title would cite the candidate, which stays
# low for sentences even when the papers are right, so it is not shown as the match.
MATCH_WEIGHTS = {"term_coverage": 0.45, "title_coverage": 0.20, "semantic": 0.35}
MATCH_LEVELS = ((0.75, "strong"), (0.5, "good"), (0.0, "partial"))
ON_TOPIC_SHARE = 0.7        # sorting by year or citations: papers within 70% of the best match first


def match_scores(x):
    """0..1 topic match of each row of a feature matrix (see MATCH_WEIGHTS)."""
    x = np.asarray(x, np.float32).reshape(-1, len(FEATURES))
    w = np.zeros(len(FEATURES), np.float32)
    for f, v in MATCH_WEIGHTS.items():
        w[FEATURES.index(f)] = v
    return np.clip(x @ w, 0.0, 1.0)


def match_level(m):
    return next(name for floor, name in MATCH_LEVELS if m >= floor)


def fuse(rows, others, k=FUSION_K, weights=None):
    """Reciprocal rank fusion of our ranked rows with other orders (lists of paper ids: OpenAlex's
    search by meaning, its fewer-words search). Each paper scores 1/(k + its place) in every list
    it is in (times that list's weight, 1 by default), and rows are re-sorted by the sum (merge sort,
    stable). A paper near the top of several lists wins; one list alone cannot push a paper far
    up, because k = 60 flattens the gap between places 1 and 10. `others` may be one list."""
    if others and not isinstance(others[0], (list, tuple, np.ndarray)):
        others = [others]
    weights = list(weights) if weights is not None else [1.0] * len(others)
    places = []
    for other in others:
        place = {}
        for i, d in enumerate(other):
            place.setdefault(int(d), i + 1)
        places.append((place, weights[len(places)]))
    scored = [(1.0 / (k + i + 1) + sum(w / (k + pl[row[0]]) for pl, w in places if row[0] in pl), row)
              for i, row in enumerate(rows)]
    return [row for _, row in merge_sort(scored, key=lambda t: t[0], reverse=True)]


def _title_words(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


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
    N_RELATED = 8               # related papers (Dijkstra) and citation links (BFS) in the paper drawer
    N_TITLES = 10               # most cited papers whose title contains a title-like query
    TITLE_MIN_CITATIONS = 50    # an exact title match moves to the top only if cited this often
    # other orders fused with the ranker's (name -> label, weight in the fusion)
    FUSED = {"semantic": ("OpenAlex (by meaning)", 1.0), "relaxed": ("OpenAlex (fewer words)", 0.5)}

    def __init__(self, models_dir="data/models", client=None, lexicon=None, ranker=None, capacity=20000,
                 now_year=None):
        self.models_dir = models_dir
        self.lex = lexicon if lexicon is not None else Lexicon.load(lexicon_dir(models_dir))
        self.ranker = ranker if ranker is not None else load_ranker(os.path.join(models_dir, "ranker.json"))
        self.index = PaperIndex(self.lex, capacity=capacity)
        self.client = client
        self.now_year = now_year
        self.lock = threading.RLock()        # the in-memory index is shared by all requests
        self.understanding = QueryUnderstanding(self.lex)
        self.pool = ThreadPoolExecutor(max_workers=4)   # keyword, meaning, fewer-words and title searches run together

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
             ranker=None, external="auto", trace=None, exact=False):
        """Ranked rows (paper key, score, relevance, features). external: auto | always | never.
        exact=True searches the words as typed, without query understanding."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        trace = trace if trace is not None else {}
        parsed = parse_query(q)
        understood = None
        if not exact and parsed["words"] and not parsed["identifier"] and not parsed["boolean"]:
            understood = self.understanding.understand(parsed["raw"], self.now_year or time.localtime().tm_year)
            trace["understood"] = understood
            if understood["query"] != parsed["raw"]:
                parsed = parse_query(understood["query"])
            if understood["recent_from"] and not year_from:
                year_from = understood["recent_from"]
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
            if mode == "learned":
                rows = self._exact_titles_first(rows, parsed, trace)
            keys_out = [(self.index.key_of(d), s, v, x) for d, s, v, x in rows]
        go_online = external == "always" or (external == "auto" and confident < self.MIN_CONFIDENT)
        if not go_online or self.client is None:
            trace["online"] = {"used": False, "reason": "not connected" if self.client is None and go_online
                               else f"{confident} confident matches in memory" if external == "auto" else "turned off"}
            return keys_out
        papers, report, t0, parts, saving = self._search_online(parsed, understood, year_from, year_to, field)
        with self.lock:
            ids = self.index.add(papers)
            trace["online"] = {"used": True, "providers": report, "fetched": len(papers),
                               "new": self.index.added,
                               "reason": "requested" if external == "always" else f"only {confident} confident matches in memory",
                               "ms": round((time.perf_counter() - t0) * 1000, 1), **saving}
            rows = self._rank_local(parsed, filters, mode, r, forced=ids, trace=trace)
            if mode == "learned":
                start, lists, names = 0, [], []
                for name, count in parts:
                    if name in self.FUSED and count:
                        lists.append(ids[start:start + count])
                        names.append(self.FUSED[name])
                    start += count
                if lists:
                    rows = fuse(rows, lists, weights=[w for _, w in names])
                    trace["fusion"] = {"with": " and ".join(n for n, _ in names), "papers": sum(map(len, lists)), "k": FUSION_K}
                rows = self._exact_titles_first(rows, parsed, trace)
            return [(self.index.key_of(d), s, v, x) for d, s, v, x in rows]

    def _exact_titles_first(self, rows, parsed, trace):
        """A well-cited paper whose title is exactly the query goes first, then well-cited papers whose
        title starts with a query of 4+ words ("BERT pre-training of deep bidirectional transformers").
        The ranker cannot know that the user typed a title; without this the original "Attention Is
        All You Need" ranks below a dozen papers that reuse its title. "Well cited" (at least
        TITLE_MIN_CITATIONS) keeps an obscure paper that happens to be named like a topic
        ("Machine learning for protein structure prediction", 2 citations) from jumping ahead."""
        q = _title_words(parsed["text"])
        if len(q) < 2:
            return rows

        def tier(row):
            p = self.index.paper(row[0])
            if (p.get("citations") or 0) < self.TITLE_MIN_CITATIONS:
                return 0
            t = _title_words(p["title"])
            return 2 if t == q else 1 if len(q) >= 4 and t[:len(q)] == q else 0
        tiers = [tier(row) for row in rows]
        if not any(tiers):
            return rows
        trace["exact_title"] = sum(1 for t in tiers if t == 2)
        ordered = merge_sort(list(zip(tiers, range(len(rows)))), key=lambda tr: tr[0], reverse=True)   # stable
        return [rows[i] for _, i in ordered]

    def _query_field(self, text, field=None):
        """The research field that picks the free source: the user's choice, else the classifier's guess."""
        if field in DOMAINS:
            return field
        toks = terms(text)
        if not toks:
            return None
        p = self.lex.text_features([([], toks)])[1][0]
        return DOMAINS[int(np.argmax(p))]

    def _search_online(self, parsed, understood, year_from, year_to, field=None):
        """Keyword search through the provider cascade; for a sentence also OpenAlex's search by
        meaning, for a title-like query also its search by title. They run at the same time.
        In saver mode (the day's OpenAlex budget is nearly used up) the extra searches are skipped.
        Returns (papers, report, start time, [(part name, number of papers)] in papers' order,
        what was done to save budget, for the trace)."""
        t0 = time.perf_counter()
        text = parsed["raw"]
        plain = " ".join(t for t in text.split() if not t.startswith("-")).replace("*", "")
        saver, why = self.client.saver() if hasattr(self.client, "saver") else (False, "")
        qfield = self._query_field(plain, field)
        saving = {"field_source": qfield}
        extra = []
        if saver:
            u = understood or {}
            skipped = [label for on, label in ((u.get("semantic"), "by meaning"), (u.get("relaxed"), "with fewer words"),
                                               (u.get("title"), "by title")) if on]
            saving["saver"] = {"reason": why, "skipped": skipped}
        elif understood is not None and hasattr(self.client, "search_openalex"):
            if understood["semantic"]:
                extra.append(("semantic", self.pool.submit(self.client.search_openalex, "semantic", plain,
                                                           self.ONLINE_LIMIT, year_from=year_from, year_to=year_to)))
            if understood.get("relaxed"):
                extra.append(("relaxed", self.pool.submit(self.client.search_openalex, "relaxed", understood["relaxed"],
                                                          self.ONLINE_LIMIT, year_from=year_from, year_to=year_to)))
            if understood["title"]:
                extra.append(("titles", self.pool.submit(self.client.search_openalex, "titles", plain,
                                                         self.N_TITLES, year_from=year_from, year_to=year_to)))
        try:
            papers, report = self.client.search(text, limit=self.ONLINE_LIMIT, year_from=year_from, year_to=year_to,
                                                **({"field": qfield, "saver": saver} if hasattr(self.client, "saver") else {}))
        except Exception as e:                          # an API problem never breaks the search
            papers, report = [], [{"provider": "online", "ok": False, "error": f"{type(e).__name__}: {e}"[:160]}]
        parts = [("keyword", len(papers))]
        for name, job in extra:
            try:
                found, entry = job.result()
            except Exception as e:
                found, entry = [], {"provider": "OpenAlex", "ok": False, "error": f"{type(e).__name__}: {e}"[:160]}
            papers = papers + found                     # duplicates are merged by PaperIndex.add
            report = report + [entry]
            parts.append((name, len(found)))
        if hasattr(self.client, "budget"):
            saving["budget"] = self.client.budget()
        return papers, report, t0, parts, saving

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
               sort_by="relevance", page=1, per_page=10, external="auto", mode="learned", exact=False):
        """What app.py and the CLI call: one page of results plus a trace of what happened."""
        t0 = time.perf_counter()
        trace = {}
        rows = self.rank(q, year_from, year_to, author, field, strict_field, mode=mode, external=external,
                         trace=trace, exact=exact)
        with self.lock:
            items = []
            for key, score, rel, x in rows:
                d = self.index.find(key) if key else None
                if d is None:
                    continue
                items.append((d, score, rel, x))
            match = match_scores([it[3] for it in items]) if items else np.zeros(0, np.float32)
            if "identifier" in trace:
                match[:] = 1.0                                  # the paper that was asked for
            items = [it + (float(m),) for it, m in zip(items, match)]
            if sort_by in ("year", "citations") and items:
                # Newest or most cited among the papers that are on topic; the rest follow in
                # relevance order. Sorting every candidate put weak matches first.
                bar = ON_TOPIC_SHARE * float(match.max())
                on = [it for it in items if it[4] >= bar]
                col = 0 if sort_by == "year" else 1                # stored metadata columns
                items = merge_sort(on, key=lambda it: float(self.index.meta.view[it[0], col]), reverse=True) + \
                    [it for it in items if it[4] < bar]
                trace["sorted_on_topic"] = len(on)
            total = len(items)
            start = max(0, (page - 1) * per_page)
            results = []
            missing = self._missing_words(trace)
            for d, score, rel, x, m in items[start:start + per_page]:
                self.index.touch(d)                     # shown papers count as recently used
                p = {k: v for k, v in self.index.paper(d).items() if not k.startswith("_")}
                p.update({"key": self.index.key_of(d), "score": round(score, 4), "relevance": round(rel, 3),
                          "match": round(m, 3), "match_level": match_level(m), "missing": missing(d),
                          "field": DOMAINS[int(self.index.domains.view[d].argmax())],
                          "features": {f: round(float(v), 3) for f, v in zip(FEATURES, x)}})
                results.append(p)
        trace["sort"] = sort_by
        trace["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"total": total, "page": page, "per_page": per_page, "results": results, "trace": trace}

    def _missing_words(self, trace):
        """A function: paper id -> the query words it does not contain (shown as "Missing: night",
        so a result that skips a word is not mistaken for a wrong one). A paper's word ids are
        sorted in the forward index, so each check is a binary search, O(log n) per word."""
        words = list(dict.fromkeys((trace.get("query") or {}).get("words") or []))
        if not words or "identifier" in trace:
            return lambda d: []
        shown = {}
        for tok in (trace["query"].get("text") or "").split():
            for t in terms(tok):
                shown.setdefault(t, re.sub(r"[^\w-]", "", tok.lower()))
        groups = [(shown.get(w, w.rstrip("*")), np.asarray(self.index.expand(w), np.int64)) for w in words]

        def missing(d):
            ids = self.index.fwd[d][0]
            out = []
            for word, g in groups:
                pos = np.searchsorted(ids, g)
                if not len(ids) or not np.any((pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == g)):
                    out.append(word)
            return out
        return missing

    def paper(self, key):
        """One paper by its key (from memory, or online if it was evicted since), with its
        related papers. The first time a paper is opened, the papers around it (references,
        OpenAlex's related works, most cited citing works) are fetched into memory."""
        with self.lock:
            d = self.index.find(key)
            if d is not None:
                self.index.touch(d)
        if d is None:
            kind, _, value = key.partition(":")
            if self.client is None or kind not in ("doi", "arxiv", "openalex"):
                return None
            paper, _ = self.client.get_paper(value)
            if not paper:
                return None
            with self.lock:
                d = self.index.add([paper])[0]
        with self.lock:
            key, saved = self.index.key_of(d), dict(self.index.paper(d))
        self._fetch_neighbourhood(key)
        with self.lock:                     # adding papers can evict or renumber: find it again by key
            d = self.index.find(key)
            if d is None:
                again = {k: v for k, v in saved.items() if not k.startswith("_")}
                d = self.index.add([dict(again, refs=saved.get("_refs") or [])])[0]
            return self._public(d, self.index.key_of(d))

    def _fetch_neighbourhood(self, key):
        with self.lock:
            d = self.index.find(key)
            p = self.index.paper(d) if d is not None else None
            number = self.index._work_number(p) if p else None
            if number is None or "_related" in p or self.client is None \
                    or not hasattr(self.client, "neighbourhood"):
                return
        related, papers, report = self.client.neighbourhood(number)
        with self.lock:
            if papers:
                self.index.add(papers)
            d = self.index.find("openalex:w%d" % number)
            if d is not None and report.get("ok"):
                self.index.paper(d)["_related"] = related

    def _public(self, d, key):
        """Paper d as the API returns it, with its related papers (Dijkstra over citation, related-work
        and similarity links) and its citation links (BFS over citations only), both among papers in memory."""
        idx = self.index
        p = {k: v for k, v in idx.paper(d).items() if not k.startswith("_")}
        p["key"] = key
        p["citation_data"] = idx.has_citation_data(d)
        p["related"] = [{"key": idx.key_of(r), "title": idx.paper(r)["title"], "year": idx.paper(r)["year"],
                         "closeness": round(float(np.exp(-dist)), 3), "link": link,
                         "via": idx.paper(via)["title"] if via != d else None}
                        for r, dist, via, link in idx.related(d, limit=self.N_RELATED)]
        cited = set(idx.cites(d))
        p["citation_links"] = [{"key": idx.key_of(r), "title": idx.paper(r)["title"], "year": idx.paper(r)["year"],
                           "hops": hops, "via": idx.paper(via)["title"] if hops > 1 else None,
                           "relation": ("reference" if r in cited else "citing") if hops == 1 else None}
                          for r, hops, via in idx.citation_links(d, max_depth=2, limit=self.N_RELATED)]
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

    def suggest_titles(self, text, limit=5):
        """Papers in memory whose title starts with what was typed (radix trie), most cited first."""
        if len(text.strip()) < 3:
            return []
        with self.lock:
            return [{"key": self.index.key_of(d), "title": self.index.paper(d)["title"],
                     "year": self.index.paper(d)["year"]} for d in self.index.titles_starting(text, limit)]

    def state(self):
        with self.lock:
            s = self.index.stats()
        s.update({"ranker": self.ranker.describe(), "lexicon": self.lex.info,
                  "online": self.client is not None, "fields": DOMAINS})
        if hasattr(self.client, "budget"):
            s["budget"] = self.client.budget()
        return s
