"""The papers the engine has fetched so far, indexed in memory. Nothing is written to disk.

Every paper that comes back from OpenAlex / Semantic Scholar / Crossref is added
here, so the next search on a similar topic is answered locally in milliseconds
and only goes online when the local answer is weak.

Data structures and why:
* Inverted index (sparse term x paper matrix, compressed sparse column form):
  for each word, the sorted ids of the papers containing it and their BM25
  weights. A query reads only its own words' posting lists, never the text of
  the papers. Weights use the background Lexicon's idf, so scores do not depend
  on which papers happen to be in memory. New papers always get larger ids than
  the papers already indexed, so their postings are appended to the end of each
  word's list by one linear merge (O(P) for P postings) before the next query,
  instead of re-sorting every posting (O(P log P)). Evicted papers stay in the
  lists, are skipped when read, and are removed when the arrays are compacted.
* Forward index (per paper: its word ids and weights) to rebuild the inverted
  index and compute coverage without re-tokenising anything.
* Dynamic arrays (numpy arrays that double their capacity when full, amortised
  O(1) append) for vectors, domain probabilities and metadata columns.
* Exact vector search over the in-memory vectors: at most `capacity` papers x
  160 dimensions, one matrix-vector product, about a millisecond; the k nearest
  are picked with a size-k min-heap (dsa/heap.py), O(n log k).
* LRU cache (hash map + doubly linked list, dsa/lru.py) over paper keys: when
  more than `capacity` papers are held, the least recently used is evicted, so
  memory stays bounded however long the app runs. Evicted slots are compacted
  away once they make up a quarter of the arrays.
* Hash table from identity keys (arXiv id, DOI, OpenAlex id, title) to papers, to
  merge the same paper arriving from two sources (hand-written, separate
  chaining: dsa/hashtable.py).
* Binary search tree keyed by year (dsa/bst.py): each node holds the papers of
  one publication year, so a year-range filter visits only the years inside the
  range instead of checking every paper's year.
* Citation graph as adjacency lists: each paper's reference list (OpenAlex work
  numbers, kept in the paper) gives its outgoing edges, and a hash map from a
  work number to the papers in memory that cite it gives the incoming ones.
  Breadth-first search over it (dsa/graph.py) lists the papers closest to one
  paper by citation links, for "related papers" in the paper drawer.
"""
from collections import Counter

import numpy as np

from dsa.bst import YearTree
from dsa.graph import bfs
from dsa.hashtable import HashTable
from dsa.heap import top_k as heap_top_k
from dsa.lru import LRUCache
from engine.domains import DOMAINS, blend_with_labels, weak_label_index
from engine.metadata import columns_of
from engine.papers import complete, paper_keys
from engine.text import terms


class _Grow:
    """Dynamic array of fixed-width rows on top of numpy: append is amortised O(1)."""

    def __init__(self, width, dtype=np.float32, fill=0):
        self.width, self.dtype, self.fill = width, dtype, fill
        self.buf = np.full((16, width) if width else 16, fill, dtype)
        self.n = 0

    def extend(self, rows):
        rows = np.asarray(rows, self.dtype)
        need = self.n + len(rows)
        if need > len(self.buf):
            cap = max(need, 2 * len(self.buf))                 # doubling
            new = np.full((cap, self.width) if self.width else cap, self.fill, self.dtype)
            new[:self.n] = self.buf[:self.n]
            self.buf = new
        self.buf[self.n:need] = rows
        self.n = need

    @property
    def view(self):
        return self.buf[:self.n]

    def keep(self, idx):
        rows = self.buf[idx].copy()
        self.n = 0
        self.buf = np.full((max(16, len(rows)), self.width) if self.width else max(16, len(rows)), self.fill, self.dtype)
        self.extend(rows)


class PaperIndex:
    def __init__(self, lexicon, capacity=20000, label_weight=0.6):
        self.lex = lexicon
        self.capacity = capacity
        self.label_weight = label_weight
        self.papers = []                    # internal id -> paper dict (None once evicted)
        self.fwd = []                       # internal id -> (term ids, BM25 weights, title term ids)
        self.key_to_doc = HashTable()       # identity key -> internal id (separate chaining, dsa/hashtable.py)
        self.extra = {}                     # words missing from the lexicon -> ids after the lexicon's
        self.vectors = _Grow(max(1, lexicon.dim))
        self.domains = _Grow(len(DOMAINS))
        self.meta = _Grow(4)                # year, citations, peer_reviewed, venue_quality (engine/metadata.py)
        self.alive = _Grow(0, bool, False)
        self.by_year = YearTree()           # year -> papers of that year (binary search tree, dsa/bst.py)
        self.citers = {}                    # OpenAlex work number -> ids of the papers in memory that cite it
        self.lru = LRUCache(capacity, on_evict=self._evict) if capacity else None
        self.dead = 0
        self.generation = 0                 # changes whenever ids or contents change
        self.added = 0                      # papers that were new in the last add()
        self._csc = None                    # inverted index arrays, see _postings()
        self._indexed = 0                   # papers 0.._indexed-1 are in self._csc
        self._dead_indexed = 0              # evicted papers whose postings are still in self._csc
        self._df_cache = (None, None)       # (generation, df over papers in memory)

    def __len__(self):
        return len(self.papers) - self.dead

    @property
    def n(self):
        return len(self.papers)

    # ------------------------------------------------------------- term ids
    def _term(self, word, create=False):
        i = self.lex.term_id(word)
        if i >= 0:
            return i
        j = self.extra.get(word)
        if j is None and create:
            j = self.extra[word] = len(self.lex.vocab) + len(self.extra)
        return -1 if j is None else j

    def idf(self, ids):
        ids = np.asarray(ids, np.int64)
        known = ids < len(self.lex.vocab)
        out = np.full(len(ids), self.lex.idf_unknown, np.float32)
        out[known] = self.lex.idf[ids[known]]
        return out

    def expand(self, word, max_terms=40):
        """Query word -> term ids; 'optim*' -> every known word with that prefix
        (binary search for the range in the sorted lexicon, plus words seen only here)."""
        if not word.endswith("*"):
            i = self._term(word)
            return [] if i < 0 else [i]
        prefix = word[:-1]
        lo, hi = self.lex.prefix_range(prefix)
        ids = list(range(lo, hi)) + [j for w, j in self.extra.items() if w.startswith(prefix)]
        if len(ids) > max_terms:
            df = self._df()
            ids = sorted(ids, key=lambda i: -(df[i] if i < len(df) else 0))[:max_terms]
        return ids

    def query_ids(self, words):
        weights = Counter()
        for w in words:
            for i in self.expand(w):
                weights[i] += 1.0
        ids = sorted(weights)
        return ids, np.asarray([weights[i] for i in ids], np.float32)

    def ceiling(self, ids, weights):
        """Best BM25 score any paper could reach for these terms (scales BM25 to 0..1)."""
        if not len(ids):
            return 1.0
        return float(np.dot(self.idf(ids), weights) * (self.lex.k1 + 1)) or 1.0

    # ------------------------------------------------------------- adding papers
    def find(self, key):
        return self.key_to_doc.get(key)

    def add(self, papers):
        """Add papers (deduplicated by identity keys). Returns their internal ids in input
        order; ids stay valid until the next call to add()."""
        self.maintain()
        out, fresh = [], []
        for p in papers:
            p = complete(p)
            if not p["title"]:
                continue
            ks = paper_keys(p)
            if p.get("openalex_id"):
                ks.append("openalex:" + p["openalex_id"].rsplit("/", 1)[-1].lower())
            d = next((self.key_to_doc[k] for k in ks if k in self.key_to_doc), None)
            if d is not None:
                if d < self.n:                 # (d >= n: a duplicate inside this same batch)
                    self._fill_gaps(d, p, ks)
                out.append(d)
                continue
            d = self.n + len(fresh)
            for k in ks:
                self.key_to_doc.setdefault(k, d)
            p["_keys"] = ks
            p["_refs"] = list(p.pop("refs", None) or [])
            fresh.append(p)
            out.append(d)
        for d in out:                      # papers seen again become recent before anything is evicted
            self.touch(d)
        self.added = len(fresh)
        if fresh:
            first = self.n
            self._append(fresh)
            if self.lru is not None:
                for d in range(first, self.n):
                    self.lru.put(self.papers[d]["_keys"][0], d)
        return out

    def maintain(self):
        """Compact evicted slots once they are a quarter of the arrays (renumbers papers)."""
        if self.dead > max(64, self.n // 4):
            self._compact()

    def _fill_gaps(self, d, p, ks):
        """Same paper from another source: add identity keys and metadata it was missing."""
        old = self.papers[d]
        if old is None:
            return
        for k in ks:
            if k not in self.key_to_doc:
                self.key_to_doc[k] = d
                old["_keys"].append(k)
        changed = False
        for field in ("citations", "venue", "published", "venue_quality", "doi", "oa_url", "openalex_id", "topic"):
            if old.get(field) in (None, "") and p.get(field) not in (None, ""):
                old[field] = p[field]
                changed = True
        if len(p.get("abstract") or "") > len(old.get("abstract") or ""):
            old["abstract"] = p["abstract"]
        if p.get("refs") and not old["_refs"]:          # e.g. first seen on Semantic Scholar, now on OpenAlex
            old["_refs"] = list(p["refs"])
            self._link(d)
        if changed:
            year = int(self.meta.buf[d, 0])
            self.meta.buf[d] = columns_of(old)
            if int(self.meta.buf[d, 0]) != year:
                self.by_year.remove(year, d)
                self.by_year.insert(int(self.meta.buf[d, 0]), d)
            self.generation += 1

    def _append(self, fresh):
        first = self.n
        lex = self.lex
        toks = [(terms(p["title"]), terms(p["abstract"])) for p in fresh]
        vecs, probs = lex.text_features(toks)
        labels = [weak_label_index(p) for p in fresh]
        for (tt, bt), p in zip(toks, fresh):
            c = Counter()
            for t in tt:
                c[t] += lex.title_weight
            for t in bt:
                c[t] += 1.0
            ids = np.asarray([self._term(t, create=True) for t in c], np.int64)
            tf = np.asarray(list(c.values()), np.float32)
            dl = float(tf.sum())
            norm = lex.k1 * (1 - lex.b + lex.b * dl / (lex.avgdl or 1.0))
            w = tf * (lex.k1 + 1) / (tf + norm) * self.idf(ids)
            order = np.argsort(ids)
            title_ids = np.unique(np.asarray([self._term(t) for t in set(tt)], np.int64))
            self.fwd.append((ids[order], w[order].astype(np.float32), title_ids))
            self.papers.append(p)
        self.vectors.extend(vecs)
        self.domains.extend(blend_with_labels(probs, labels, self.label_weight))
        self.meta.extend([columns_of(p) for p in fresh])
        self.alive.extend(np.ones(len(fresh), bool))
        for d in range(first, self.n):
            self.by_year.insert(int(self.meta.buf[d, 0]), d)
            self._link(d)
        self.generation += 1                # the new postings are merged in by the next _postings()

    # ------------------------------------------------------------- recency / eviction
    def touch(self, d):
        p = self.papers[d] if 0 <= d < self.n else None
        if p is not None and self.lru is not None:
            self.lru.touch(p["_keys"][0])

    def _evict(self, key, d):
        p = self.papers[d]
        if p is None:
            return
        for k in p["_keys"]:
            if self.key_to_doc.get(k) == d:
                del self.key_to_doc[k]
        self.by_year.remove(int(self.meta.buf[d, 0]), d)
        self.papers[d] = None
        self.fwd[d] = (np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.int64))
        self.alive.buf[d] = False
        self.dead += 1
        if d < self._indexed:               # its postings stay in the arrays until the next compaction
            self._dead_indexed += 1
        self.generation += 1

    def _compact(self):
        """Drop evicted slots and renumber the remaining papers 0..n-1."""
        keep = np.flatnonzero(self.alive.view)
        new_id = {int(old): new for new, old in enumerate(keep)}
        self.papers = [self.papers[i] for i in keep]
        self.fwd = [self.fwd[i] for i in keep]
        for g in (self.vectors, self.domains, self.meta, self.alive):
            g.keep(keep)
        self.by_year = YearTree.balanced((int(y), d) for d, y in enumerate(self.meta.view[:, 0]))
        self.citers = {}
        for d in range(self.n):
            self._link(d)
        renumbered = HashTable()
        for k, d in self.key_to_doc.items():
            if d in new_id:
                renumbered[k] = new_id[d]
        self.key_to_doc = renumbered
        if self.lru is not None:
            entries = [(k, self.lru.pop(k)) for k in self.lru.keys()]     # oldest first, so the order is kept
            for k, d in entries:
                if d in new_id:
                    self.lru.put(k, new_id[d])
        self.dead = 0
        self._csc = None                    # ids changed: the next query rebuilds the inverted index
        self._indexed = self._dead_indexed = 0
        self.generation += 1

    # ------------------------------------------------------------- citation graph
    def _link(self, d):
        """Record paper d as a citer of every work in its reference list."""
        for n in self.papers[d]["_refs"]:
            self.citers.setdefault(n, []).append(d)

    @staticmethod
    def _work_number(p):
        k = next((k for k in p["_keys"] if k.startswith("openalex:w")), None)
        return int(k[len("openalex:w"):]) if k and k[len("openalex:w"):].isdigit() else None

    def cites(self, d):
        """Papers in memory that paper d cites."""
        p = self.papers[d]
        found = [self.key_to_doc.get("openalex:w%d" % n) for n in p["_refs"]] if p else []
        return [c for c in found if c is not None and c != d and self.alive.buf[c]]

    def neighbours(self, d):
        """Papers in memory joined to paper d by a citation: the ones it cites, then the ones citing it."""
        p = self.papers[d]
        if p is None:
            return []
        alive = self.alive.buf
        out = self.cites(d)
        n = self._work_number(p)
        out += [c for c in self.citers.get(n, ()) if alive[c]] if n is not None else []
        return [c for c in out if c != d]

    def related(self, d, max_depth=2, limit=10):
        """[(id, hops, via)]: the papers closest to d in the citation graph, by breadth-first search."""
        return bfs(d, self.neighbours, max_depth=max_depth, limit=limit)

    def has_citation_data(self, d):
        p = self.papers[d]
        return bool(p and (p["_refs"] or self.citers.get(self._work_number(p))))

    # ------------------------------------------------------------- inverted index
    def _build(self, lo, hi, n_terms):
        """Inverted index (CSC arrays) of papers lo..hi-1 only, sorted by word, then by paper."""
        fwd = self.fwd[lo:hi]
        doc_ids = np.arange(lo, hi, dtype=np.int64)
        lens = np.asarray([len(f[0]) for f in fwd], np.int64)
        docs = np.repeat(doc_ids, lens)
        ids = np.concatenate([f[0] for f in fwd]) if fwd else np.zeros(0, np.int64)
        ws = np.concatenate([f[1] for f in fwd]) if fwd else np.zeros(0, np.float32)
        order = np.lexsort((docs, ids))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(ids, minlength=n_terms))])
        tlens = np.asarray([len(f[2]) for f in fwd], np.int64)
        tdocs = np.repeat(doc_ids, tlens)
        tids = np.concatenate([f[2] for f in fwd]) if fwd else np.zeros(0, np.int64)
        torder = np.lexsort((tdocs, tids))
        tindptr = np.concatenate([[0], np.cumsum(np.bincount(tids, minlength=n_terms))])
        return indptr, docs[order], ws[order], tindptr, tdocs[torder]

    @staticmethod
    def _append_lists(indptr_a, cols_a, indptr_b, cols_b):
        """Per word, list a followed by list b, for every word at once: O(len(a) + len(b)).
        Entry k of word t in a moves right by the number of b entries of the words before t;
        entry j of word t in b lands after all of word t's entries in a."""
        n_terms = len(indptr_b) - 1
        indptr_a = np.concatenate([indptr_a, np.full(n_terms + 1 - len(indptr_a), indptr_a[-1])])  # new words
        pos_a = np.arange(indptr_a[-1]) + np.repeat(indptr_b[:-1], np.diff(indptr_a))
        pos_b = np.arange(indptr_b[-1]) + np.repeat(indptr_a[1:], np.diff(indptr_b))
        out = []
        for a, b in zip(cols_a, cols_b):
            merged = np.empty(len(a) + len(b), a.dtype)
            merged[pos_a], merged[pos_b] = a, b
            out.append(merged)
        return indptr_a + indptr_b, out

    def _postings(self):
        """Inverted index as flat arrays (indptr, docs, weights, title indptr, title docs): the
        papers containing word t are docs[indptr[t]:indptr[t+1]], in increasing order."""
        n_terms = len(self.lex.vocab) + len(self.extra) + 1
        if self._csc is None:
            self._csc = self._build(0, self.n, n_terms)
            self._dead_indexed = 0          # evicted papers have empty forward entries: nothing indexed
        elif self._indexed < self.n:        # papers added since: their ids are larger, so append
            indptr, docs, ws, tindptr, tdocs = self._csc
            new = self._build(self._indexed, self.n, n_terms)
            indptr2, (docs, ws) = self._append_lists(indptr, (docs, ws), new[0], (new[1], new[2]))
            tindptr2, (tdocs,) = self._append_lists(tindptr, (tdocs,), new[3], (new[4],))
            self._csc = (indptr2, docs, ws, tindptr2, tdocs)
        self._indexed = self.n
        return self._csc

    def _df(self):
        """Papers in memory containing each word (evicted papers still in the arrays not counted)."""
        indptr, docs = self._postings()[:2]
        if not self._dead_indexed:
            return np.diff(indptr)
        gen, df = self._df_cache
        if gen != self.generation:
            words = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
            df = np.bincount(words, weights=self.alive.view[docs], minlength=len(indptr) - 1).astype(np.int64)
            self._df_cache = (self.generation, df)
        return df

    def postings(self, t):
        indptr, docs, _, _, _ = self._postings()
        out = docs[indptr[t]:indptr[t + 1]] if t + 1 < len(indptr) else docs[:0]
        return out[self.alive.view[out]] if self._dead_indexed else out

    def docs_with(self, word):
        """Sorted ids of the papers containing a word (prefix* allowed)."""
        ids = self.expand(word)
        if not ids:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate([self.postings(i) for i in ids]))

    def bm25(self, ids, weights):
        """BM25 score of every paper in memory, term at a time over the posting lists."""
        indptr, docs, ws, _, _ = self._postings()
        acc = np.zeros(self.n, np.float32)
        for t, w in zip(ids, weights):
            if t + 1 < len(indptr):
                lo, hi = indptr[t], indptr[t + 1]
                acc[docs[lo:hi]] += w * ws[lo:hi]
        if self._dead_indexed:
            acc[~self.alive.view] = 0.0     # evicted papers still in the lists score nothing
        return acc

    def coverage(self, groups, docs):
        """Share of the query words found in each candidate's title, and anywhere in it."""
        indptr, pdocs, _, tindptr, tdocs = self._postings()
        docs = np.asarray(docs, np.int64)
        title_cov, term_cov = np.zeros(len(docs), np.float32), np.zeros(len(docs), np.float32)
        if not groups or not len(docs):
            return title_cov, term_cov
        for g in groups:
            in_doc, in_title = np.zeros(len(docs), bool), np.zeros(len(docs), bool)
            for t in g:
                if t + 1 < len(indptr):
                    in_doc |= np.isin(docs, pdocs[indptr[t]:indptr[t + 1]], assume_unique=False)
                    in_title |= np.isin(docs, tdocs[tindptr[t]:tindptr[t + 1]], assume_unique=False)
            title_cov += in_title
            term_cov += in_doc
        return title_cov / len(groups), term_cov / len(groups)

    # ------------------------------------------------------------- vectors
    def search_vectors(self, q, k, allowed=None):
        """Exact nearest neighbours: (ids, cosine) of the k most similar papers in memory."""
        if not self.n or q is None:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        s = self.vectors.view @ q
        ok = self.alive.view if allowed is None else (self.alive.view & allowed)
        cand = np.flatnonzero(ok)
        idx = np.asarray(heap_top_k(s[cand].tolist(), cand.tolist(), k), np.int64)   # size-k min-heap
        return idx, s[idx]

    # ------------------------------------------------------------- access
    def paper(self, d):
        return self.papers[d]

    def years_between(self, lo, hi):
        """Ids of the papers in memory with lo <= year <= hi (year 0 = unknown)."""
        return self.by_year.range(lo, hi)

    def key_of(self, d):
        p = self.papers[d]
        return p["_keys"][0] if p else None

    def stats(self):
        df = self._df()
        return {"papers_in_memory": len(self), "capacity": self.capacity, "postings": int(df.sum()),
                "words_in_memory": int((df > 0).sum()), "words_not_in_lexicon": len(self.extra),
                "vector_dim": int(self.vectors.width)}
