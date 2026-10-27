"""Background language statistics, learned once from real papers and saved with the models.

Search results come from the internet a few dozen at a time, so statistics
computed from them alone would be useless: every fetched paper contains the
query words, which would make those words look common (idf near 0). Instead the
engine uses statistics of a large real corpus (OpenAlex works, see ml/train.py),
saved as plain arrays in data/models/lexicon/:

  vocabulary + document frequencies -> BM25 idf of every word
  TF-IDF columns (the most informative words) and their idf
  LSA projection (TF-IDF -> truncated SVD)  -> paper and query embeddings
  domain classifier weights                 -> P(research domain | text)

This is a model, not a copy of the papers: no titles or abstracts are stored.

Data structures: the vocabulary is a sorted array, so a prefix query ("optim*")
is one contiguous range found by binary search (bisect), plus a hash map from
word to id for exact lookups, O(1) on average; the TF-IDF matrices are sparse (CSR) because a text uses a
tiny fraction of the vocabulary.
"""
import bisect
import json
import math
import os
from collections import Counter

import numpy as np
from scipy import sparse

from engine.domains import DomainClassifier, weak_label_index
from engine.text import terms


class CountBuilder:
    """Collects (paper, word, count) while papers stream in, in compact numpy chunks
    (a few bytes per entry instead of a Python object), so 1M abstracts fit in memory."""

    def __init__(self, title_weight=2.0, flush_every=20000):
        self.title_weight = title_weight
        self.term_id = {}                       # word -> provisional id (build time only)
        self.n_docs = 0
        self.flush_every = flush_every
        self._cols, self._vals, self._lens = [], [], []
        self.chunks = []                        # (lens, cols, vals) numpy arrays

    def _id(self, t):
        i = self.term_id.get(t)
        if i is None:
            i = self.term_id[t] = len(self.term_id)
        return i

    def add(self, title_toks, body_toks):
        counts = Counter()
        for t in title_toks:
            counts[t] += self.title_weight
        for t in body_toks:
            counts[t] += 1.0
        self._cols += [self._id(t) for t in counts]
        self._vals += list(counts.values())
        self._lens.append(len(counts))
        self.n_docs += 1
        if len(self._lens) >= self.flush_every:
            self._flush()

    def _flush(self):
        if self._lens:
            self.chunks.append((np.asarray(self._lens, np.int64), np.asarray(self._cols, np.int64),
                                np.asarray(self._vals, np.float32)))
        self._cols, self._vals, self._lens = [], [], []

    def finish(self):
        """(sorted vocabulary, term-frequency CSR matrix papers x words)."""
        self._flush()
        vocab = sorted(self.term_id)
        remap = np.empty(max(1, len(vocab)), np.int64)          # provisional id -> sorted id
        for new, t in enumerate(vocab):
            remap[self.term_id[t]] = new
        self.term_id = None
        width = max(1, len(vocab))
        parts = []
        for lens, cols, vals in self.chunks:
            indptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            m = sparse.csr_matrix((vals, remap[cols] if len(cols) else cols, indptr), shape=(len(lens), width))
            m.sort_indices()
            parts.append(m)
        self.chunks = []
        tf = sparse.vstack(parts, format="csr") if parts else sparse.csr_matrix((0, width), dtype=np.float32)
        return vocab, tf


class Lexicon:
    def __init__(self, vocab, df, n_docs, avgdl, cols, col_idf, lsa=None, classifier=None,
                 k1=1.2, b=0.75, title_weight=2.0, info=None):
        self.vocab = vocab                          # sorted list of words: prefix ranges by binary search
        self.ids = {w: i for i, w in enumerate(vocab)}   # hash map word -> id: O(1) exact lookups
        self.df = np.asarray(df, np.int64)
        self.n_docs, self.avgdl = n_docs, avgdl
        self.k1, self.b, self.title_weight = k1, b, title_weight
        self.idf = np.log1p((n_docs - self.df + 0.5) / (self.df + 0.5)).astype(np.float32)   # BM25 (Lucene)
        self.idf_unknown = float(math.log1p((n_docs - 1 + 0.5) / 1.5))       # a word seen once at most
        self.cols = np.asarray(cols, np.int64)      # vocabulary ids of the TF-IDF columns (sorted)
        self.col_idf = np.asarray(col_idf, np.float32)
        self.lsa = lsa                              # (len(cols), dim) term -> latent space
        self.classifier = classifier or DomainClassifier()
        self.info = info or {}

    # ------------------------------------------------------------- vocabulary
    def term_id(self, word):
        """Exact lookup, done for every word of every paper added: a hash map, O(1) on average
        (binary search in the sorted vocabulary would be O(log V), about 18 string comparisons)."""
        return self.ids.get(word, -1)

    def prefix_range(self, prefix):
        return bisect.bisect_left(self.vocab, prefix), bisect.bisect_left(self.vocab, prefix + "￿")

    @property
    def dim(self):
        return 0 if self.lsa is None else self.lsa.shape[1]

    def word_vector(self, term):
        """A word's own direction in the LSA space (its idf-weighted row of the projection), unit
        length; None for a word that is not one of the TF-IDF columns."""
        if self.lsa is None:
            return None
        i = self.term_id(term)
        j = int(np.searchsorted(self.cols, i)) if i >= 0 else -1      # cols is sorted: binary search
        if j < 0 or j >= len(self.cols) or self.cols[j] != i:
            return None
        v = self.lsa[j] * self.col_idf[j]
        n = float(np.linalg.norm(v))
        return v / n if n else None

    # ------------------------------------------------------------- text -> vectors
    def count_rows(self, docs):
        """docs: list of (title tokens, body tokens) -> CSR counts over the vocabulary
        (title words count `title_weight` times; unknown words are dropped)."""
        rows, cols, vals = [], [], []
        for r, (tt, bt) in enumerate(docs):
            c = {}
            for t in tt:
                c[t] = c.get(t, 0.0) + self.title_weight
            for t in bt:
                c[t] = c.get(t, 0.0) + 1.0
            for t, v in c.items():
                i = self.term_id(t)
                if i >= 0:
                    rows.append(r)
                    cols.append(i)
                    vals.append(v)
        return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                                 shape=(len(docs), max(1, len(self.vocab))))

    def tfidf(self, counts):
        """CSR counts over the vocabulary -> L2-normalised sublinear TF-IDF over the columns."""
        x = sparse.csr_matrix(counts[:, self.cols], dtype=np.float32)
        x.data = (1.0 + np.log(x.data)) * self.col_idf[x.indices]
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sparse.csr_matrix(sparse.diags(1.0 / norms) @ x, dtype=np.float32)

    def embed(self, x):
        """TF-IDF rows -> unit-length LSA vectors (cosine similarity = dot product)."""
        if self.lsa is None or x.shape[0] == 0:
            return np.zeros((x.shape[0], max(1, self.dim)), np.float32)
        v = np.asarray(x @ self.lsa, np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n

    def text_features(self, docs):
        """(vectors, domain probabilities) for a list of (title tokens, body tokens)."""
        x = self.tfidf(self.count_rows(docs))
        return self.embed(x), self.classifier.predict_proba(x)

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, papers, dim=160, max_columns=30000, min_df=2, max_df=0.6, vocab_min_df=2,
              svd_sample=150000, clf_sample=200000, seed=0, progress=print):
        """Learn the statistics from an iterable of real papers (title + abstract)."""
        from sklearn.decomposition import TruncatedSVD

        cb, labels = CountBuilder(), []
        for p in papers:
            cb.add(terms(p.get("title", "")), terms(p.get("abstract", "")))
            labels.append(weak_label_index(p))
            if progress and cb.n_docs % 100000 == 0:
                progress(f"  lexicon: read {cb.n_docs:,} papers")
        vocab, tf = cb.finish()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        avgdl = float(tf.sum() / max(1, n))
        # TF-IDF columns: informative words (not too rare, not in most papers), the most frequent first
        ok = np.flatnonzero((df >= min_df) & (df <= max_df * n)) if n >= 50 else np.flatnonzero(df > 0)
        ok = np.sort(ok[np.argsort(-df[ok], kind="stable")[:max_columns]])
        col_idf = (np.log((1.0 + n) / (1.0 + df[ok])) + 1.0).astype(np.float32)
        # the saved vocabulary drops words seen in only one paper: their idf equals idf_unknown anyway
        keep = np.flatnonzero(df >= vocab_min_df) if n >= 50 else np.arange(len(vocab))
        keep = np.union1d(keep, ok)
        remap = np.full(len(vocab), -1, np.int64)
        remap[keep] = np.arange(len(keep))
        lex = cls([vocab[i] for i in keep], df[keep], n, avgdl, remap[ok], col_idf,
                  info={"papers": n, "words_seen": len(vocab)})
        x = lex.tfidf(tf[:, keep])
        rng = np.random.default_rng(seed)
        k = max(1, min(dim, x.shape[1] - 1, n - 1))
        rows = rng.choice(n, min(n, svd_sample), replace=False)
        svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=seed).fit(x[rows])
        lex.lsa = svd.components_.T.astype(np.float32)
        labels = np.asarray(labels)
        lab_rows = np.flatnonzero(labels >= 0)
        if len(set(labels[lab_rows].tolist())) >= 2:
            lab_rows = rng.choice(lab_rows, min(len(lab_rows), clf_sample), replace=False)
            lex.classifier = DomainClassifier().fit(x[lab_rows], labels[lab_rows], seed=seed)
        lex.info.update({"dim": int(k), "columns": int(len(ok)), "vocabulary": len(keep),
                         "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 3)})
        lex.train_x, lex.train_labels = x, labels      # kept for ml/train.py's held-out evaluation
        return lex

    # ------------------------------------------------------------- save / load
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.vocab))
        np.save(os.path.join(folder, "df.npy"), self.df.astype(np.int32))
        np.save(os.path.join(folder, "tfidf_cols.npy"), self.cols.astype(np.int32))
        np.save(os.path.join(folder, "tfidf_idf.npy"), self.col_idf)
        if self.lsa is not None:
            np.save(os.path.join(folder, "lsa.npy"), self.lsa.astype(np.float16))   # half precision: half the size
        self.classifier.save(folder)
        with open(os.path.join(folder, "lexicon.json"), "w", encoding="utf-8") as f:
            json.dump({"n_docs": self.n_docs, "avgdl": self.avgdl, "k1": self.k1, "b": self.b,
                       "title_weight": self.title_weight, "info": self.info}, f, indent=1)

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, "lexicon.json"), encoding="utf-8") as f:
            p = json.load(f)
        with open(os.path.join(folder, "vocab.txt"), encoding="utf-8") as f:
            vocab = f.read().split("\n")
        lsa_path = os.path.join(folder, "lsa.npy")
        lsa = np.load(lsa_path).astype(np.float32) if os.path.isfile(lsa_path) else None
        return cls(vocab, np.load(os.path.join(folder, "df.npy")), p["n_docs"], p["avgdl"],
                   np.load(os.path.join(folder, "tfidf_cols.npy")), np.load(os.path.join(folder, "tfidf_idf.npy")),
                   lsa, DomainClassifier.load(folder, mmap=False), p["k1"], p["b"], p["title_weight"], p.get("info"))


def lexicon_dir(models_dir):
    return os.path.join(models_dir, "lexicon")
