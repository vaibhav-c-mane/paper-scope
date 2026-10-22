# Design notes

## The idea in one paragraph

PaperScope does not keep a copy of the literature. Every search starts in a small
in-memory index of the papers the session has already fetched; when that index
has fewer than three confident matches, the query goes to OpenAlex (then
Semantic Scholar, then Crossref), and the papers that come back are added to the
index and ranked together with what was already there. The ranking itself is a
gradient-boosted tree model (LambdaMART) trained on real citation data. The only
files on disk are the trained models in `data/models/`.

```
query ─► parse (stack) ─► memory: BM25 top 200 ∪ nearest 200 vectors ─► filters ─► 9 features ─► LambdaMART
                                         │ fewer than 3 confident matches
                                         ▼
                         OpenAlex → Semantic Scholar → Crossref (rate-limited, cached in memory)
                                         │ new papers added to the index, ranking computed again
                                         ▼
                                results ordered with merge sort
```

## Data structures, and why each one

| Where | Structure | Why this one |
|---|---|---|
| `engine/memindex.py` | **Inverted index** as a sparse term × paper matrix in compressed sparse column form (posting lists of paper ids, sorted, with precomputed BM25 weights) | A query reads only the posting lists of its own words, never the text of any paper: no full-text scan. Only non-zero cells are stored. New papers always get larger ids, so before the next query their postings are appended to each word's list by one linear merge (O(P)) instead of re-sorting everything (O(P log P)); evicted papers are skipped until compaction. See docs/PROJECT.md, section 2.1. |
| `engine/memindex.py` | **Forward index** (per paper: word ids, BM25 weights, title word ids) | Lets the inverted index be rebuilt and evicted papers be removed without tokenising anything again. |
| `engine/memindex.py` | **Vector index**: one dense matrix of unit-length embeddings, searched exactly (matrix × vector, then the k nearest kept by a size-k min-heap, `dsa/heap.py`) | The index holds at most `capacity` papers (20,000 by default): 20,000 × 160 floats is 12 MB and one search is about a millisecond, so an approximate index (HNSW, IVF) would add complexity and recall loss for no gain at this size. |
| `engine/memindex.py` | **Dynamic arrays** (numpy buffers that double when full) for vectors, field probabilities and metadata | Amortised O(1) append, contiguous memory for vectorised scoring. |
| `engine/memindex.py`, `dsa/lru.py` | **LRU cache** (hash map + doubly linked list) over the papers in memory | Bounds memory however long the app runs: when the index is full, the least recently used paper is evicted in O(1). Evicted slots are compacted away once they are a quarter of the arrays. |
| `engine/sources.py`, `dsa/lru.py` | **LRU cache with expiry** for API answers (searches, single papers, and "not found") | A repeated query in the same session costs no network call; answers expire (a day for searches, an hour for "not found") because the online indexes change; size is bounded. Replaces the old SQLite cache, so nothing is written to disk. |
| `engine/memindex.py`, `dsa/hashtable.py` | **Hash table**, hand-written (polynomial hash, separate chaining, rehash at load 0.75), from identity keys (arXiv id, DOI, OpenAlex id, normalised title) to papers | The same paper from two providers becomes one entry, and missing metadata is filled in from the second copy. About 80 times slower than Python's dict per lookup, a cost of about 1 ms per online search. See docs/PROJECT.md, section 2.4. |
| `engine/pipeline.py`, `engine/memindex.py`, `dsa/heap.py` | **Binary min-heap** of size k | Keeps the 200 best BM25 scores and the 200 nearest vectors in O(n log k) without sorting all n. Slower than numpy's `argpartition` (3.3 ms against 0.03 ms for 11,866 papers), accepted to use the syllabus heap. See section 2.10. |
| `engine/memindex.py`, `dsa/bst.py` | **Binary search tree keyed by year** | A year-range filter visits only the years in range, not every paper; insert and remove follow adds and evictions, and the tree is rebuilt balanced at each compaction. See section 2.11. |
| `engine/memindex.py`, `dsa/graph.py`, `dsa/queue.py` | **Citation graph** (adjacency lists: reference lists + a map from a work to its citers) and **breadth-first search** with a **circular queue** | "Related papers" in the paper drawer: the papers in memory closest by citation links, one link away first, then two. See section 2.12. |
| `engine/lexicon.py` | **Sorted vocabulary + binary search**, plus a **hash map** word → id | A prefix query (`optim*`) or autocomplete is one contiguous range found with two binary searches; exact lookups (every word of every added paper) use the hash map, O(1) instead of O(log V). |
| `engine/ranker.py` | **Binary trees in array form** (feature, threshold, left, right, value) | The 153 ranker trees are walked for all candidates at once, keeping only the (candidate, tree) pairs not yet at a leaf. See docs/PROJECT.md, section 2.3. |
| `dsa/stack.py`, `dsa/expression.py` | **Array-based stack** | Shunting-yard (infix → postfix) and postfix evaluation of `AND / OR / NOT` queries. One stack implementation only: a Python list already gives O(1) push and pop at the end, so a linked-list stack would add nothing. |
| `dsa/merge.py` | **Two-pointer merge** of sorted lists | `AND`, `OR`, `NOT` between posting lists in O(m + n). |
| `dsa/merge.py` | **Merge sort** | Orders results (by score, newest, most cited). O(n log n) in every case and stable, so papers with equal years keep their relevance order. The other sorts (bubble, selection, insertion, quick, heap, shell, radix, counting) were removed: none is needed. |
| `engine/sources.py` | **Token bucket** per provider | Keeps each API within its rate limit (OpenAlex 8/s, Semantic Scholar 1/s, Crossref 3/s). An API that says "wait hours" (HTTP 429 with a long Retry-After) is skipped until then instead of retried, and each later provider in the cascade keeps 1.5 s of the search deadline, so one refusing API never empties the results. |

## Domain awareness

* **Field of a paper.** 13 research fields. A paper's own labels decide when it
  has them (arXiv categories, or the OpenAlex topic → subfield → field), mixed
  60/40 with the prediction of a linear classifier over TF-IDF, so cross-listed
  papers keep some probability elsewhere.
* **Field of a query.** Half from the same classifier on the query text, half
  from the fields of the ten best first-pass candidates (pseudo-relevance
  feedback). A field chosen by the user gets 80% of the weight; "only this
  field" turns it into a filter.
* **Field match** is a ranking feature: P(field | query) · P(field | paper).
* **Metadata** features (−1 = unknown, so trees can treat "unknown" separately):
  citations per year since publication (so a two-year-old paper with 200
  citations beats a twenty-year-old one with 300), recency, peer reviewed
  (journal/conference vs preprint), venue quality from OpenAlex's venue listings
  (national rankings, CCF, CWTS core journals; arXiv journal references are
  matched against a short list of flagship venues).

## The learned ranker

* **Features (9):** BM25 score / best possible BM25, embedding cosine, share of
  query words in the title, share anywhere, field match, citations, recency,
  peer reviewed, venue quality.
* **Model:** LambdaMART: boosted regression trees fit to the gradient of NDCG
  (`ml/lambdamart.py`, trees from scikit-learn, Newton leaf values, query
  subsampling, early stopping on validation NDCG@10). Trees are saved as plain
  JSON and scored by `engine/ranker.py`, all trees at once, one level per numpy
  step, so the app needs no scikit-learn model files.
* **Baseline:** a linear pairwise model (logistic regression on feature
  differences), kept to show what the trees add.
* **Calibration:** Platt scaling maps scores to a probability of relevance; the
  "confident match" threshold is where validation precision reaches 50%.

### Labels: real, not synthetic

A paper's reference list is a record of what its authors found relevant. So for
a seed paper: the query is its title, the relevant papers are the ones it cites,
and the search is run "as of" its publication year (later papers are excluded,
recency is measured from that year). This is the standard citation-recommendation
set-up. It under-counts relevance (nobody cites everything relevant), which lowers
every system's score equally, so comparisons between systems stay fair.

`ml/dataset.py` builds these labels three ways: from the OpenAlex API
(`--online`; seeds from OpenAlex's random sample, candidates from the same search
the app uses), from a pool of real OpenAlex works (`--pool`), or from your own
judgments (`--qrels`).

### How the shipped models were trained

The APIs were not reachable from the machine the models were built on, and when
they were tested live (2026-10-19) OpenAlex refused requests without an API key
(its free shared budget was used up). With a key the online route runs, but only about
1 in 7 live citation queries finds a cited paper in its top 100, so a useful online
training set needs several thousand queries. The shipped models therefore use the `--pool` route:
600,013 real works streamed from the public OpenAlex snapshot
(`scripts/fetch_openalex.py`, CC0 data), and for each seed paper the candidates
were retrieved from that pool with the same BM25 + vector search the app runs
over its memory, standing in for the OpenAlex search. Everything else (features,
labels, training) is identical to the online route. With an OpenAlex key
(`OPENALEX_API_KEY=... python -m ml.train --online`) it retrains from the live API.

The lexicon is learned from the same pool. It uses no relevance labels (only word
statistics, an SVD and the papers' own field labels), so the test queries'
citation labels never reach it.

## What was removed, and why

| Removed | Why |
|---|---|
| Bubble, selection, insertion, quick, heap, shell, radix, counting sort and the sorting race | Merge sort covers the one ordering the search needs, stably. (The heap itself came back in v3.4 for top-k selection, not as a sort.) |
| Linear, indexed-sequential and Fibonacci search, the search race | Binary search over the sorted vocabulary is the only search the engine does. |
| Linked-list stack (the array stack stays), singly/circular linked lists, deque, Josephus problem | Not used by searching, indexing or ranking. The doubly linked list survives inside the LRU cache, and a circular queue came back in v3.4 inside the BFS for related papers, where each does a real job. |
| Array address calculation, the textbook sparse-matrix class with simple/fast transpose, the dynamic-array class | Their ideas are used for real (numpy dynamic arrays, scipy/numpy CSC posting lists) instead of as demos. |
| DSA Lab page, search history, "recently viewed", "featured papers" | Demonstrations, not search. |
| Bundled hand-written sample papers and sample judgments | Synthetic. Training and evaluation use real OpenAlex papers and citation labels only. |
| Local corpus loading, SQLite cache, saving fetched papers to a file, background metadata enrichment | Everything runs online now; nothing about papers is stored on disk. |
| Optional sentence-transformers embedder | Large download (PyTorch); the LSA embeddings learned from 600k real abstracts are enough for the semantic feature and keep the app light. |
