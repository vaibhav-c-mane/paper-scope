# PaperScope: project documentation

This document describes PaperScope as the code in this folder implements it
(version 4.1, 27 October 2026). Part 1 explains what the project does and how it
is built, including a plain explanation of the learned ranker (section 11), of
how everyday queries are understood (section 12, new in 3.5), how the match
shown next to each result is computed (section 13, new in 3.6), and how long
queries, related papers and title suggestions work (sections 14 and 15, new in
4.0), and how the free API budget is saved (section 16, new in 4.1). Part 2 justifies each core data structure, compares it with alternatives
on small worked examples, records the three changes made after measuring them,
and describes the hand-written structures added so that every syllabus unit
does real work in the project: a heap, a hash table, a binary search tree and
breadth-first search with a circular queue (3.4); an indexed priority queue
with Dijkstra's shortest paths, and a radix trie (4.0); a log-structured hash
table for the optional cache file (4.1).

Every number in this file was measured on the cloud machine used for the live
tests, unless the text says otherwise. File references point to the code, so
each statement can be checked.

---

# Part 1: The project

## 1. Purpose

PaperScope is a research paper search engine built for a data structures and
algorithms course. You type a topic ("graph neural networks for molecules"), a
boolean query (`deep AND (vision OR image) NOT survey`) or an identifier (a DOI,
an arXiv id or an OpenAlex id), and it returns ranked papers with their year,
venue, authors, citation count and abstract. The topic can also be typed the way
people ask a librarian, with typos ("I want research papers on how to make
nueral networks smaller"): PaperScope first works out what to search for
(section 12).

It does not ship a copy of the literature. Papers come from three free online
scholarly APIs: OpenAlex, then Semantic Scholar, then Crossref. Every paper
fetched during a session is kept in an in-memory index, so a later search on a
similar topic is answered from memory in milliseconds, without asking the
internet again. Nothing about papers is written to disk. The only files the
running app reads are the trained models in `data/models/`.

The results are ordered by a learned ranker (gradient-boosted regression trees
trained with LambdaMART) that combines nine signals: word match, meaning
similarity, title and term coverage, research-field match, citations, recency,
peer-review status and venue quality (section 11 explains it step by step).

Opening a paper shows its related papers: the papers in memory joined to it by
citations, closest first, found by breadth-first search over the citation graph.

## 2. Architecture

PaperScope has four layers. Each layer only calls the one below it.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Interfaces   static/ (web page)   app.py (Flask JSON API)                │
 │              scripts/search.py (command line)                            │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ Pipeline     engine/pipeline.py  SearchPipeline: parse, retrieve,        │
 │              filter, compute features, rank, decide whether to go online │
 ├───────────────────────────────┬──────────────────────────────────────────┤
 │ In-memory index               │ Online sources                           │
 │ engine/memindex.py PaperIndex │ engine/sources.py ScholarlyClient        │
 │ engine/lexicon.py  Lexicon    │ OpenAlex → Semantic Scholar → Crossref   │
 │ engine/ranker.py   rankers    │ rate limits, retries, cache (LRU + TTL)  │
 │ engine/features.py, metadata  │                                          │
 ├───────────────────────────────┴──────────────────────────────────────────┤
 │ Syllabus structures  dsa/  stack, infix→postfix, merge + merge sort, LRU │
 │                            heap, hash table, year BST, queue + BFS       │
 └──────────────────────────────────────────────────────────────────────────┘
 Offline, not used while searching:  ml/ (training, evaluation), scripts/fetch_openalex.py
```

One search follows this path (`engine/pipeline.py`, `SearchPipeline.rank` and `search`):

```
query text
  │ QueryUnderstanding.understand (not for identifiers, boolean queries or exact=1):
  │   spelling, request words dropped, "recent" → year filter, by meaning? by title?
  │   a fewer-words version without the least related words (long queries)
  ▼
  │ parse_query: identifier? boolean operators? words, -exclusions, prefix*
  ▼
filters: year range (year BST), author, field, exclusions, AND/OR/NOT result set
  ▼
candidates from memory:  BM25 top 200 (inverted index)  ∪  nearest 200 vectors
                         (each top 200 kept by a size-200 min-heap)
  ▼
9 features per candidate  ─►  ranker score + relevance (0..1)
  ▼
fewer than 3 candidates with relevance ≥ threshold (0.396 for the shipped model)?
  │ yes ─► ScholarlyClient.search (≤ 50 papers)            ┐ in parallel
  │       + OpenAlex by meaning (a sentence, ≤ 50 papers)    │
  │       + OpenAlex with fewer words (long query, ≤ 50)     │
  │       + OpenAlex by title, most cited (a phrase, ≤ 25)   ┘
  │       ─► PaperIndex.add ─► rank again ─► fuse with the by-meaning and fewer-words orders
  │       ─► a well-cited paper whose title is the query goes first
  ▼
rows sorted by score with merge sort (re-sorted by year or citations on request)
  ▼
one page of results + a trace of every step (shown in the web page's side panel)
```

## 3. Directory structure

```
paper-search/
├── app.py                  Flask web server and JSON API (/api/search, /api/paper, /api/suggest, /api/state)
├── requirements.txt        flask, numpy, scipy, scikit-learn (scikit-learn is needed for training only)
├── README.md               how to install, run, train; results; live test
├── RUN_ON_WINDOWS.md       step-by-step Windows instructions
├── .env.example            template for API keys and budget settings (copy to .env, which stays private)
├── dsa/                    the syllabus data structures, written by hand
│   ├── stack.py            array-based stack
│   ├── expression.py       tokenizer, infix → postfix (shunting-yard), postfix evaluation
│   ├── merge.py            merge of sorted lists, merge sort, AND / OR / NOT on sorted id lists
│   ├── lru.py              LRU cache: hash map + doubly linked list, optional expiry time
│   ├── heap.py             binary min-heap on an array; top-k selection
│   ├── hashtable.py        hash table: polynomial hash, separate chaining, rehashing
│   ├── bst.py              binary search tree keyed by year; range queries
│   ├── queue.py            circular queue on an array
│   ├── priority_queue.py   indexed min-heap: priorities with decrease-key
│   ├── graph.py            breadth-first search and Dijkstra's shortest paths over adjacency lists
│   ├── trie.py             radix trie (compressed prefix tree)
│   └── logstore.py         log-structured hash table: append-only file + hash map of offsets
├── engine/                 the search engine
│   ├── pipeline.py         SearchPipeline: the whole search flow
│   ├── understand.py       QueryUnderstanding: spelling, request words, recency, query kind
│   ├── memindex.py         PaperIndex: inverted index, vectors, metadata, eviction
│   ├── lexicon.py          Lexicon: vocabulary, idf, TF-IDF columns, LSA embeddings
│   ├── sources.py          OpenAlex, arXiv, Europe PMC, Semantic Scholar, Crossref clients, the cascade, budget
│   ├── ranker.py           GBDT (trees), linear and hand-set rankers
│   ├── features.py         the nine ranking features
│   ├── metadata.py         citations, recency, peer review, venue quality as 0..1 signals
│   ├── domains.py          13 research fields, weak labels, field classifier
│   ├── papers.py           one paper record format for every source
│   └── text.py             tokenizer, stop words, small stemmer
├── ml/                     offline training and evaluation (not used while searching)
│   ├── dataset.py          citation-based relevance labels (online, pool or your own judgments)
│   ├── lambdamart.py       LambdaMART training on scikit-learn regression trees
│   ├── train.py            python -m ml.train: lexicon + rankers → data/models/
│   ├── evaluate.py         python -m ml.evaluate: NDCG@10, MRR, P@10, R@50
│   └── metrics.py          ranking metrics
├── data/models/            trained models (15 MB): lexicon/ arrays, ranker.json, ranker_linear.json, eval_report.json
├── scripts/
│   ├── search.py           command-line search with the same pipeline
│   ├── fetch_openalex.py   streams real papers from the OpenAlex snapshot for offline training
│   ├── ds_examples.py      reproduces the worked examples in Part 2 of this document
│   └── check_queries.py    26 everyday queries against live OpenAlex (search quality, section 12)
├── static/                 index.html, app.js, style.css (the web page)
├── tests/                  97 unit tests, no network needed (fake HTTP transport)
└── docs/                   DESIGN.md, PROJECT.md (this file), screenshots
```

## 4. Main modules and their responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | Creates one `SearchPipeline` with a `ScholarlyClient`, validates request parameters and returns JSON. Memory limit from `PAPERSCOPE_CAPACITY` (default 20,000 papers). |
| `static/app.js` | The single page: search form, result list, side panel ("trace") that shows how the search worked, paper drawer, word suggestions as you type. |
| `scripts/search.py` | The same pipeline from the terminal. Several queries in one call share the in-memory index, so the second can be answered from memory. |
| `engine/pipeline.py` | `parse_query` splits the query. `PreparedQuery` computes everything that depends only on the query (word ids, embedding, field distribution). `SearchPipeline` retrieves candidates, filters, computes features, ranks, decides whether to go online, sorts and pages the results, looks papers up by identifier and lists a paper's related papers. A re-entrant lock protects the shared index; the network call happens outside the lock. |
| `engine/understand.py` | `QueryUnderstanding` turns the typed text into what to search for: corrects typos with a noisy-channel spelling model whose word counts are the Lexicon's document frequencies, drops request words ("find me papers on"), turns "recent" / "latest" into a year filter, and says whether to also search by meaning or by title. |
| `engine/memindex.py` | `PaperIndex` holds every fetched paper: forward index, inverted index with BM25 weights, embedding matrix, field probabilities, metadata columns, the identity-key hash table, the year tree, the citation graph and the LRU order used for eviction. |
| `engine/lexicon.py` | `Lexicon` holds statistics learned once from 600,013 real OpenAlex works: the vocabulary (240,705 words), document frequencies (for idf), the TF-IDF columns and the LSA projection that turns text into 160-number vectors, and the field classifier. It stores no titles or abstracts. |
| `engine/sources.py` | One client per API (`OpenAlex`, `ArXiv`, `EuropePMC`, `SemanticScholar`, `Crossref`) that turns each answer into the common paper format (OpenAlex answers include each paper's reference list). `ScholarlyClient` runs the cascade, caches answers in an LRU with expiry, skips a provider that just failed (for a minute, or as long as its rate limit asks, at most 6 hours), gives each of the next two providers 1.5 s of the deadline, reads the day's OpenAlex budget from the answers' headers and turns on saver mode below 20% (section 16), routes by field to arXiv or Europe PMC, and can keep answers in the optional cache file (2.15), and rate-limits each provider with a token bucket. API keys come from `OPENALEX_API_KEY` and `S2_API_KEY` or from a private `.env` file. |
| `engine/ranker.py` | `GBDTRanker` (the shipped model, 153 trees), `LinearRanker` (baseline) and `HeuristicRanker` (hand-set weights, used if no model is trained). Each gives a raw score and a 0..1 relevance (Platt scaling). Models are plain JSON. |
| `engine/features.py`, `engine/metadata.py` | The nine features. Metadata features are 0..1, or −1 when unknown, so trees can treat "unknown" separately from "bad". |
| `engine/domains.py` | Maps arXiv categories and OpenAlex topics to 13 fields; a logistic-regression classifier predicts the field of text that has no label (queries, Crossref papers). |
| `engine/papers.py`, `engine/text.py` | The common paper record and its identity keys; the text analyzer shared by papers and queries. |
| `dsa/*` | The hand-written syllabus structures used by the engine (see section 5). |
| `ml/*` | Builds citation-based labels (query = a paper's title, relevant = the papers it cites), trains LambdaMART and the linear baseline, calibrates the confidence threshold, evaluates every system on held-out queries. |

## 5. Core data structures in use

| Structure | Where | Used for |
|---|---|---|
| Inverted index stored as flat sorted arrays (CSC form: `indptr`, `docs`, `weights`) | `engine/memindex.py` | BM25 scoring, word lookups for boolean queries and exclusions, coverage features |
| Forward index (per paper: word ids, BM25 weights, title word ids) | `engine/memindex.py` | Building the inverted index without re-tokenising |
| Hash table, hand-written (polynomial hash, separate chaining, doubling rehash), from identity keys (arXiv id, DOI, OpenAlex id, normalised title) to paper ids | `dsa/hashtable.py`, used by `engine/memindex.py` | Merging the same paper arriving from two sources; finding a paper by identifier |
| LRU cache: hash map + doubly linked list | `dsa/lru.py`, used by `engine/memindex.py` and `engine/sources.py` | Evicting the least recently used paper when memory is full; caching API answers with an expiry time |
| Dynamic arrays (numpy buffers that double when full) | `engine/memindex.py` (`_Grow`) | Embedding matrix, field probabilities, metadata columns, alive flags |
| Dense embedding matrix + size-k binary min-heap | `engine/memindex.py`, `engine/pipeline.py` (`top_k`), `dsa/heap.py` | Exact nearest-neighbour search; picking the top 200 BM25 and vector candidates |
| Binary search tree keyed by year | `dsa/bst.py`, used by `engine/memindex.py` (`by_year`) | Year-range filter |
| Sorted vocabulary array + hash map word → id (Python dict) | `engine/lexicon.py` | Prefix queries (`optim*`) and suggestions by binary search; exact word lookups by hashing |
| Array-based stack | `dsa/stack.py`, `dsa/expression.py` | Infix → postfix conversion and postfix evaluation of AND / OR / NOT |
| Sorted lists + two-pointer merge | `dsa/merge.py` | AND, OR, NOT between posting lists |
| Merge sort | `dsa/merge.py` | Ordering result rows by score, year or citations (stable) |
| Binary trees in array form | `engine/ranker.py` | The 153 regression trees of the ranker |
| Sparse matrices (CSR) | `engine/lexicon.py` | TF-IDF of new papers and queries |
| Token bucket | `engine/sources.py` | Rate limits per API |
| Citation graph as adjacency lists (each paper's reference list, plus a hash map from a work to the papers citing it) | `engine/memindex.py` (`citers`, `neighbours`); `ml/dataset.py` (`Pool.refs`) | Related papers; training labels (which papers a seed paper cites) |
| Circular queue + breadth-first search | `dsa/queue.py`, `dsa/graph.py` | Citation links of a paper, nearest first |
| Weighted graph (citation, related-work and similarity edges) + indexed priority queue + Dijkstra | `engine/memindex.py` (`weighted_neighbours`, `related`), `dsa/priority_queue.py`, `dsa/graph.py` | Related papers, closest first by shortest path |
| Radix trie keyed by normalised title | `dsa/trie.py`, used by `engine/memindex.py` (`titles`) | Title suggestions while typing |
| Binary search in each paper's sorted word ids | `engine/pipeline.py` (`_missing_words`) | "Missing: night" under a result |
| Log-structured hash table (append-only file + hash map key → offset) | `dsa/logstore.py`, used by `engine/sources.py` (`TieredCache`) | The optional cache file of API answers |

## 6. How the components interact

**A search answered from memory.** The browser calls `/api/search?q=...`;
`app.py` calls `SearchPipeline.search`, which calls `rank`. Unless the query is
an identifier, a boolean query or marked `exact=1`, `QueryUnderstanding.understand`
first rewrites it (section 12) and the trace records what it changed. `parse_query`
separates words, exclusions and operators. Under the lock, `_rank_local`
builds the filter mask (`_allowed`; a year range is read from the year tree), evaluates a boolean expression if there is
one (`_boolean`: tokens → postfix on a stack → posting-list merges), and asks
`PaperIndex` for candidates: `bm25` reads the posting lists of the query words
and `top_k` keeps the best 200 with a size-200 min-heap; `search_vectors`
multiplies the embedding matrix by the query vector and keeps the nearest 200
the same way. The union is scored:
`feature_matrix` computes the nine features and the ranker turns them into a
score and a relevance. If at least three candidates reach the threshold, the
rows are sorted with merge sort and one page is returned with a trace.

**A search that goes online.** With fewer than three confident candidates,
`rank` releases the lock and calls `ScholarlyClient.search`. For each provider
in order it first checks the cache (an LRU with expiry: a day for results, an
hour for "nothing found"), then the cool-down table, then sends the request
through the provider's token bucket with a timeout and retries. It stops once the
providers asked so far have returned at least 25 distinct papers (half of the
50 asked for) or when the 6-second deadline is reached. The papers are normalised to one
format and passed to `PaperIndex.add`, which merges duplicates through the
identity-key hash table, tokenises and embeds the new papers, appends their rows
to the dynamic arrays, their years to the year tree, their reference lists to the
citation graph and their keys to the LRU (evicting the oldest papers if the
index is full). The next read of the inverted index merges the new postings
in. Then the ranking is computed again over everything in memory, with the new
papers forced into the candidate set. A query written as a sentence was also
sent to OpenAlex's search by meaning, and a short phrase to its title search
(both in parallel with the cascade, on a pool of three threads); their papers
join memory the same way. For a sentence, the ranker's order is then combined
with the by-meaning order by reciprocal rank fusion, and a paper with at least
50 citations whose title is the query goes first.

**An identifier.** A DOI, arXiv id or OpenAlex id is looked up in the identity
map first; if missing, `ScholarlyClient.get_paper` asks OpenAlex, then Semantic
Scholar, then (for DOIs) Crossref, and the paper is added to memory.

**Related papers.** `/api/paper?key=...` returns the paper with up to 8 related
papers. `PaperIndex.related` runs breadth-first search from it over the citation
links between papers in memory, at most 2 links away: first the papers it cites
and the papers citing it, then the papers one link further, each with the paper
it was reached through. The web page shows them under the abstract; clicking one
opens it. Only OpenAlex lists references, so papers that came only from Semantic
Scholar or Crossref have no citation links.

**Suggestions.** `/api/suggest?prefix=prot` runs two binary searches in the
sorted vocabulary and returns the most frequent words in that range.

**Training (offline).** `python -m ml.train` builds the lexicon from real
papers, creates citation-labelled queries, computes the same features the
search computes (`PreparedQuery`, `feature_matrix`), trains LambdaMART and the
linear model, and writes `data/models/`. The search engine only reads those
files.

## 7. Setup and running

Requirements: Python 3.9 or newer, internet access to api.openalex.org,
api.semanticscholar.org and api.crossref.org.

```bash
cd paper-search
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1 (see RUN_ON_WINDOWS.md)
python -m pip install -r requirements.txt
cp .env.example .env                 # then write your free OpenAlex key after OPENALEX_API_KEY=
python app.py                        # open http://127.0.0.1:5000
```

Other commands:

```bash
python scripts/search.py "graph neural networks for molecules"     # command line
python scripts/search.py 10.48550/arXiv.1706.03762                 # one paper by id
python -m unittest                                                  # 75 tests, no network
python scripts/ds_examples.py                                       # the examples in Part 2
python -m ml.train --online                                         # retrain from OpenAlex (needs a key)
python -m ml.evaluate --online --queries 300                        # evaluate on fresh queries
```

Without an OpenAlex key the search still works through Crossref, but OpenAlex
refuses requests once the free budget shared by everyone on the same network is
used up (seen on 19 October 2026). With a free key each OpenAlex request costs
$0.001 of a $1 daily allowance.

## 8. Notable design decisions

* **Online first, nothing stored.** Papers live only in memory for the session.
  This follows the owner's requirement ("everything online, no local data
  storage") and is why the index has a fixed capacity and an eviction policy.
* **Memory before the network.** The network is used only when memory has fewer
  than three confident matches, so related searches in a session are fast and
  cost no API budget.
* **Background statistics.** BM25 idf and the embeddings come from 600,013 real
  works learned once, not from the few dozen papers in memory; otherwise every
  fetched paper contains the query words and those words would look common.
* **Real data only.** Training and evaluation use real OpenAlex papers and
  citation labels; the small corpus in `tests/fixtures.py` only exercises the
  code.
* **Plain-file models.** Rankers and the lexicon are JSON and numpy arrays, so
  they load without pickle and without scikit-learn at run time.
* **Every syllabus unit does a real job.** Version 3.3 implemented only what
  the search strictly needed (one sort, one stack). After the owner's review,
  version 3.4 added a heap, a hash table, a binary search tree and BFS with a
  circular queue, each where it does real work in a search (section 10). They
  are written by hand in Python, which costs some speed against numpy and
  Python's built-ins written in C (section 9).
* **Query understanding without a large language model.** Version 3.5 fixes
  queries that found the wrong papers (sentences, typos, "attention is all you
  need") with a spelling model learned from the same 600,013 papers, a few
  hand-written word lists, and two more OpenAlex searches (by meaning, by
  title). A neural model would add hundreds of megabytes and seconds per query
  on a laptop, and OpenAlex's search by meaning already runs an embedding model
  over 250 million works. Every change is shown and can be undone (section 12).
* **Explainable results.** Every search returns a trace (stack steps, candidate
  counts, providers asked, features of each paper), shown in the web page.

## 9. Known limitations

* **API keys and limits.** OpenAlex needs a free key once the shared budget is
  used up; Semantic Scholar mostly answers "too many requests" without a key;
  Crossref often has no abstracts, which weakens word and meaning features for
  its papers. arXiv gives no citation counts. arXiv and Europe PMC were tested
  with recorded answers only, because the test machine could not reach them.
* **Cache file for one app at a time.** Two copies of the app writing the same
  cache file at once would mix their lines; use one file per running app.
* **Ranker trained offline.** The shipped models were trained on a snapshot pool
  (`--pool`), because the APIs were not reachable when they were built. Live
  online training works, but only about 1 in 7 live citation queries finds a
  cited paper in its top 100, so a useful online training set needs thousands of
  queries over several days of API budget.
* **Session memory only.** Everything fetched is forgotten when the app stops.
* **More requests per new topic.** A long query sends up to four OpenAlex
  requests (keywords, by meaning, fewer words, by title), and opening a paper
  for the first time three more (its record, its references and related works,
  its most cited citing works). With a key each request costs $0.001 of the $1
  daily allowance; repeats come from the cache.
* **Fewer words is a guess.** The words dropped are those least related in
  meaning to the rest of the query, which is sometimes a word that matters
  ("phones" in "make neural networks smaller so they run on phones"). The full
  query is searched too and the fewer-words order counts half in the fusion, so
  this costs little, but it cannot find what both searches miss.
* **Understanding costs time and budget.** A new topic written as a sentence or
  a phrase sends up to two more OpenAlex requests, and the median search with
  fresh memory went from 1.0 s to 1.7 s (search by meaning takes 1.3–2 s). The
  request and recency words are English word lists written by hand, so an
  unusual phrasing can keep a request word or drop a topic word; "Search
  instead for" undoes it. Spelling is corrected only for words fewer than 10 of
  the 600,013 papers use, so a typo that is itself a real word ("form" for
  "from") is left alone.
* **One lock.** All requests share one index under one lock, so searches from
  several users are processed one at a time (the network part runs in parallel).
* **Hand-written structures cost speed.** Measured back to back on 11,866 real
  papers in memory (22 October 2026): a search from memory takes 20.9 ms with
  the version 3.4 structures against 14.6 ms with the numpy and built-in
  versions they replaced. Picking the top 200 of 11,866 scores takes 3.3 ms with
  the Python heap against 0.03 ms with numpy's `argpartition`, and it runs twice
  per search; a key lookup takes about 4 µs in the hand-written hash table
  against 0.05 µs in Python's dict; the year tree answers "2018 to 2020" in
  0.14 ms against 0.03 ms for a numpy comparison of every year. The owner chose
  this cost so the syllabus structures do the real work.
* **Related papers come only from memory.** BFS walks the papers already
  fetched, so a paper opened right after an identifier search may show none
  until its topic has been searched. Asking OpenAlex for reference lists makes
  each search answer larger to download.
* **Linear scans that remain.** The author filter checks every paper's author
  string (O(n) per search with an author), and a prefix with more than 40
  matching words keeps the 40 most common.
* **Merge sort in Python.** It is about 20 times slower than Python's built-in
  sort on 400 rows (0.85–1.6 ms vs 0.04–0.09 ms), a cost accepted to keep the
  syllabus algorithm; it is a few percent of a search.
* **Index compaction pause.** After evictions, stale postings stay in the
  inverted index until a quarter of the slots are dead; then one search pays for
  a full rebuild (about 350 ms at 20,000 papers, roughly once every 100 online
  searches when the index is full).

## 10. Syllabus: what was implemented and why

The rule was to implement a syllabus topic only where the search engine has a
real job for it. Version 3.4 added the heap, the hand-written hash table, the
binary search tree, the circular queue and BFS, so every unit now has at least
one structure doing work in a search. Version 4.0 added the indexed priority
queue with Dijkstra's shortest paths (related papers) and the radix trie (title
suggestions), each for a job the search did badly or not at all before.

| Unit | Used | Where and why | Not used, and why |
|---|---|---|---|
| 1 Arrays, searching, sorting | Arrays and dynamic arrays; sparse matrix (CSC/CSR); time and space complexity; binary search; merge sort; heap (array form) | Every index structure is an array; posting lists are a sparse term × paper matrix; binary search over the sorted vocabulary gives prefix ranges, and in each paper's sorted word ids finds the query words it misses; merge sort orders results stably; a size-k min-heap picks the 200 best candidates and the words a long query can drop | Linear, indexed, Fibonacci search (no unsorted or indexed-sequential data to search); bubble, selection, insertion, quick, radix, shell, bucket sort and heap sort (one stable O(n log n) sort is enough, and heap sort is not stable, see 2.8); address calculation (numpy does it) |
| 2 Linked lists | Doubly linked list; dynamic allocation; vectors | The LRU order (O(1) move-to-end and remove); `_Grow` is a vector that doubles | Singly, circular and generalized lists: no job for them in the search |
| 3 Stacks and queues | Array-based stack; infix → postfix; postfix evaluation; circular queue; priority queue (indexed, with decrease-key) | Boolean queries use two stacks; the BFS for citation links keeps its frontier in a circular queue; Dijkstra for related papers keeps its frontier in the priority queue; the radix trie is walked with a stack | Linked-list stack (no advantage, see 2.6); deque, Josephus: no job in the search |
| 4 Trees | Binary search tree (insert, delete, range search by pruned in-order traversal); binary trees in array representation; non-recursive traversal; trie (radix, with insert, delete and prefix search) | The year tree answers year-range filters; the ranker's 153 regression trees are stored as arrays and walked iteratively; a radix trie (a general tree whose edges carry substrings) suggests titles | Threaded trees, general-tree conversion, AVL or red-black balancing (the year tree has a few hundred nodes at most and is rebuilt balanced at every compaction) |
| 5 Graphs | Adjacency lists; breadth-first search; weighted graph; Dijkstra's shortest paths | Citation links: BFS over the citation graph of the papers in memory; related papers: Dijkstra over citation, related-work and similarity edges weighted by how different two papers are (2.13); the citation graph also gives training labels in `ml/dataset.py` | DFS (its order is not "closest first", see 2.12); spanning trees (no job: the drawer needs the papers nearest one paper, not a tree joining all of them) |
| 6 Hashing | Hash function (polynomial rolling hash); collision resolution by separate chaining; load factor and rehashing; hash index over a file (log-structured hash table) | The identity-key table that merges duplicate papers is written by hand (`dsa/hashtable.py`); the optional cache file is an append-only log with a hash map from each key to its record's offset (`dsa/logstore.py`, 2.15); the vocabulary, the LRU map and the API cache use Python's dict | Open addressing written by hand (separate chaining is simpler to delete from); a hand-written table for the vocabulary (240,705 words: Python's dict is about 80 times faster there and the vocabulary is read for every word of every paper) |

## 11. How the learned ranker works, step by step

This section explains the ranker without the machine learning vocabulary, so
it can be presented at a second-year level.

**1. Nine numbers describe how well a paper fits a query.** For every candidate
paper, `engine/features.py` computes nine features between 0 and 1: word match
(BM25), meaning similarity (embeddings), how much of the title and of the query
the paper covers, whether its field matches the query's field, citations,
recency, peer review and venue quality (−1 when unknown).

**2. One regression tree is a list of yes/no questions.** Each inner node asks
one question about one feature ("is word match ≤ 0.46?"); the answer sends the
paper left or right, until it reaches a leaf that holds a number. Example 10 in
Part 2 walks one paper through a three-question tree.

**3. The score is the sum of 153 trees.** The shipped model has 153 such trees,
each with at most 31 leaves. A paper's score is the sum of the 153 leaf numbers
it reaches; papers are shown in decreasing score. Each tree only adds a small
correction to the trees before it.

**4. How the trees were learned (LambdaMART), in plain steps.**
1. Training data: 3,177 real papers used as queries. The query is the paper's
   title; the papers it cites count as relevant answers.
2. Start with every score at 0.
3. In each round, for every query, look at pairs (a cited paper, a paper that is
   not cited). If the current scores put the pair in the wrong order, the cited
   paper should move up and the other down. The push is stronger when swapping
   the two would change the quality of the top 10 (NDCG@10) a lot, so mistakes
   near the top count most. These pushes are called lambdas.
4. Fit one new tree that predicts each paper's push from its nine features, then
   add the tree's leaf values, multiplied by 0.08 (the learning rate), to the
   scores.
5. Check the quality on 460 separate validation queries after each round, and
   stop when it has not improved for 60 rounds. The best result came after 153
   trees (213 rounds were run).

**5. From score to "relevant".** The score is turned into a 0..1 relevance with
Platt scaling: relevance = 1 / (1 + e^−(0.668 × score − 3.868)). A paper counts
as a confident match at relevance 0.396 or more, and the search goes online when
fewer than 3 papers in memory are confident. This relevance is the probability
that a paper titled like the query would cite the candidate, which is what the
trees were trained on; it orders papers well but is not shown to the user as
"how well it matches" (section 13).

**6. What the trees rely on.** Each feature's share of the improvement made by
the trees' questions (`importance` in `data/models/ranker.json`): citations 0.28, word match
0.24, meaning similarity 0.13, field match 0.12, title coverage 0.08, recency
0.07, query coverage 0.06, venue quality 0.02, peer review 0.

**7. Why trees and not a weighted sum.** On 1,199 held-out test queries the
trees score NDCG@10 0.240, a learned weighted sum of the same nine features
(`data/models/ranker_linear.json`) 0.228, and hand-set weights 0.145. The trees
can learn rules a weighted sum cannot, such as "citations matter only when the
words also match". The owner chose to keep the trees as the default.

In one sentence: the ranker adds up the answers of 153 small decision trees,
each trained to fix the ordering mistakes the trees before it still made.

## 12. Understanding everyday queries (version 3.5)

**The problem.** Searched word for word, "I want research papers on how to make
neural networks smaller" looks for "want", "research" and "make" as if they were
the topic; "nueral netwrok pruning" matches nothing; "recent work on diffusion
models" treats "recent" as a topic word; and the original "Attention Is All You
Need" came 8th, under papers that reuse its title.

**What runs** (`engine/understand.py`, then `engine/pipeline.py`):

1. *Spelling.* A noisy-channel corrector (Norvig, "How to write a spelling
   corrector"). The language model is how many of the 600,013 lexicon papers use
   each word; the error model says one edit (a letter deleted, inserted,
   replaced, or two neighbours swapped) is likelier than two. A word used by
   fewer than 10 papers is replaced by the candidate one edit away used by at
   least 50 times as many papers (and at least 100); a word no paper uses also
   tries two edits if it has 5 to 9 letters. Each candidate is one hash-map
   lookup in the lexicon, O(1): about 54n + 25 candidates for a word of n letters
   at one edit, about 100,000 at two (0.1–0.4 s, so only when needed).
   Acronyms (GNN, LLM) and words under 4 letters are never changed.
2. *Request words.* A run of words at the start such as "can you find me some
   papers on" is dropped when it has at least two words and one that shows a
   request is being made (papers, research, me, please, ...). Pairs such as
   "best papers on" or "latest research about" are dropped anywhere, and a
   trailing "papers". Elsewhere the same words stay, so "attention is all you
   need", "best response dynamics" and "operations research" are untouched.
3. *Recency.* "recent", "latest", "newest", "current" or "new papers" among the
   dropped words becomes a year filter: the last five years (from 2022 in 2026).
4. *Query kind.* A question, a request, or 5 or more topic words (or a
   corrected one) also goes to OpenAlex's search by meaning (`search.semantic`,
   an embedding search); 3 or more keywords also go to its title search, most
   cited first.
5. *Fusion.* For a by-meaning query, the ranker's order and OpenAlex's order are
   combined by reciprocal rank fusion: score = 1/(60 + rank in one list) +
   1/(60 + rank in the other), so a paper high in both lists wins.
6. *Exact title.* A paper with at least 50 citations whose title equals the
   query (or starts with it, for 4 or more words) is moved to the top with a
   stable merge sort; the citation bar keeps an obscure paper that reuses a
   famous title out of first place.

The page shows "Showing results for ...", the corrected words and what was left
out, with a "Search instead for" link that repeats the search with `exact=1`
(command line: `--exact`).

**Measured result.** 26 hand-written queries (sentences, questions, typos, known
titles and plain keywords), each paired with a pattern a right title must
match, run against live OpenAlex with fresh memory on 27 October 2026
(`python scripts/check_queries.py`; `--exact` runs them without understanding):

| | version 3.4 | version 3.5 |
|---|---|---|
| right papers in the top 10 (precision@10) | 0.69 | 0.81 |
| first result is a right paper | 21 of 26 | 23 of 26 |
| the 3 queries with typos | no right paper | 10 of 10 right, each |
| "attention is all you need": the original paper | 8th | 1st |
| median time with fresh memory | 1.0 s | 1.7 s |

Three queries got slightly worse: "how does climate change affect crop yields"
(1.0 → 0.8 of the top 10, first right paper 1st → 2nd), "can machine learning
predict heart disease" (1.0 → 0.9) and "self-driving cars seeing pedestrians at
night" (first right paper 1st → 2nd, though its top 10 improved from 0.6 to
0.8). Live answers change from day to day, so differences of one paper are noise. Known-title queries such as "deep residual learning for image
recognition" still score 0.1 because only one paper has that title; it is first.

## 13. The match shown next to each result (version 3.6)

**The problem.** Up to version 3.5 the bar next to each result showed the
ranker's relevance (section 11, step 5): the probability that a paper with the
query as its title would cite the candidate. It orders papers well, but its
values are small for anything that is not a short title-like phrase, and it
falls further for new papers (few citations). Measured live on the 26 queries
of section 12 with Internet set to "always" (top 10 of each, 27 October 2026),
the bar was below 10% for 21% of the results, and right papers had the same
median bar (9%) as wrong ones. For "I want research papers on how to make
neural networks smaller", SqueezeNet and Slimmable Neural Networks showed 6%
and 4%. Sorting by newest was worse: it sorted every candidate by year, so weak
matches came first, and 60% of the top 10 showed less than 10%.

**What it shows now.** A topic match from the three features a user can check
by reading the paper, with citations, year and venue left out
(`engine/pipeline.py`, `match_scores`):

    match = 0.45 × share of the query words found in the paper
          + 0.20 × share of the query words found in its title
          + 0.35 × meaning similarity (cosine of the 160-number vectors)

It is labelled strong (75% or more), good (50% or more) or partial, with the
percentage. The ranker still decides the order, so a well-cited paper can sit
above one with a slightly higher match.

**Sorting by newest or most cited** now sorts only the papers whose match is
at least 70% of the best match in the list (with merge sort, so ties keep their
relevance order); the other papers follow in relevance order. The side panel
says how many papers were sorted.

**Measured result** (same 26 queries, same day):

| top 10 of each query | relevance bar (3.5) | match (3.6) |
|---|---|---|
| median, sorted by relevance | 38% | 76% |
| below 10%, sorted by relevance | 21% | 0% |
| median, sorted by newest | 1% | 77% |
| below 10%, sorted by newest | 60% | 0% |

The weights were set by hand, not trained. On the rough title patterns of
section 12 the match separates right from wrong titles better than the old bar
(AUC 0.71 against 0.64), but those patterns miss many right papers, so this is
only a sanity check. The lowest query, "self-driving cars seeing pedestrians at
night", averages 35%: its papers cover only some of the words.

## 14. Long queries: fewer words, and missing words (version 4.0)

**The problem.** "How does reinforcement learning improve autonomous vehicle
navigation at night?" did not find "Optimizing Autonomous Vehicle Navigation
Through Reinforcement Learning in Dynamic Urban Environments" (World Electric
Vehicle Journal, 2025, 6 citations), which a web search lists for that question.
OpenAlex has it, but its keyword search wants every word: with "night" and
"improve" it matches 2,376 works and the paper is not among them; without those
two words it matches 35,019 and the paper is 19th. It is not in OpenAlex's
by-meaning top 50 either.

**What runs.** `QueryUnderstanding.relax` (`engine/understand.py`):

1. For a query with at least 5 topic words, each word gets a centrality: the
   cosine between its own direction in the LSA space and the mean direction of
   the other words (`Lexicon.word_vector`; the directions were learned from the
   600,013 lexicon papers). For this query: autonomous 0.66, reinforcement 0.40,
   vehicle 0.40, navigation 0.38, learning 0.20, night 0.08, improve −0.02.
2. Words below 0.1 may be dropped, the least related first, at most a third of
   the words; the k least related are picked with the size-k heap of
   `dsa/heap.py`. Here: "improve" and "night".
3. The fewer-words query ("reinforcement learning autonomous vehicle
   navigation") is one more OpenAlex keyword search, run in parallel with the
   others, and its order is fused in at weight 0.5 (the by-meaning order counts
   1). Weight 1 lifted this paper further but cut "make neural networks smaller
   so they run on phones" from 7 to 2 right titles in the top 10, because its
   fewer-words version drops "phones"; 0.5 left the other 26 queries as they
   were.

Every result also says which query words it does not contain ("Missing:
improve night"), so a paper that skips a word reads as a partial match rather
than a wrong one. A paper's word ids are kept sorted in the forward index, so
each word is one binary search (`np.searchsorted`), O(log n).

**Measured result** (live, fresh memory per query, 27 October 2026):

| | before (3.6) | now (4.0) |
|---|---|---|
| that paper, sorted by relevance | not found | 38th of 135 (14th by newest) |
| "reinforcement learning for autonomous vehicle navigation in urban environments" | | 2nd |
| 27 check queries: right titles in the top 10 | 74% (`--exact`) | 81% |
| 27 check queries: first result right | 24 (`--exact`) | 26 |
| median time per new query | 1.0 s (`--exact`) | 1.8 s |

The paper stays outside the first page for the question as asked because about
35 papers match more of it (several are about driving at night) or are far more
cited; sorted by most cited, a 2025 paper with 6 citations comes late whatever
the query.

## 15. Related papers and title suggestions (version 4.0)

**Related papers.** Until 3.6 the drawer listed papers joined to the opened one
by citations, found by BFS among the papers in memory; a paper whose references
had not been fetched had none. Now:

1. The first time a paper is opened, `ScholarlyClient.neighbourhood` fetches the
   papers around it from OpenAlex: its record (references and OpenAlex's
   "related works"), those works (one batch of at most 50), and its 25 most
   cited citing works. Three requests, cached for a week.
2. `PaperIndex.weighted_neighbours` gives each paper's edges: the papers it
   cites, that cite it, that OpenAlex lists as related, and its 5 nearest by
   meaning. An edge's length is how different the two papers are, 1 − cosine of
   their vectors (at least 0.02), halved for a citation and multiplied by 0.6
   for a related work.
3. `dsa/graph.py`'s `dijkstra` lists the 8 papers with the shortest paths from
   the opened one, using the indexed priority queue of `dsa/priority_queue.py`
   (2.13). The drawer shows each one's closeness (e^−distance), the link that
   reached it, and the paper it was reached through.

The citation links (BFS, 2.12) are still shown under them. Measured on 1,206
papers in memory: 3.8 ms median per opened paper (6.4 ms at most), after the
first opening's fetch (about 1.5 s).

**Title suggestions.** While you type, titles of papers in memory that start
with the text are suggested under the word suggestions, most cited first, and
clicking one opens the paper. The titles live in a radix trie (2.14), updated
when papers are added, evicted or renumbered. On the same 1,206 papers (82,767
title characters) the trie has 1,865 nodes, and a suggestion takes 0.10 ms
median.

## 16. Saving the API budget (version 4.1)

A free OpenAlex key gives $1 a day. Each answer says what it cost and what is
left, in its headers: `X-RateLimit-Cost-USD` ($0.001 for a search, $0.0001 for
a lookup by id or a citation filter), `X-RateLimit-Remaining-USD`,
`X-RateLimit-Limit-USD` and `X-RateLimit-Reset` (seconds until the budget is
full again). A new question starts up to four searches: keywords, by meaning
(sentences), with fewer words (long queries) and by title (title-like
queries). Version 4.1 adds four free ways to spend less:

1. **Budget meter.** `Http.fetch` hands every answer's headers to
   `Budget.observe` (engine/sources.py). The page header and `/api/state` show
   the remaining budget, when it is full again, and what this app has spent
   since it started.
2. **Saver mode.** `ScholarlyClient.saver()` turns it on when less than 20% of
   the day's budget is left, when OpenAlex has said the budget is used up, or
   when `PAPERSCOPE_SAVER=on`. Then `plan()` asks the free source for the
   query's field first, then Semantic Scholar, then OpenAlex, then Crossref, and
   the pipeline skips the three extra OpenAlex searches. The side panel says so.
3. **Free sources by field.** `ArXiv` (Atom XML, no key, one request every 3
   seconds as arXiv asks) for computer science, physics, mathematics and
   statistics; `EuropePMC` (JSON, no key) for biology and medicine. The field is
   the one chosen in the form, or the field classifier's guess for the query
   text (engine/domains.py). Outside saver mode the free source comes right
   after OpenAlex, so it is asked when OpenAlex fails or finds fewer than 25
   papers. arXiv gives no citation counts, so its papers rank on words, meaning,
   field and recency.
4. **Cache file, opt-in.** `PAPERSCOPE_CACHE_FILE` in `.env` adds a second
   level under the in-memory LRU (`TieredCache`): answers are also written to a
   log-structured hash table on disk (2.15) and kept for 7 days. A miss in
   memory looks in the file and copies the answer back into memory. "Not found"
   answers are not written. With the variable empty (the default) nothing is
   written to disk, as before.

Requests are not sent through several OpenAlex accounts to get more budget:
that is against OpenAlex's terms and can get the key blocked.

**Measured** on the 27 live check queries, a fresh memory for each:

| Mode | OpenAlex cost | Right titles in the top 10 | First result right | Median time |
|---|---|---|---|---|
| Normal | $0.072 (72 searches) | 81% | 26 of 27 | 1.8 s |
| Saver | $0.023 | 81% | 24 of 27 | 2.8 s |

The normal row's quality and time are from the 4.0 run (that path did not
change); its cost counts the searches each query starts. Saver mode lost the
first place on "make neural networks smaller so they run on phones" (2nd),
"self-driving cars seeing pedestrians at night" (6th) and "translating
languages without parallel data" (2nd), the queries the search by meaning had
helped most. The test machine could not reach arXiv or Europe PMC (its network
allows only some hosts) and Semantic Scholar answered "too many requests", so
those attempts only added time, and every saver search ended at one OpenAlex
keyword search. On a normal connection the free sources answer first.

The cache file, on 500 answers of 50 papers each (37.5 MB): writing an answer
0.29 ms, reading one 0.20 ms, rebuilding the hash map when the app starts
110 ms. An OpenAlex search takes about 1 s.

---

# Part 2: Data structure justification and evaluation

## How the structures were evaluated

The operations that run most often are, per search: reading posting lists for
BM25, the vector search, picking the top candidates, computing features,
scoring candidates with the ranker, and sorting the result rows. Per online
search (each new topic) the engine also adds about 50 papers: tokenising them,
looking up every word in the vocabulary, merging duplicates by identity key,
updating the inverted index and the LRU order.

Each structure was compared with at least two alternatives in two ways:

1. **A small worked example**, the same input run through the chosen structure
   and the alternatives, with the operations counted. The blocks below are the
   output of `python scripts/ds_examples.py` (inputs are a few made-up words and
   ids, chosen to be easy to follow by hand).
2. **A measurement at full size**: 20,000 real OpenAlex works (fetched with the
   project's own client for this measurement only and kept outside the project),
   the default memory limit. The index then holds 1,639,034 postings and 59,450
   identity keys. Times are medians of repeated runs. The structures added in
   version 3.4 were measured the same way on 12,000 real works (11,866 after
   duplicates were merged, 35,653 identity keys) on 22 October 2026, against
   the version 3.3 code run back to back on the same machine.

Notation: n papers in memory, P postings in the inverted index, p postings of
the new papers, V words in the vocabulary (240,705), k the number of results kept.

## Summary of the outcome

| Structure | Verdict | Effect at 20,000 papers |
|---|---|---|
| Inverted index update | **Changed**: full re-sort → linear append merge | Adding 50 papers: 392–465 ms → 75–92 ms |
| Vocabulary lookup | **Changed**: binary search only → hash map for exact lookups (sorted array kept for prefixes) | 36,622 lookups: 42–54 ms → 11 ms, costs 14.4 MB |
| Ranker tree walk | **Changed**: level-by-level matrix → shrinking array of active pairs | Scoring 400 candidates: 31–49 ms → 11–17 ms |
| Identity-key hash table | **Changed in 3.4** (syllabus): Python dict → hand-written separate chaining | About 4 µs per lookup vs 0.05 µs (dict); about 1 ms per online search |
| LRU (hash map + doubly linked list) | Kept | 0.7–1.1 µs per operation; under 0.1 ms per search |
| Array stack | Kept | Queries have a few dozen tokens; alternatives give no gain |
| Sorted lists + merge | Kept | O(m + n), output already sorted |
| Merge sort | Kept (syllabus constraint, and stability is needed) | 0.85–1.6 ms per 400 rows |
| Dynamic arrays | Kept | 2,000 appends: 2.8–3.1 ms vs 128–139 ms with exact-size copies |
| Top-k candidate selection | **Changed in 3.4** (syllabus): numpy partial selection → size-k min-heap | Top 200 of 11,866: 0.03 ms → 3.3 ms, twice per search |
| Year-range filter | **Added in 3.4**: binary search tree keyed by year | 0.14 ms per range vs 0.03 ms for a numpy scan of every year |
| Related papers | **Added in 3.4**: citation graph (adjacency lists) + BFS with a circular queue | New feature: 0.14 ms median per paper opened (377 papers, 1,211 links) |

The three changes of version 3.3 keep the external behaviour identical. The
version 3.4 changes keep it identical except where two candidates tie exactly
at the 200th place: the heap then keeps the smaller paper id, where numpy kept
an arbitrary one. Section "Verification" shows how both were checked.

Overall, at 20,000 papers: a search answered from memory went from 62–70 ms to
40–42 ms, the CPU part of a search that goes online (adding 50 papers and
updating the index) went from 392–465 ms to 75–92 ms, and filling memory with
20,000 papers in batches of 50 went from 84–87 s to 27–28 s.

A later fix (v3.3, not a data structure change) stopped `_rank_local` from
computing the ranker score twice: `relevance` now reuses the score already
computed. In one back-to-back run on the same machine, a search from memory took
34.6 ms before and 30.0 ms after, with identical results.

Version 3.4 traded some of that speed back for the syllabus structures: on
11,866 papers a search from memory took 14.6 ms with the version 3.3 code and
20.9 ms with version 3.4, almost all of the difference being the Python heap;
adding 50 papers went from 39.7 ms to 43.1 ms (hash table, year tree and
citation graph together).

## 2.1 Inverted index (changed)

**What it is.** For every word, the sorted list of papers that contain it, with
each paper's BM25 weight for that word. All lists are stored back to back in two
flat arrays (`docs`, `weights`), and `indptr[t]` says where word t's list
starts: the papers containing word t are `docs[indptr[t]:indptr[t+1]]`. This is
the compressed sparse column (CSC) form of the sparse term × paper matrix.

**Why it was chosen.** A query reads only its own words' lists. BM25 for every
paper is one vectorised pass per query word (`acc[docs[lo:hi]] += w * ws[lo:hi]`),
so the text of the papers is never scanned. Only non-zero cells are stored.

**What changed.** The arrays used to be rebuilt from scratch after every batch
of new papers: concatenate all postings and sort them again by (word, paper),
O(P log P). But new papers always get larger ids than every paper already
indexed, so each word's new postings belong at the end of its list. The new
`PaperIndex._append_lists` computes, for every posting, its position in the
merged arrays with two `np.repeat` calls and writes it there: O(P + p), no
comparisons. Evicted papers' postings stay in the arrays and are skipped when
read (`postings`, `bm25`, `_df`) until compaction renumbers the papers and
rebuilds once.

```
=== Example 1: inverted index, adding one paper
before   indptr [0, 2, 3, 5, 6] docs [0, 1, 1, 0, 2, 2]
         graph -> docs[0:2] = [0, 1]
chosen   append merge: new postings [3, 3] for words ['graph', 'quantum']
         after  indptr [0, 3, 4, 6, 8] docs [0, 1, 3, 1, 0, 2, 2, 3]
         operations: 8 array writes, no comparisons (paper 3 > every old id)
alt. 1   full rebuild: sort all 8 (word, paper) pairs again, about n log2 n = 24 comparisons; same result: True
alt. 2   hash map of Python lists: {'graph': [0, 1, 3], 'image': [1], 'network': [0, 2], 'quantum': [2, 3]}
         append is 1 operation per new posting, but every id is a separate Python object
```

(Words are numbered graph 0, image 1, network 2, quantum 3; papers 0–2 contain
"graph network", "graph image" and "network quantum"; paper 3 contains "graph quantum".)

| | Add p postings | Read one word's list | BM25 over q words | Space |
|---|---|---|---|---|
| **Chosen: CSC arrays, append merge** | O(P + p) | O(1) to find, then a slice | vectorised, O(postings read) | 12 bytes per posting |
| Alt. 1: CSC arrays, full re-sort (old code) | O((P + p) log(P + p)) | same | same | same |
| Alt. 2: hash map word → Python lists | O(p) | O(1) | Python loop per posting | about 80 bytes per posting |

Measured at 20,000 papers (P = 1,639,034): adding 50 papers and updating the
index took **392–465 ms with the full re-sort and 75–92 ms with the append
merge** (re-sort alone: about 346 ms). The hash map of lists would make appends
cheap, but it needs 131 MB instead of 21.7 MB, and BM25 over its lists took
1.72 ms per query instead of 0.09 ms because every posting is visited by Python
code. The append merge keeps the fast reads and removes the re-sort.

## 2.2 Vocabulary lookup (changed)

**What it is.** The lexicon's 240,705 words in a sorted array, plus (new) a hash
map from word to its position.

**Why both.** Two operations need the vocabulary. Exact lookup ("which id is
'network'?") runs for every word of every paper added, about 120 lookups per
paper. Prefix lookup (`optim*` in a query, suggestions while typing) needs every
word starting with a prefix; in a sorted array these form one contiguous range,
found with two binary searches. The old code used binary search for both, so
each exact lookup cost about 18 string comparisons (log2 240,705). A hash map
answers it in O(1) on average.

```
=== Example 7: looking up words in the vocabulary
vocab     ['graph', 'image', 'net', 'network', 'neural', 'optimal', 'optimize', 'quantum']
chosen   exact 'network': hash map -> id 3 (1 hash, 1 comparison)
         prefix 'optim*': two binary searches in the sorted array -> ids 5..6 ['optimal', 'optimize']
alt. 1   binary search only: exact 'network' takes 3 string comparisons (log2 V; 18 for 240,705 words)
alt. 2   trie: 'network' walks 7 nodes, prefix 'optim*' walks 5 then lists the subtree; 39 nodes for 8 words
```

| | Exact lookup | Prefix range | Extra space |
|---|---|---|---|
| **Chosen: sorted array + hash map** | O(1) average | O(log V) | 14.4 MB for the map |
| Alt. 1: sorted array only (old code) | O(log V) | O(log V) | none |
| Alt. 2: trie | O(word length) | O(prefix length) + subtree | 126.9 MB as Python dicts |
| Alt. 3: hash map only | O(1) | O(V) scan (no order) | 14.4 MB |

Measured on 36,622 real word lookups: **42–54 ms with binary search, 11 ms with
the hash map**. In the end-to-end test this saved about 11 ms per batch of 50
added papers. The trie would serve prefixes well, but in Python it needs about
nine times the memory of the map, and the sorted array already gives prefix
ranges in O(log V).

## 2.3 Ranker trees (changed)

**What it is.** The 153 regression trees of the shipped model are binary trees
in array form: for node i, `feature[i]`, `threshold[i]`, `left[i]`, `right[i]`
(−1 for a leaf) and `value[i]`. All trees are packed into the same arrays (9,333
nodes). A candidate's score is the sum of the leaf values it reaches.

**What changed.** The old walk kept a dense matrix with the current node of
every (candidate, tree) pair and updated the whole matrix once per tree level,
as many times as the deepest tree is deep (19), even for pairs already at a leaf.
The average path is only 4.5 nodes long. The new walk keeps a shrinking array of
the pairs that are not at a leaf yet (`active`) and drops each pair as soon as it
reaches its leaf, so each pair costs one step per node on its own path.

```
=== Example 10: scoring one candidate with one regression tree
chosen   arrays feature/threshold/left/right/value; x = [0.8, 0.75]: path [0, 2, 4] -> 0.9
         (all candidates and trees walk together; a pair leaves the active array at its leaf)
alt. 1   node objects with left/right pointers: same path, one Python object per node,
         walked one candidate at a time in Python
alt. 2   dense level-by-level matrix of positions: every (candidate, tree) pair is updated at every
         level of the deepest tree (19 for the shipped model) even after reaching its leaf
```

(The tree in the example: node 0 asks "bm25 ≤ 0.5?"; yes leads to leaf 1
(−0.3); no leads to node 2, "semantic ≤ 0.7?", whose leaves are 3 (0.2) and 4 (0.9).)

| | Work to score c candidates with T trees | Space |
|---|---|---|
| **Chosen: arrays + active pairs** | O(c · total path length), about c · T · 4.5 steps, vectorised | arrays of the nodes + c · T positions |
| Alt. 1: node objects (linked representation) | same steps, but one Python step each | one object per node |
| Alt. 2: arrays + dense level-by-level matrix (old code) | O(c · T · max depth) = c · T · 19 steps, vectorised | same as chosen |

Measured on 400 candidates: **31–49 ms with the dense matrix, 11–17 ms with
the active pairs**; node objects walked in Python took 22–32 ms. At the time of
this change the ranker ran twice per search (score, then relevance), so this was
the main reason a search from memory went from 62–70 ms to 40–42 ms. Since v3.3
it runs once.

## 2.4 Identity-key hash table (changed in 3.4)

**What it is.** `PaperIndex.key_to_doc`, a hash table from keys such as
`doi:10.1038/nature14539`, `arxiv:1706.03762`, `openalex:w2741809807` and a
normalised title to the paper's id. Since version 3.4 it is the hand-written
`dsa/hashtable.py` instead of a Python dict:

* hash function: polynomial rolling hash of the key's characters,
  h = h × 31 + code(character), kept to 64 bits; bucket = h mod number of buckets;
* collisions: separate chaining, each bucket a short list of (key, id) pairs;
* overflow: when entries / buckets passes 0.75, the number of buckets doubles
  and every entry is re-hashed.

**Why a hash table.** Every paper that comes back from an API is checked against
it (about 3 keys per paper, 50 papers per online search) to merge the same paper
arriving from two sources, and every identifier search and result page uses it.

```
=== Example 2: finding a paper by an identity key (hand-written hash table)
chosen   put arxiv:1706.03762         hash mod 4 = bucket 0
chosen   put doi:10.1038/nature14539  hash mod 4 = bucket 1
chosen   put openalex:w2741809807     hash mod 4 = bucket 1
chosen   put doi:10.1145/3292500      hash mod 4 = bucket 2  -> load 4/4 > 0.75: rehash into 8 buckets
         buckets now: [[], ['doi:10.1038/nature14539'], ['doi:10.1145/3292500'], [], ['arxiv:1706.03762'], ['openalex:w2741809807'], [], []]
         get doi:10.1145/3292500: bucket 2, compare 1 key(s) -> 3
alt. 1   list scan: compare keys one by one -> 4 comparisons (n in the worst case)
alt. 2   sorted list + binary search: 2 comparisons (log2 n); but inserting a new key shifts 3 keys to the right
```

| | Lookup | Insert | Delete | Space |
|---|---|---|---|---|
| **Chosen: hash table, separate chaining** | O(1) average (chains stay short at load ≤ 0.75) | O(1) amortised (doubling rehash) | O(1) average | buckets + one pair per key |
| Alt. 1: list of (key, id) pairs | O(n) | O(1) | O(n) | smallest |
| Alt. 2: sorted list + binary search | O(log n) | O(n) (shift) | O(n) (shift) | small |
| Alt. 3: Python dict (version 3.3) | O(1) average, in C | O(1) amortised | O(1) | open addressing |

Measured: with 59,450 keys (version 3.3) a dict lookup took 0.06 µs, binary
search 0.6–0.8 µs and a list scan 1.6 ms. With the 35,653 keys of the version
3.4 measurement, the hand-written table had load factor 0.54 and its longest
chain held 6 keys; a lookup took about 4 µs against 0.05 µs for a dict. Python's
dict uses the same idea in C, so it is about 80 times faster. The hand-written
table was chosen anyway, at the owner's request, so that hashing, collision
handling and rehashing are the project's own code; at a few hundred lookups per
online search its cost is about 1 ms. The vocabulary (240,705 words, read for
every word of every paper) keeps Python's dict, where the factor of 80 would
show.

## 2.5 LRU cache: hash map + doubly linked list (kept)

**What it is.** `dsa/lru.py`. The hash map finds a key's node; the doubly linked
list orders nodes from least to most recently used, with sentinel head and tail
nodes. Used twice: for the papers in memory (evict the least recently used one
when 20,000 are held) and for API answers (with an expiry time).

```
=== Example 3: keeping the most recently used papers (capacity 3)
chosen   put A: list oldest->newest ['A']
chosen   put B: list oldest->newest ['A', 'B']
chosen   put C: list oldest->newest ['A', 'B', 'C']
chosen   get A: list oldest->newest ['B', 'C', 'A']
chosen   put D: list oldest->newest ['C', 'A', 'D']  evicted B
         every step: 1 hash lookup + at most 4 pointer changes, O(1)
alt. 1   array in recency order: same final order ['C', 'A', 'D'], but get/evict shift elements (5 moves here; O(n) per operation)
alt. 2   min-heap of last-use times: evicts B as well, O(log n) per operation, but before the eviction it held 5 entries for 4 papers (stale entries stay until popped)
```

| | get / touch | put + evict | Space |
|---|---|---|---|
| **Chosen: hash map + doubly linked list** | O(1) | O(1) | map + one node per entry |
| Alt. 1: array kept in recency order | O(n) (find, remove, append) | O(n) (remove the front) | one array |
| Alt. 2: min-heap of last-use times (lazy deletion) | O(log n) | O(log n) amortised | grows with every touch until stale entries are popped |

Measured with 20,000 entries: 0.7–1.1 µs per operation for `dsa/lru.py`; about
100 µs per touch for the array. The heap with lazy deletion ran the benchmark in
11–13 ms against 19–28 ms, but only because Python's `heapq` is written in C:
it does more work per operation (O(log n)) and its memory grows with every touch.
Python's `OrderedDict` (the same hash map + doubly linked list, in C) ran it in
3.5–6.3 ms. A search does about 60 LRU operations (50 papers added, 10 shown), so
even the fastest option would save under 0.1 ms per search. The hand-written
structure stays: it is the syllabus doubly linked list doing a real job, at a
cost too small to measure in a search.

## 2.6 Array-based stack (kept)

**What it is.** `dsa/stack.py`: push and pop at the end of one Python list
(a dynamic array). Two stacks are used for boolean queries: one for operators
in the shunting-yard conversion from infix to postfix, one for operand lists
when evaluating the postfix expression.

```
=== Example 4: boolean query on a stack: deep AND (vision OR image) NOT survey
         token deep    stack []
         token AND     stack ['AND']
         token (       stack ['AND', '(']
         token vision  stack ['AND', '(']
         token OR      stack ['AND', '(', 'OR']
         token image   stack ['AND', '(', 'OR']
         token )       stack ['AND']
         token AND     stack ['AND']
         token NOT     stack ['AND', 'NOT1']
         token survey  stack ['AND', 'NOT1']
         token end     stack []
chosen   postfix: deep vision image OR AND survey NOT1 AND (10 tokens, each pushed and popped at most once: O(n))
         array stack (dsa/stack.py): push/pop at the end of one array, no extra object per item
alt. 1   linked-list stack: same O(1) push/pop, plus one node object (value + next pointer) per push
alt. 2   recursive-descent parser: same O(n), but the stack is Python's call stack, so the
         intermediate stack states (shown in the search trace) cannot be recorded
```

(The second `AND` is the implicit one the tokenizer inserts: "x NOT y" means
"x AND NOT y". `NOT1` is NOT with one operand.)

| | push / pop | Space | Notes |
|---|---|---|---|
| **Chosen: array stack** | O(1) amortised | one array | items next to each other in memory |
| Alt. 1: linked-list stack | O(1) | one node object per item | no gain for a few dozen tokens |
| Alt. 2: recursion (call stack) | O(1) per call | call frames | no explicit stack to show in the trace; recursion depth limit |

The current choice is sound: the stack never holds more than a few dozen tokens,
and the explicit stack is what lets the web page show each step.

## 2.7 Sorted posting lists and the merge (kept)

**What it is.** `dsa/merge.py`: `intersect` (AND), `union` (OR) and
`difference` (AND NOT) walk two sorted lists with one pointer each.

```
=== Example 5: AND of two sorted posting lists
chosen   merge pass: [4, 9], 5 comparisons (at most m + n), result already sorted
alt. 1   nested loops: 16 comparisons (m * n), result [4, 9]
alt. 2   hash set: 8 hash operations, result [4, 9] only after sorting again (a set has no order), plus a set object per list
```

| | AND / OR / NOT of lists of length m and n | Extra space | Output |
|---|---|---|---|
| **Chosen: two-pointer merge** | O(m + n) | the output only | sorted, ready for the next operator |
| Alt. 1: nested loops | O(m · n) | the output only | sorted |
| Alt. 2: hash sets | O(m + n) expected, + O(k log k) to sort the result | a set per operand | unordered |

The posting lists are already sorted (section 2.1), so the merge needs no extra
work and its output is sorted for the next operator on the stack. The current
choice is sound.

## 2.8 Merge sort (kept)

**What it is.** `dsa/merge.py`, bottom-up merge sort (no recursion). It orders the
candidate rows by score in `_rank_local`, and again by year or citations when the
user asks.

```
=== Example 6: sorting results by year, newest first (rows already in relevance order)
input     [('A', 2019), ('B', 2021), ('C', 2019), ('D', 2021)]
chosen   merge sort: [('B', 2021), ('D', 2021), ('A', 2019), ('C', 2019)] (B before D and A before C, as in the relevance order: stable)
alt. 1   quicksort: [('B', 2021), ('D', 2021), ('C', 2019), ('A', 2019)] (equal years swapped: not stable; O(n^2) worst case)
alt. 2   heap sort: [('B', 2021), ('D', 2021), ('C', 2019), ('A', 2019)] (equal years reordered by the heap: not stable)
```

| | Time | Extra space | Stable |
|---|---|---|---|
| **Chosen: merge sort** | O(n log n) always | O(n) | yes |
| Alt. 1: quicksort | O(n log n) average, O(n²) worst | O(log n) | no |
| Alt. 2: heap sort | O(n log n) always | O(1) | no |

Stability matters here: when results are sorted by year, papers from the same
year must stay in relevance order, which quicksort and heap sort do not
guarantee (example above: A was more relevant than C, and both alternatives put
C first). Merge sort is also the one sort the project owner chose to keep. On
400 rows it takes 0.85–1.6 ms (Python's built-in sort, also a stable
merge-based sort, takes 0.04–0.09 ms, see Known limitations).

## 2.9 Dynamic arrays (kept)

**What it is.** `_Grow` in `engine/memindex.py`: a numpy buffer of fixed-width
rows that doubles its capacity when full. It holds the 160-number embedding of
each paper, its 13 field probabilities, its 4 metadata columns and its alive flag.

```
=== Example 8: appending 20 vector rows one at a time
chosen   doubling array: capacity 16 -> [32], 16 rows copied in total (amortised O(1) each)
alt. 1   exact-size array, new copy on every append: 190 rows copied (O(n) each, O(n^2) total)
alt. 2   Python list of row lists: appends are cheap, but there is no single matrix to multiply,
         so 'vectors @ query' becomes a Python loop over every paper
```

| | Append | Read all rows as one matrix | Space |
|---|---|---|---|
| **Chosen: doubling numpy buffer** | O(1) amortised | O(1) (a view) | up to 2× the rows held |
| Alt. 1: exact-size array copied on each append | O(n) | O(1) | exact |
| Alt. 2: Python list of rows | O(1) amortised | O(n) conversion before every vector search | a Python object per number |

Measured: 2,000 single-row appends of 160 numbers took 2.8–3.1 ms with
doubling and 128–139 ms with exact-size copies. The current choice is sound.

## 2.10 Top-k candidate selection with a heap (changed in 3.4)

**What it is.** The embeddings form one dense matrix; a vector search is one
matrix-vector product (cosine similarity, because vectors have unit length).
Then the 200 best scores are picked, both for the vector search
(`PaperIndex.search_vectors`) and for BM25 (`top_k` in `engine/pipeline.py`).
Since version 3.4 this uses `dsa/heap.py`: a binary min-heap in array form (the
children of position i are 2i+1 and 2i+2) that holds the best k scores seen so
far, with the weakest at the root. Each new score is compared with the root; if
it is better it replaces the root and sinks to its place. Equal scores keep the
smaller paper id, so the result does not depend on the order of the scores.

```
=== Example 9: the 3 best of 8 scores (size-k min-heap)
scores    [0.2, 0.9, 0.1, 0.7, 0.4, 0.8, 0.3, 0.6]
chosen   paper 0 (0.2): heap not full, push   heap [0.2]
         paper 1 (0.9): heap not full, push   heap [0.2, 0.9]
         paper 2 (0.1): heap not full, push   heap [0.1, 0.9, 0.2]
         paper 3 (0.7): beats the root, replace heap [0.2, 0.9, 0.7]
         paper 4 (0.4): beats the root, replace heap [0.4, 0.9, 0.7]
         paper 5 (0.8): beats the root, replace heap [0.7, 0.9, 0.8]
         paper 6 (0.3): not above root 0.7, dropped
         paper 7 (0.6): not above root 0.7, dropped
         best 3: papers [1, 3, 5]; 5 root comparisons, 3 sift-downs of at most log2 k levels: O(n log k), O(k) memory
alt. 1   partial selection (numpy argpartition): [1, 3, 5] O(n) on average, but needs all n scores in one array
alt. 2   full sort (argsort): [1, 5, 3] O(n log n), orders all 8 to keep 3
```

| | Time for top k of n | Extra space | Result |
|---|---|---|---|
| **Chosen: size-k min-heap** | O(n log k) | O(k) | the k best; ties: smaller id |
| Alt. 1: partial selection (`np.argpartition`, version 3.3) | O(n) average, in C | O(n) | the k best; ties: arbitrary |
| Alt. 2: full sort | O(n log n) | O(n) | all n in order |

Measured, top 200 of 11,866 scores: 0.03 ms with `argpartition` and 3.3 ms with
the Python heap (0.63–0.66 ms for a full sort of 20,000 in version 3.3). The heap
does fewer comparisons than a sort, but every comparison is a Python step while
`argpartition` runs in C. It runs twice per search, so it is the main reason a
search from memory went from 14.6 ms to 20.9 ms. It was chosen at the owner's
request, so the heap is the project's own code doing a real job. An approximate
nearest-neighbour index (HNSW, IVF, KD-tree) was also considered: at 20,000 ×
160 numbers (12 MB) the exact product is already about a millisecond, so it
would add complexity and lose some results for no gain.

## 2.11 Year-range filter with a binary search tree (added in 3.4)

**What it is.** `dsa/bst.py`, `YearTree`: one node per publication year holding
the set of papers of that year; earlier years in the left subtree, later years in
the right. `PaperIndex` inserts each new paper, removes evicted ones, and builds
a balanced tree (median year at the root) at every compaction. A filter "from
2018 to 2020" is an in-order traversal that skips every subtree that cannot hold
years in the range (`range(lo, hi)`), written with an explicit stack instead of
recursion.

```
=== Example 11: papers from 2018 to 2020 (binary search tree keyed by year)
insert    [('A', 2019), ('B', 2015), ('C', 2021), ('D', 2017), ('E', 2020), ('F', 2023), ('G', 2019), ('H', 2012)]
chosen   tree of 7 years, height 3; range(2018, 2020) visits years [2015, 2017, 2019, 2020, 2021] -> ['A', 'G', 'E']
         (5 of 7 nodes: subtrees outside the range are skipped; O(h + years visited + results))
alt. 1   check every paper's year: 8 comparisons (O(n)) -> ['A', 'E', 'G']
alt. 2   sorted array + binary search: rows 3..5 -> ['A', 'G', 'E'] in O(log n), but adding a 2016 paper shifts 6 rows
```

| | Range query | Add a paper | Remove a paper | Space |
|---|---|---|---|---|
| **Chosen: BST keyed by year** | O(h + years visited + results) | O(h) | O(h) | one node per year + the ids |
| Alt. 1: check every paper's year (version 3.3, vectorised with numpy) | O(n) | O(1) | O(1) | one number per paper |
| Alt. 2: sorted array of (year, id) + binary search | O(log n + results) | O(n) (shift) | O(n) (shift) | one pair per paper |

h is the tree's height: there are only a few dozen to a few hundred distinct
years, so h stays small (6 for the 12 years of the measurement; the balanced
rebuild keeps it near log2 of the number of years). Measured on 11,866 papers:
0.14 ms per range query with the tree, 0.03 ms for numpy comparing every year at
C speed. The tree does far fewer comparisons (it visits years, not papers) but in
Python; it was added so a tree does the filter, at a cost of about 0.1 ms per
filtered search. The sorted array would answer ranges as fast but pays a shift
for every paper added.

## 2.12 Citation graph and breadth-first search (added in 3.4)

**What it is.** OpenAlex returns each paper's reference list. `PaperIndex`
keeps it with the paper (the paper's outgoing edges) and records each reference
in a hash map from a work number to the papers in memory that cite it (incoming
edges). Together they form the citation graph as adjacency lists.
`PaperIndex.related` runs `dsa/graph.py`'s breadth-first search from one paper,
at most 2 links away and at most 8 papers. The frontier is a circular queue
(`dsa/queue.py`: an array reused in a circle, enqueue at the rear and dequeue at
the front in O(1)), and a set of visited papers makes sure each paper is listed
once.

```
=== Example 12: related papers by citation links (BFS with a circular queue)
cites     {'P': ['A', 'B'], 'A': ['C'], 'B': ['C', 'D'], 'C': ['E'], 'D': [], 'E': [], 'F': ['P']}
chosen   dequeue P (hop 0): queue ['A', 'B', 'F']
         dequeue A (hop 1): queue ['B', 'F', 'C']
         dequeue B (hop 1): queue ['F', 'C', 'D']
         dequeue F (hop 1): queue ['C', 'D']
         dequeue C (hop 2): queue ['D']
         dequeue D (hop 2): queue []
         BFS order (paper, hops): [('A', 1), ('B', 1), ('F', 1), ('C', 2), ('D', 2)] - every 1-hop paper before any 2-hop one
alt. 1   depth-first search (stack) order, with path length: [('A', 1), ('C', 2), ('E', 3), ('B', 3), ('D', 4), ('F', 1)]
         it follows one chain as deep as it goes: B, which P cites, is reached by a 3-link path, and F, which cites P, comes last
alt. 2   adjacency matrix: 7x7 = 49 cells for 7 citations; finding a paper's neighbours reads a whole row (O(n))
```

| | Neighbours of a paper | Whole search | Space |
|---|---|---|---|
| **Chosen: adjacency lists + BFS** | O(degree) | O(V + E) of the part reached, closest papers first | O(V + E) |
| Alt. 1: depth-first search (stack) on the same lists | O(degree) | O(V + E), but deep chains come before direct neighbours | O(V + E) |
| Alt. 2: adjacency matrix | O(n) (a whole row) | O(V²) | n² cells: 400 million for 20,000 papers |

And for the queue inside BFS:

| | Enqueue | Dequeue | Notes |
|---|---|---|---|
| **Chosen: circular queue on an array** | O(1) amortised (doubles when full) | O(1) | no shifting; the array is reused |
| Alt. 1: Python list with `pop(0)` | O(1) | O(n) (every item shifts left) | simplest |
| Alt. 2: linked-list queue | O(1) | O(1) | one node object per item |

BFS was chosen over DFS because related papers should come closest first: BFS
lists every paper one link away before any paper two links away (example above).
Measured live on 377 papers fetched from OpenAlex for eight graph learning
topics (16,525 references, 1,211 of them between papers in memory): finding the
related papers took 0.14 ms median and 0.75 ms at most per paper; 59 of the 377
papers had no link to another paper in memory.

## 2.13 Indexed priority queue and Dijkstra's shortest paths (added in 4.0)

**What it is.** `dsa/priority_queue.py` is a binary min-heap of (priority, key)
pairs in an array, plus a hash map from each key to its position in the array.
Every sift updates the map, so a key is found in O(1) and its priority can be
lowered in place ("decrease-key", O(log n)). `dsa/graph.py`'s `dijkstra` keeps
the unsettled papers in it, with the shortest distance found so far as the
priority: pop the closest paper, settle it, and for each edge lower the
neighbour's distance if the path through the settled paper is shorter.

```
=== Example 13: related papers by shortest paths (Dijkstra with an indexed priority queue)
edges     A-B 1, A-C 4, B-C 1, B-D 5, C-D 1
chosen   pop A 0: B 1, C 4
         pop B 1: C lowered 4 -> 2, D 6
         pop C 2: D lowered 6 -> 3
         pop D 3: nothing to lower
         order (paper, distance): [('B', 1.0), ('C', 2.0), ('D', 3.0)] - C through B beats the direct edge
alt. 1   BFS ignores lengths (paper, links): [('B', 1), ('C', 1), ('D', 2)] - C (length 4) ranks with B (length 1)
alt. 2   heap without decrease-key: one entry per improvement, stale ones skipped when popped; up to E = 5 entries instead of V = 4
```

| | Closest first | Cost | Space |
|---|---|---|---|
| **Chosen: Dijkstra with an indexed priority queue** | by total length | O((V + E) log V) for the part reached, stops after 8 papers | O(V) queue |
| Alt. 1: BFS (2.12) | by number of links only | O(V + E) | O(V) queue |
| Alt. 2: Dijkstra with a plain heap and stale entries | by total length | O((V + E) log E) | O(E) heap |
| Alt. 3: sort every paper by similarity to the opened one | by direct similarity only: misses a paper reached through a citation | O(n log n) per paper | O(n) |

Why: similarity alone (alt. 3) misses papers that are close through a chain
(a paper cited by a close neighbour), and BFS (alt. 1) treats a weak "similar"
link like a citation. The indexed queue keeps one entry per paper, which keeps
the queue small when a paper is reached by many edges (citations plus
similarity).

## 2.14 Radix trie for title suggestions (added in 4.0)

**What it is.** `dsa/trie.py` stores strings along paths from the root, with
each chain of single-child nodes merged into one edge labelled by a substring.
Insert walks the key and splits an edge where the key leaves it; remove takes
the value off and merges a node left with one child back into its edge;
`with_prefix` walks to the prefix (which may end inside an edge) and lists the
keys below it depth first with the array stack of `dsa/stack.py`.

```
=== Example 14: title suggestions (radix trie)
titles    "attention is all you need", "attention is not explanation", "attention mechanisms"
chosen   radix trie, one edge per merged chain:
         root
           -"attention "->
             -"is "->
               -"all you need"-> [1]
               -"not explanation"-> [2]
             -"mechanisms"-> [3]
         6 nodes; with_prefix('attention is') -> ['attention is all you need', 'attention is not explanation']
alt. 1   plain trie: one node per character: 51 nodes for the same three titles
alt. 2   sorted list + binary search for the prefix range: O(log n) to find, but every insert or eviction shifts the list (O(n))
```

| | Find titles with a prefix | Insert / remove | Nodes for 1,206 titles |
|---|---|---|---|
| **Chosen: radix trie** | O(prefix length + titles listed) | O(title length) | 1,865 |
| Alt. 1: plain trie | O(prefix length + characters below) | O(title length) | up to 82,768 (one per distinct prefix; the titles have 82,767 characters) |
| Alt. 2: sorted list + binary search | O(log n + titles listed) | O(n) shift | none (one list) |
| Alt. 3: scan every title | O(n × prefix length) | O(1) | none |

Why: titles arrive and leave all the time (every new search adds up to 175 papers,
eviction removes the oldest), so a structure with cheap insert and remove fits
better than a sorted list, and merging chains keeps the node count near the
number of titles instead of the number of characters.

## 2.15 Log-structured hash table for the cache file (added in 4.1)

**What it is.** `dsa/logstore.py` keeps records in a file that is only ever
appended to, one JSON line per record (key, expiry time, value), and a hash map
in memory from each key to the offset and length of its newest line. `get`
is one seek and one read. An update appends a new line and moves the map's
pointer; the old line becomes garbage. When the file has more than twice as
many lines as live keys (plus 1,000), or is over its size limit, `compact`
copies only the newest unexpired records to a new file and swaps it in; if that
is still over half the limit, the oldest records are dropped. Loading reads the
log once from start to end, so later lines win; a half-written last line from a
crash is cut off.

```
=== Example 15: cache file (log-structured hash table)
writes    put gnn, put qec, put gnn again
chosen    append-only log + hash map key -> (offset, length):
          offset   0: ["q:graph neural networks",null,["W1","W2"]]
          offset  45: ["q:quantum error correction",null,["W7"]]
          offset  88: ["q:graph neural networks",null,["W1","W2","W3"]]
          map: {'q:graph neural networks': (88, 50), 'q:quantum error correction': (45, 43)}
          get gnn = one seek to offset 88: ['W1', 'W2', 'W3']
          after a restart: the map is rebuilt by reading the log once -> 2 keys from 3 lines
          compact(): 2 lines, 93 bytes (the old gnn line is gone)
alt. 1    one JSON file rewritten on every put: O(size of the whole cache) per write, and a crash mid-write loses everything
alt. 2    a database (SQLite, a B-tree on disk): O(log n) per lookup, not hand-written, more than a cache of whole answers needs
```

| | Write an answer | Read an answer | Memory | After a crash |
|---|---|---|---|---|
| **Chosen: append-only log + hash map** | O(1) append | O(1) seek | one entry per key, values stay on disk | at most the last line is lost |
| Alt. 1: one JSON file, rewritten | O(whole file) | O(1) once all is loaded | every value in memory | the whole file can be lost |
| Alt. 2: SQLite (B-tree on disk) | O(log n) | O(log n) | page cache | safe (journal) |

Why: the cache is written once per new question and read on every repeat, and
its values are whole answers of 50 papers (about 75 KB), so keeping only
offsets in memory and never rewriting old bytes fits it. The garbage left by
updates is the price, paid back by compaction.

## Verification of the changes

1. **Unit tests.** `python -m unittest`: **75 tests, all pass**. Tests added for
   the version 3.3 changes:
   * `test_appended_postings_equal_a_full_rebuild` (tests/test_index.py): after
     every batch, through evictions and two compactions, the merged index gives
     exactly the same posting lists, BM25 scores and word counts as an index
     rebuilt from scratch.
   * `test_hash_map_and_binary_search_agree` (tests/test_index.py): the hash map
     returns the same id as binary search for every 7th vocabulary word and for
     missing words.
   * The existing `test_packed_scoring_equals_walking_each_tree`
     (tests/test_ml.py) checks the new tree walk against walking each tree
     separately.

   Tests added for version 3.4 (tests/test_dsa.py unless noted):
   * heap: `test_heap_order_and_pop`, and `test_top_k_equals_sorting` (with many
     equal scores, the heap returns the same papers as a full sort);
   * hash table: `test_collisions_in_one_bucket` (every key forced into one
     bucket), `test_rehash_keeps_load_low` (1,000 keys), `test_matches_a_dict`
     (5,000 random inserts, updates and deletes compared with a dict after each);
   * circular queue: `test_fifo_and_wraparound`;
   * BFS: `test_level_order_and_depth`, and `test_hops_are_shortest_paths`
     (on a random graph, the hops BFS reports equal shortest path lengths);
   * year tree: `test_range_matches_a_scan` (800 inserts, 200 removals),
     `test_balanced_build` (a sorted insert order gives height 127, the balanced
     build 7, same answers);
   * in the index: `test_year_tree_follows_adds_evictions_and_compaction` and
     `test_citation_graph_survives_eviction_and_compaction` (tests/test_index.py);
   * in the API: `test_related_papers_follow_citation_links` (tests/test_app.py):
     a chain of five papers gives the related papers in BFS order.
2. **Same results as before, on real data.** The old and new code were run on
   the same sequence: 12,000 real papers added in batches of 50 with a memory
   limit of 3,000 (so papers were evicted and the index compacted), and searches
   with topics, boolean queries, a prefix query, an exclusion, an author filter,
   sorting by year and (since version 3.4) three year filters. For the version
   3.3 changes all outputs compared equal byte for byte. For version 3.4, 479 of
   480 search outputs were identical; the one difference was a search with an
   author filter where two papers tied exactly at the 200th BM25 place (both
   4.268903). The heap kept the paper with the smaller id, numpy an arbitrary
   one, so the candidate count changed from 298 to 297. The first page of results
   was the same.
3. **Live.** With an OpenAlex key, searches fetched papers with their reference
   lists, the year filter returned only papers from the chosen years, and the
   paper drawer in the browser listed related papers (for "Stability Properties
   of Graph Neural Networks": two papers it cites, then six papers two links
   away); clicking one opened it.
4. **Speed.** Measured as described above; the numbers are in the summary table.

To repeat the examples: `python scripts/ds_examples.py` (or one example:
`python scripts/ds_examples.py 6`).
