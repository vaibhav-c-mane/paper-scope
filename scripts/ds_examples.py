"""The small worked examples in docs/PROJECT.md, part 2: each core data structure
next to the alternatives it was compared with, on the same tiny input, with the
operations counted. Run it to reproduce every example block in that document:

  python scripts/ds_examples.py            all examples
  python scripts/ds_examples.py 3          only example 3

The inputs are a few made-up words and ids, small enough to follow by hand; they
illustrate the operations and are not used for training or measurements.
"""
import bisect
import heapq
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                      # noqa: E402

from dsa import merge                                   # noqa: E402
from dsa.bst import YearTree                            # noqa: E402
from dsa.graph import bfs, dijkstra                     # noqa: E402
from dsa.hashtable import HashTable, string_hash        # noqa: E402
from dsa.heap import MinHeap, top_k                     # noqa: E402
from dsa.queue import CircularQueue                     # noqa: E402
from dsa.expression import infix_to_postfix, tokenize_query   # noqa: E402
from dsa.lru import LRUCache                            # noqa: E402
from dsa.priority_queue import IndexedMinPQ              # noqa: E402
from dsa.trie import RadixTrie                          # noqa: E402
from engine.memindex import PaperIndex, _Grow           # noqa: E402


def title(n, text):
    print(f"\n=== Example {n}: {text}")


# ---------------------------------------------------------------- 1. inverted index update
def ex1():
    title(1, "inverted index, adding one paper")
    words = ["graph", "image", "network", "quantum"]
    papers = {0: ["graph", "network"], 1: ["graph", "image"], 2: ["network", "quantum"]}
    new = {3: ["graph", "quantum"]}

    def csc(ps):
        pairs = sorted((words.index(w), d) for d, ws in ps.items() for w in ws)   # by word, then paper
        indptr = np.searchsorted([t for t, _ in pairs], np.arange(len(words) + 1))
        return indptr, np.asarray([d for _, d in pairs])

    indptr, docs = csc(papers)
    print("before   indptr", indptr.tolist(), "docs", docs.tolist())
    print("         graph -> docs[0:2] =", docs[0:2].tolist())
    ip_new, d_new = csc(new)
    merged_ptr, (merged,) = PaperIndex._append_lists(indptr, (docs,), ip_new, (d_new,))
    print("chosen   append merge: new postings", d_new.tolist(), "for words", [words[t] for t in np.repeat(
        np.arange(len(words)), np.diff(ip_new))])
    print("         after  indptr", merged_ptr.tolist(), "docs", merged.tolist())
    print(f"         operations: {len(docs) + len(d_new)} array writes, no comparisons (paper 3 > every old id)")
    full_ptr, full = csc({**papers, **new})
    n = len(full)
    print(f"alt. 1   full rebuild: sort all {n} (word, paper) pairs again,"
          f" about n log2 n = {n * np.log2(n):.0f} comparisons; same result: {full.tolist() == merged.tolist()}")
    lists = {w: [d for d, ws in sorted(papers.items()) if w in ws] for w in words}
    for d, ws in new.items():
        for w in ws:
            lists[w].append(d)
    print("alt. 2   hash map of Python lists:", lists)
    print("         append is 1 operation per new posting, but every id is a separate Python object")


# ---------------------------------------------------------------- 2. identity keys
def ex2():
    title(2, "finding a paper by an identity key (hand-written hash table)")
    keys = ["arxiv:1706.03762", "doi:10.1038/nature14539", "openalex:w2741809807", "doi:10.1145/3292500"]
    t = HashTable(buckets=4)
    for d, k in enumerate(keys):
        before = len(t._buckets)
        t[k] = d
        grew = f"  -> load {len(t)}/{before} > 0.75: rehash into {len(t._buckets)} buckets" if len(t._buckets) != before else ""
        print(f"chosen   put {k:<24} hash mod {before} = bucket {string_hash(k) % before}{grew}")
    print("         buckets now:", [[k for k, _ in b] for b in t._buckets])
    probe = "doi:10.1145/3292500"
    b = string_hash(probe) % len(t._buckets)
    print(f"         get {probe}: bucket {b}, compare {[k for k, _ in t._buckets[b]].index(probe) + 1} key(s) -> {t[probe]}")
    steps = next(i for i, k in enumerate(keys, 1) if k == probe)
    print(f"alt. 1   list scan: compare keys one by one -> {steps} comparisons (n in the worst case)")
    srt = sorted(keys)
    lo, hi, comps = 0, len(srt), 0
    while lo < hi:
        mid = (lo + hi) // 2
        comps += 1
        if srt[mid] < probe:
            lo = mid + 1
        else:
            hi = mid
    print(f"alt. 2   sorted list + binary search: {comps} comparisons (log2 n);"
          f" but inserting a new key shifts {len(srt) - bisect.bisect_left(srt, 'doi:10.1/new')} keys to the right")


# ---------------------------------------------------------------- 3. LRU
def ex3():
    title(3, "keeping the most recently used papers (capacity 3)")
    ops = [("put", "A"), ("put", "B"), ("put", "C"), ("get", "A"), ("put", "D")]
    evicted = []
    c = LRUCache(3, on_evict=lambda k, v: evicted.append(k))
    for op, k in ops:
        c.put(k, k) if op == "put" else c.get(k)
        print(f"chosen   {op} {k}: list oldest->newest {c.keys()}" + (f"  evicted {evicted[-1]}" if evicted and op == "put" and k == "D" else ""))
    print("         every step: 1 hash lookup + at most 4 pointer changes, O(1)")
    order, shifts = [], 0
    for op, k in ops:
        if op == "get":
            i = order.index(k)
            shifts += len(order) - 1 - i
            order.pop(i)
            order.append(k)
        else:
            order.append(k)
            if len(order) > 3:
                order.pop(0)
                shifts += len(order)
    print(f"alt. 1   array in recency order: same final order {order}, but get/evict shift elements"
          f" ({shifts} moves here; O(n) per operation)")
    clock, last, heap, before_evict, gone = 0, {}, [], None, None
    for op, k in ops:
        clock += 1
        last[k] = clock
        heapq.heappush(heap, (clock, k))                    # a get pushes a second entry for A
        if op == "put" and len(last) > 3:
            before_evict = (len(heap), len(last))
            while True:
                t, old = heapq.heappop(heap)
                if last.get(old) == t:                      # skip the stale (1, 'A') entry
                    del last[old]
                    gone = old
                    break
    print(f"alt. 2   min-heap of last-use times: evicts {gone} as well, O(log n) per operation, but before the"
          f" eviction it held {before_evict[0]} entries for {before_evict[1]} papers (stale entries stay until popped)")


# ---------------------------------------------------------------- 4. stack
def ex4():
    title(4, "boolean query on a stack: deep AND (vision OR image) NOT survey")
    tokens = tokenize_query("deep AND (vision OR image) NOT survey")
    postfix, steps = infix_to_postfix(tokens)
    for s in steps:
        print(f"         token {s['token']:<7} stack {s['stack']}")
    print("chosen   postfix:", " ".join(postfix), f"({len(tokens)} tokens, each pushed and popped at most once: O(n))")
    print("         array stack (dsa/stack.py): push/pop at the end of one array, no extra object per item")
    print("alt. 1   linked-list stack: same O(1) push/pop, plus one node object (value + next pointer) per push")
    print("alt. 2   recursive-descent parser: same O(n), but the stack is Python's call stack, so the")
    print("         intermediate stack states (shown in the search trace) cannot be recorded")


# ---------------------------------------------------------------- 5. merging posting lists
def ex5():
    title(5, "AND of two sorted posting lists")
    a, b = [1, 4, 7, 9], [2, 4, 9, 12]
    i = j = comps = 0
    out = []
    while i < len(a) and j < len(b):
        comps += 1
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    assert out == merge.intersect(a, b)
    print(f"chosen   merge pass: {out}, {comps} comparisons (at most m + n), result already sorted")
    print(f"alt. 1   nested loops: {len(a) * len(b)} comparisons (m * n), result {[x for x in a if x in b]}")
    print(f"alt. 2   hash set: {len(a) + len(b)} hash operations, result {sorted(set(a) & set(b))} only after"
          " sorting again (a set has no order), plus a set object per list")


# ---------------------------------------------------------------- 6. sorting the results
def ex6():
    title(6, "sorting results by year, newest first (rows already in relevance order)")
    rows = [("A", 2019), ("B", 2021), ("C", 2019), ("D", 2021)]
    print("input    ", rows)
    print("chosen   merge sort:", merge.merge_sort(rows, key=lambda r: r[1], reverse=True),
          "(B before D and A before C, as in the relevance order: stable)")

    def quicksort(xs):                                   # Lomuto partition, last element as pivot
        xs = list(xs)

        def part(lo, hi):
            if lo >= hi:
                return
            p, i = xs[hi][1], lo
            for j in range(lo, hi):
                if xs[j][1] >= p:
                    xs[i], xs[j] = xs[j], xs[i]
                    i += 1
            xs[i], xs[hi] = xs[hi], xs[i]
            part(lo, i - 1)
            part(i + 1, hi)
        part(0, len(xs) - 1)
        return xs
    print("alt. 1   quicksort:", quicksort(rows), "(equal years swapped: not stable; O(n^2) worst case)")
    def heapsort(xs):                                    # in-place heap sort on the year only
        xs = list(xs)

        def sift(i, n):
            while 2 * i + 1 < n:
                c = 2 * i + 1
                if c + 1 < n and xs[c + 1][1] < xs[c][1]:
                    c += 1
                if xs[c][1] < xs[i][1]:
                    xs[i], xs[c] = xs[c], xs[i]
                    i = c
                else:
                    return
        for i in range(len(xs) // 2 - 1, -1, -1):
            sift(i, len(xs))
        for end in range(len(xs) - 1, 0, -1):          # min-heap: the oldest year goes to the end
            xs[0], xs[end] = xs[end], xs[0]
            sift(0, end)
        return xs
    print("alt. 2   heap sort:", heapsort(rows), "(equal years reordered by the heap: not stable)")


# ---------------------------------------------------------------- 7. vocabulary
def ex7():
    title(7, "looking up words in the vocabulary")
    vocab = sorted(["graph", "image", "net", "network", "neural", "optimal", "optimize", "quantum"])
    ids = {w: i for i, w in enumerate(vocab)}
    print("vocab    ", vocab)
    print("chosen   exact 'network': hash map -> id", ids["network"], "(1 hash, 1 comparison)")
    lo, hi = bisect.bisect_left(vocab, "optim"), bisect.bisect_left(vocab, "optim￿")
    print(f"         prefix 'optim*': two binary searches in the sorted array -> ids {lo}..{hi - 1} {vocab[lo:hi]}")
    lo_, hi_, c = 0, len(vocab), 0
    while lo_ < hi_:
        mid = (lo_ + hi_) // 2
        c += 1
        lo_, hi_ = (mid + 1, hi_) if vocab[mid] < "network" else (lo_, mid)
    print(f"alt. 1   binary search only: exact 'network' takes {c} string comparisons (log2 V; 18 for 240,705 words)")
    trie = {}
    for w in vocab:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node["$"] = ids[w]

    def count(n):
        return 1 + sum(count(v) for k, v in n.items() if k != "$")
    print(f"alt. 2   trie: 'network' walks 7 nodes, prefix 'optim*' walks 5 then lists the subtree;"
          f" {count(trie)} nodes for {len(vocab)} words")


# ---------------------------------------------------------------- 8. dynamic arrays
def ex8():
    title(8, "appending 20 vector rows one at a time")
    g, copies, cap_seen = _Grow(2), 0, []
    for _ in range(20):
        before = len(g.buf)
        g.extend(np.zeros((1, 2)))
        if len(g.buf) != before:
            copies += before
            cap_seen.append(len(g.buf))
    print(f"chosen   doubling array: capacity 16 -> {cap_seen}, {copies} rows copied in total (amortised O(1) each)")
    exact = sum(range(20))
    print(f"alt. 1   exact-size array, new copy on every append: {exact} rows copied (O(n) each, O(n^2) total)")
    print("alt. 2   Python list of row lists: appends are cheap, but there is no single matrix to multiply,")
    print("         so 'vectors @ query' becomes a Python loop over every paper")


# ---------------------------------------------------------------- 9. top k
def ex9():
    title(9, "the 3 best of 8 scores (size-k min-heap)")
    s = [0.2, 0.9, 0.1, 0.7, 0.4, 0.8, 0.3, 0.6]
    print("scores   ", s)
    h, compared, sifted = MinHeap(), 0, 0
    for d, x in enumerate(s):
        if len(h) < 3:
            h.push((x, -d))
            print(f"{'chosen' if d == 0 else '':<9}paper {d} ({x}): heap not full, push   heap {[v for v, _ in h.items()]}")
            continue
        compared += 1
        if (x, -d) > h.peek():
            sifted += 1
            h.replace_top((x, -d))
            print(f"         paper {d} ({x}): beats the root, replace heap {[v for v, _ in h.items()]}")
        else:
            print(f"         paper {d} ({x}): not above root {h.peek()[0]}, dropped")
    print(f"         best 3: papers {sorted(top_k(s, range(len(s)), 3))}; {compared} root comparisons,"
          f" {sifted} sift-downs of at most log2 k levels: O(n log k), O(k) memory")
    a = np.asarray(s)
    print("alt. 1   partial selection (numpy argpartition):", sorted(np.argpartition(-a, 2)[:3].tolist()),
          "O(n) on average, but needs all n scores in one array")
    print("alt. 2   full sort (argsort):", np.argsort(-a)[:3].tolist(), "O(n log n), orders all 8 to keep 3")


# ---------------------------------------------------------------- 10. ranker trees
def ex10():
    title(10, "scoring one candidate with one regression tree")
    #      node 0: feature 0 (bm25) <= 0.5 ?
    #        yes -> node 1: leaf -0.3
    #        no  -> node 2: feature 1 (semantic) <= 0.7 ?  yes -> node 3: leaf 0.2   no -> node 4: leaf 0.9
    feature, threshold = [0, -2, 1, -2, -2], [0.5, 0, 0.7, 0, 0]
    left, right, value = [1, -1, 3, -1, -1], [2, -1, 4, -1, -1], [0, -0.3, 0, 0.2, 0.9]
    x, node, path = [0.8, 0.75], 0, [0]
    while left[node] >= 0:
        node = left[node] if x[feature[node]] <= threshold[node] else right[node]
        path.append(node)
    print(f"chosen   arrays feature/threshold/left/right/value; x = {x}: path {path} -> {value[node]}")
    print("         (all candidates and trees walk together; a pair leaves the active array at its leaf)")
    print("alt. 1   node objects with left/right pointers: same path, one Python object per node,")
    print("         walked one candidate at a time in Python")
    print("alt. 2   dense level-by-level matrix of positions: every (candidate, tree) pair is updated at every")
    print("         level of the deepest tree (19 for the shipped model) even after reaching its leaf")


# ---------------------------------------------------------------- 11. year range
def ex11():
    title(11, "papers from 2018 to 2020 (binary search tree keyed by year)")
    papers = {"A": 2019, "B": 2015, "C": 2021, "D": 2017, "E": 2020, "F": 2023, "G": 2019, "H": 2012}
    tree = YearTree()
    for name, year in papers.items():
        tree.insert(year, name)
    visited = []
    node, stack = tree.root, []
    while stack or node is not None:                     # the same pruned in-order walk as YearTree.range
        if node is not None:
            stack.append(node)
            node = node.left if node.year > 2018 else None
            continue
        node = stack.pop()
        visited.append(node.year)
        node = node.right if node.year < 2020 else None
    print("insert   ", list(papers.items()))
    print(f"chosen   tree of {len(tree.years())} years, height {tree.height()}; range(2018, 2020) visits years"
          f" {visited} -> {tree.range(2018, 2020)}")
    print(f"         ({len(visited)} of {len(tree.years())} nodes: subtrees outside the range are skipped; O(h + years visited + results))")
    print(f"alt. 1   check every paper's year: {len(papers)} comparisons (O(n)) ->",
          sorted(n for n, y in papers.items() if 2018 <= y <= 2020))
    rows = sorted((y, n) for n, y in papers.items())
    lo, hi = bisect.bisect_left(rows, (2018, "")), bisect.bisect_right(rows, (2020, "~"))
    print(f"alt. 2   sorted array + binary search: rows {lo}..{hi - 1} -> {[n for _, n in rows[lo:hi]]} in O(log n),"
          f" but adding a 2016 paper shifts {len(rows) - bisect.bisect_left(rows, (2016, ''))} rows")


# ---------------------------------------------------------------- 12. related papers
def ex12():
    title(12, "related papers by citation links (BFS with a circular queue)")
    cites = {"P": ["A", "B"], "A": ["C"], "B": ["C", "D"], "C": ["E"], "D": [], "E": [], "F": ["P"]}
    graph = {v: list(c) for v, c in cites.items()}       # undirected: "cites" or "is cited by"
    for v, cs in cites.items():
        for c in cs:
            graph[c].append(v)
    print("cites    ", cites)
    q, seen, order = CircularQueue(capacity=2), {"P"}, []
    q.enqueue(("P", 0))
    while not q.is_empty():
        node, hops = q.dequeue()
        for nxt in graph[node]:
            if nxt not in seen and hops < 2:
                seen.add(nxt)
                order.append((nxt, hops + 1))
                q.enqueue((nxt, hops + 1))
        print(f"{'chosen' if node == 'P' else '':<9}dequeue {node} (hop {hops}): queue {[n for n, _ in q.items()]}")
    assert [(n, h) for n, h, _ in bfs("P", graph.get, max_depth=2)] == order
    print("         BFS order (paper, hops):", order, "- every 1-hop paper before any 2-hop one")
    dfs, stack, seen = [], [("P", 0)], set()
    while stack:                                         # depth-first: a stack instead of the queue
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node != "P":
            dfs.append((node, depth))
        stack += [(nxt, depth + 1) for nxt in reversed(graph[node]) if nxt not in seen]
    print("alt. 1   depth-first search (stack) order, with path length:", dfs)
    print("         it follows one chain as deep as it goes: B, which P cites, is reached by a 3-link path,"
          " and F, which cites P, comes last")
    names = sorted(graph)
    print(f"alt. 2   adjacency matrix: {len(names)}x{len(names)} = {len(names) ** 2} cells for"
          f" {sum(map(len, cites.values()))} citations; finding a paper's neighbours reads a whole row (O(n))")



def ex13():
    title(13, "related papers by shortest paths (Dijkstra with an indexed priority queue)")
    edges = [("A", "B", 1), ("A", "C", 4), ("B", "C", 1), ("B", "D", 5), ("C", "D", 1)]
    graph = {}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w, ""))
        graph.setdefault(v, []).append((u, w, ""))
    print("edges    ", ", ".join(f"{u}-{v} {w}" for u, v, w in edges))
    pq, dist, done = IndexedMinPQ(), {"A": 0}, set()
    pq.push("A", 0)
    while len(pq):
        node, d = pq.pop()
        done.add(node)
        notes = []
        for nxt, w, _ in graph[node]:
            if nxt in done:
                continue
            old = pq.priority(nxt) if nxt in pq else None
            if pq.push_or_decrease(nxt, d + w):
                notes.append(f"{nxt} lowered {old} -> {d + w}" if old is not None else f"{nxt} {d + w}")
                dist[nxt] = d + w
        print(f"{'chosen' if node == 'A' else '':<9}pop {node} {d}: " + (", ".join(notes) or "nothing to lower"))
    order = [(n, d) for n, d, _, _ in dijkstra("A", graph.get)]
    print("         order (paper, distance):", order, "- C through B beats the direct edge")
    hops = [(n, h) for n, h, _ in bfs("A", lambda n: [m for m, _, _ in graph[n]], max_depth=3)]
    print("alt. 1   BFS ignores lengths (paper, links):", hops, "- C (length 4) ranks with B (length 1)")
    print(f"alt. 2   heap without decrease-key: one entry per improvement, stale ones skipped when popped;"
          f" up to E = {len(edges)} entries instead of V = {len(graph)}")


def ex14():
    title(14, "title suggestions (radix trie)")
    titles = ["attention is all you need", "attention is not explanation", "attention mechanisms"]
    t = RadixTrie()
    for i, k in enumerate(titles):
        t.insert(k, i + 1)
    print("titles   ", ", ".join(f'"{k}"' for k in titles))

    def show(node, label, depth):
        print((" " * 9 + "  " * depth + (f'-"{label}"-> ' if label else "root ") + (str(sorted(node.values)) if node.values else "")).rstrip())
        for lab, child in node.edges.values():
            show(child, lab, depth + 1)
    print("chosen   radix trie, one edge per merged chain:")
    show(t.root, "", 0)
    print(f"         {t.node_count()} nodes; with_prefix('attention is') ->", sorted(k for k, _ in t.with_prefix("attention is")))
    plain = len({k[:i] for k in titles for i in range(1, len(k) + 1)}) + 1
    print(f"alt. 1   plain trie: one node per character: {plain} nodes for the same three titles")
    print("alt. 2   sorted list + binary search for the prefix range: O(log n) to find,"
          " but every insert or eviction shifts the list (O(n))")


EXAMPLES = [ex1, ex2, ex3, ex4, ex5, ex6, ex7, ex8, ex9, ex10, ex11, ex12, ex13, ex14]

if __name__ == "__main__":
    chosen = [int(a) for a in sys.argv[1:]] or range(1, len(EXAMPLES) + 1)
    for n in chosen:
        EXAMPLES[n - 1]()
