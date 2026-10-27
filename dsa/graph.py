"""Breadth-first search and Dijkstra's shortest paths over graphs given as adjacency lists.

Use in the search engine: "related papers" for the paper drawer. The papers in
memory form a citation graph: an edge joins two papers when one cites the
other (OpenAlex reports each work's reference list). BFS from one paper visits
its direct neighbours first (1 hop: papers it cites or that cite it), then
their neighbours (2 hops), and so on, so the papers found first are the most
closely linked. Each paper is visited once (the `seen` hash set) and each edge
looked at once: O(V + E) for the part of the graph reached.

Depth-first search would also visit every reachable paper, but in an order
that follows one chain of citations as deep as it goes before trying the next,
so a paper 5 hops away could be listed before a direct neighbour. For "most
closely related first", BFS's level-by-level order is the one needed.
"""
from .priority_queue import IndexedMinPQ
from .queue import CircularQueue


def bfs(start, neighbours, max_depth=2, limit=None):
    """[(node, hops, parent)] in BFS order, without `start`. `neighbours(node)` returns
    the adjacency list of a node. Stops at `max_depth` hops or after `limit` nodes."""
    seen = {start}
    queue = CircularQueue()
    queue.enqueue((start, 0))
    found = []
    while not queue.is_empty():
        node, depth = queue.dequeue()
        if depth == max_depth:
            continue
        for nxt in neighbours(node):
            if nxt in seen:
                continue
            seen.add(nxt)
            found.append((nxt, depth + 1, node))
            if limit is not None and len(found) >= limit:
                return found
            queue.enqueue((nxt, depth + 1))
    return found


def dijkstra(start, neighbours, limit=None, max_dist=float("inf")):
    """[(node, distance, parent, label of the last edge)] in order of distance from `start`,
    without `start`. `neighbours(node)` returns (next node, edge length >= 0, label) triples.

    Use in the search engine: related papers. Edges join a paper to the papers it cites,
    that cite it, that OpenAlex lists as related, and its nearest neighbours by meaning;
    an edge's length is how different the two papers are (1 - cosine of their vectors),
    shorter for a citation. BFS would treat every link as equally close; Dijkstra ranks a
    paper reached by two short links above one reached by a single long one.

    The frontier is an indexed priority queue (dsa/priority_queue.py): pop the closest
    unsettled paper, settle it, and relax its edges (lower a neighbour's distance with
    decrease-key when the path through it is shorter). O((V + E) log V) for the part of the
    graph reached; it stops after `limit` papers are settled, so a big graph is not explored."""
    dist, parent, label = {start: 0.0}, {start: None}, {start: None}
    done, out = set(), []
    pq = IndexedMinPQ()
    pq.push(start, 0.0)
    while len(pq):
        node, d = pq.pop()
        done.add(node)
        if node != start:
            out.append((node, d, parent[node], label[node]))
            if limit is not None and len(out) >= limit:
                break
        for nxt, w, lab in neighbours(node):
            if nxt in done or w < 0:
                continue
            nd = d + w
            if nd <= max_dist and pq.push_or_decrease(nxt, nd):
                dist[nxt], parent[nxt], label[nxt] = nd, node, lab
    return out
