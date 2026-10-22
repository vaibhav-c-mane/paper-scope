"""Breadth-first search over a graph given as adjacency lists.

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
