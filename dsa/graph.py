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
