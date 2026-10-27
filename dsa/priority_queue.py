"""Indexed priority queue: a binary min-heap of keys with priorities, plus a hash map
from each key to its position in the heap array.

The position map is what a plain heap (dsa/heap.py) lacks: it finds a key in O(1),
so a key's priority can be lowered in place ("decrease-key") and the key moved up
in O(log n), instead of pushing a duplicate and skipping stale copies later.

    push(key, priority)     O(log n)
    decrease(key, priority) O(log n)   only ever lowers a priority
    pop()                   O(log n)   the key with the smallest priority
    key in pq, len(pq)      O(1)

Use in the search engine: the frontier of Dijkstra's shortest paths for related
papers (dsa/graph.py). Each paper is a key, its priority the shortest distance
found so far from the opened paper; a shorter path found later lowers it.
"""


class IndexedMinPQ:
    def __init__(self):
        self._heap = []                 # [(priority, tiebreak, key)] in heap order
        self._pos = {}                  # key -> index in _heap
        self._count = 0                 # insertion counter: equal priorities leave in arrival order

    def __len__(self):
        return len(self._heap)

    def __contains__(self, key):
        return key in self._pos

    def priority(self, key):
        return self._heap[self._pos[key]][0]

    def push(self, key, priority):
        if key in self._pos:
            raise KeyError(f"{key!r} is already queued")
        self._heap.append((priority, self._count, key))
        self._count += 1
        self._pos[key] = len(self._heap) - 1
        self._up(len(self._heap) - 1)

    def decrease(self, key, priority):
        i = self._pos[key]
        old, order, _ = self._heap[i]
        if priority > old:
            raise ValueError("decrease() cannot raise a priority")
        self._heap[i] = (priority, order, key)
        self._up(i)

    def push_or_decrease(self, key, priority):
        """Queue the key, or lower its priority if this one is smaller. True if anything changed."""
        if key not in self._pos:
            self.push(key, priority)
            return True
        if priority < self.priority(key):
            self.decrease(key, priority)
            return True
        return False

    def pop(self):
        """(key, priority) with the smallest priority."""
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        top = self._heap[0]
        last = self._heap.pop()
        del self._pos[top[2]]
        if self._heap:
            self._heap[0] = last
            self._pos[last[2]] = 0
            self._down(0)
        return top[2], top[0]

    # ---- sifting, keeping _pos in step with every move
    def _place(self, i, entry):
        self._heap[i] = entry
        self._pos[entry[2]] = i

    def _up(self, i):
        a = self._heap
        entry = a[i]
        while i > 0:
            parent = (i - 1) // 2
            if a[parent][:2] <= entry[:2]:
                break
            self._place(i, a[parent])
            i = parent
        self._place(i, entry)

    def _down(self, i):
        a, n = self._heap, len(self._heap)
        entry = a[i]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and a[child + 1][:2] < a[child][:2]:
                child += 1
            if entry[:2] <= a[child][:2]:
                break
            self._place(i, a[child])
            i = child
        self._place(i, entry)
