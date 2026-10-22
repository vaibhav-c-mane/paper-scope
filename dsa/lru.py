"""LRU cache: a hash map for O(1) lookup plus a doubly linked list for O(1) recency order.

The list runs from least to most recently used. `get` and `put` move an entry to
the end; when the cache is over capacity the entry at the front is evicted. With
`ttl`, entries also expire after that many seconds (used for online API answers,
which can change). Used for the papers kept in memory and for API responses, so
both stay within a fixed size however long the app runs.
"""
import time


class _Node:
    __slots__ = ("key", "value", "expires", "prev", "next")

    def __init__(self, key=None, value=None, expires=None):
        self.key, self.value, self.expires = key, value, expires
        self.prev = self.next = None


class LRUCache:
    def __init__(self, capacity, ttl=None, on_evict=None, clock=time.monotonic):
        self.capacity = capacity
        self.ttl = ttl
        self.on_evict = on_evict            # called with (key, value) when an entry is pushed out
        self.clock = clock
        self._map = {}
        self._head = _Node()                # sentinels: head.next is the oldest entry, tail.prev the newest
        self._tail = _Node()
        self._head.next, self._tail.prev = self._tail, self._head

    # ------------------------------------------------------------- list helpers
    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _append(self, node):
        node.prev, node.next = self._tail.prev, self._tail
        self._tail.prev.next = node
        self._tail.prev = node

    # ------------------------------------------------------------- API
    def get(self, key, default=None):
        node = self._map.get(key)
        if node is None:
            return default
        if node.expires is not None and self.clock() >= node.expires:
            self._unlink(node)
            del self._map[key]
            return default
        self._unlink(node)
        self._append(node)
        return node.value

    def put(self, key, value, ttl=None):
        ttl = self.ttl if ttl is None else ttl
        expires = self.clock() + ttl if ttl else None
        node = self._map.get(key)
        if node is not None:
            node.value, node.expires = value, expires
            self._unlink(node)
            self._append(node)
            return
        node = self._map[key] = _Node(key, value, expires)
        self._append(node)
        while self.capacity and len(self._map) > self.capacity:
            oldest = self._head.next
            self._unlink(oldest)
            del self._map[oldest.key]
            if self.on_evict:
                self.on_evict(oldest.key, oldest.value)

    def touch(self, key):
        """Mark as recently used without reading the value."""
        node = self._map.get(key)
        if node is not None:
            self._unlink(node)
            self._append(node)

    def pop(self, key, default=None):
        node = self._map.pop(key, None)
        if node is None:
            return default
        self._unlink(node)
        return node.value

    def __contains__(self, key):
        return key in self._map

    def __len__(self):
        return len(self._map)

    def keys(self):
        """Keys from least to most recently used."""
        out, node = [], self._head.next
        while node is not self._tail:
            out.append(node.key)
            node = node.next
        return out

    def clear(self):
        self._map.clear()
        self._head.next, self._tail.prev = self._tail, self._head
