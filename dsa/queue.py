"""Circular queue on an array: enqueue at the rear, dequeue at the front, both O(1).

`front` is the index of the first item and `size` the number of items; the rear
is (front + size) mod capacity, so the array is reused in a circle instead of
shifting items after every dequeue. When it is full, the capacity doubles and
the items are copied in order (amortised O(1) per enqueue).

Use in the search engine: the frontier of the breadth-first search over the
citation graph (dsa/graph.py), which must visit papers in the order they were
discovered (first in, first out).
"""


class QueueUnderflow(IndexError):
    pass


class CircularQueue:
    def __init__(self, capacity=8):
        self._a = [None] * capacity
        self._front = 0
        self._size = 0

    def enqueue(self, x):
        if self._size == len(self._a):
            self._grow()
        self._a[(self._front + self._size) % len(self._a)] = x
        self._size += 1

    def dequeue(self):
        if not self._size:
            raise QueueUnderflow("dequeue from an empty queue")
        x = self._a[self._front]
        self._a[self._front] = None
        self._front = (self._front + 1) % len(self._a)
        self._size -= 1
        return x

    def is_empty(self):
        return not self._size

    def __len__(self):
        return self._size

    def items(self):
        """Front ... rear, for display."""
        return [self._a[(self._front + i) % len(self._a)] for i in range(self._size)]

    def _grow(self):
        self._a = self.items() + [None] * len(self._a)
        self._front = 0
