class MinHeap:
    def __init__(self, items=None):
        self._a = list(items or [])
        for i in range(len(self._a) // 2 - 1, -1, -1):      # heapify: O(n), bottom-up
            self._sift_down(i)

    def __len__(self):
        return len(self._a)

    def peek(self):
        if not self._a:
            raise IndexError("peek from an empty heap")
        return self._a[0]

    def push(self, x):
        self._a.append(x)
        self._sift_up(len(self._a) - 1)

    def pop(self):
        """Remove and return the smallest item."""
        if not self._a:
            raise IndexError("pop from an empty heap")
        last = self._a.pop()
        if not self._a:
            return last
        top, self._a[0] = self._a[0], last
        self._sift_down(0)
        return top

    def replace_top(self, x):
        """Pop the smallest item and push x in one sift: O(log n)."""
        top, self._a[0] = self._a[0], x
        self._sift_down(0)
        return top

    def items(self):
        """The array, level by level (for display and tests)."""
        return list(self._a)

    def _sift_up(self, i):
        a = self._a
        x = a[i]
        while i > 0:
            parent = (i - 1) // 2
            if a[parent] <= x:
                break
            a[i] = a[parent]
            i = parent
        a[i] = x

    def _sift_down(self, i):
        a, n = self._a, len(self._a)
        x = a[i]
        while True:
            child = 2 * i + 1
            if child >= n:
                break
            if child + 1 < n and a[child + 1] < a[child]:
                child += 1
            if x <= a[child]:
                break
            a[i] = a[child]
            i = child
        a[i] = x


def top_k(scores, ids, k):
    """The ids of the k largest scores. Equal scores: the smaller id wins, so the
    answer does not depend on the order the scores arrive in."""
    if k <= 0:
        return []
    heap = MinHeap()
    for s, d in zip(scores, ids):
        entry = (s, -d)                     # weakest = lowest score, then the largest id
        if len(heap) < k:
            heap.push(entry)
        elif entry > heap.peek():
            heap.replace_top(entry)
    return [-d for _, d in heap.items()]
