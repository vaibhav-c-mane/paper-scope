"""Hash table with separate chaining, written by hand.

    bucket index = hash(key) mod number_of_buckets

* Hash function: polynomial rolling hash of the key's characters,
  h = h * 31 + code(char), kept to 64 bits. Two different keys can land in the
  same bucket (a collision).
* Collision resolution: separate chaining. Each bucket is a short list of
  [key, value] pairs; a lookup hashes the key, then compares keys only inside
  that one bucket.
* Overflow: when the load factor (entries / buckets) passes 0.75, the number of
  buckets doubles and every entry is re-hashed into the new buckets (rehashing).
  Doubling keeps the chains short, so get, put and delete are O(1) on average,
  and the occasional O(n) rehash is amortised over the inserts before it.

Use in the search engine: the identity-key map of the in-memory index
(engine/memindex.py), from keys such as "doi:10.1038/nature14539" or
"arxiv:1706.03762" to paper ids. Every paper that arrives from an API is checked
against it, so the same paper from two sources becomes one entry.
"""

_MASK = (1 << 64) - 1


def string_hash(key):
    """Polynomial rolling hash of a string, 64 bits."""
    h = 0
    for ch in str(key):
        h = (h * 31 + ord(ch)) & _MASK
    return h


class HashTable:
    MAX_LOAD = 0.75

    def __init__(self, buckets=16):
        self._buckets = [[] for _ in range(buckets)]
        self._size = 0

    def _bucket(self, key):
        return self._buckets[string_hash(key) % len(self._buckets)]

    # ---- the dict-like operations the index uses
    def get(self, key, default=None):
        for pair in self._bucket(key):
            if pair[0] == key:
                return pair[1]
        return default

    def __getitem__(self, key):
        for pair in self._bucket(key):
            if pair[0] == key:
                return pair[1]
        raise KeyError(key)

    def __contains__(self, key):
        return any(pair[0] == key for pair in self._bucket(key))

    def __setitem__(self, key, value):
        bucket = self._bucket(key)
        for pair in bucket:
            if pair[0] == key:
                pair[1] = value
                return
        bucket.append([key, value])
        self._size += 1
        if self._size > self.MAX_LOAD * len(self._buckets):
            self._rehash(2 * len(self._buckets))

    def setdefault(self, key, value):
        """The value stored for key; if there is none, store `value` and return it."""
        for pair in self._bucket(key):
            if pair[0] == key:
                return pair[1]
        self[key] = value
        return value

    def __delitem__(self, key):
        bucket = self._bucket(key)
        for i, pair in enumerate(bucket):
            if pair[0] == key:
                bucket[i] = bucket[-1]          # order inside a bucket does not matter
                bucket.pop()
                self._size -= 1
                return
        raise KeyError(key)

    def __len__(self):
        return self._size

    def items(self):
        return [(k, v) for bucket in self._buckets for k, v in bucket]

    def keys(self):
        return [k for bucket in self._buckets for k, _ in bucket]

    def __iter__(self):
        return iter(self.keys())

    # ---- inspection (tests, docs/PROJECT.md)
    def load_factor(self):
        return self._size / len(self._buckets)

    def longest_chain(self):
        return max((len(b) for b in self._buckets), default=0)

    def _rehash(self, n):
        old = self._buckets
        self._buckets = [[] for _ in range(n)]
        for bucket in old:
            for pair in bucket:
                self._buckets[string_hash(pair[0]) % n].append(pair)
