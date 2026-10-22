"""The hand-written syllabus structures the engine uses. Run: python -m unittest -v"""
import random
import unittest

from dsa.bst import YearTree
from dsa.graph import bfs
from dsa.hashtable import HashTable
from dsa.heap import MinHeap, top_k
from dsa.expression import evaluate_postfix, infix_to_postfix, tokenize_query
from dsa.lru import LRUCache
from dsa.merge import difference, intersect, merge, merge_sort, union
from dsa.queue import CircularQueue, QueueUnderflow
from dsa.stack import Stack, StackUnderflow


class StackTests(unittest.TestCase):
    def test_push_pop_peek(self):
        s = Stack()
        for x in "abc":
            s.push(x)
        self.assertEqual((s.peek(), len(s), s.items()), ("c", 3, ["a", "b", "c"]))
        self.assertEqual([s.pop(), s.pop(), s.pop()], ["c", "b", "a"])
        self.assertTrue(s.is_empty())
        with self.assertRaises(StackUnderflow):
            s.pop()


class ExpressionTests(unittest.TestCase):
    def test_tokenize_adds_implicit_and(self):
        self.assertEqual(tokenize_query("deep learning -survey"), ["deep", "AND", "learning", "AND", "NOT", "survey"])
        self.assertEqual(tokenize_query("(a OR b) c"), ["(", "a", "OR", "b", ")", "AND", "c"])

    def test_shunting_yard_precedence(self):
        post, steps = infix_to_postfix(tokenize_query("deep AND (vision OR image) NOT survey"))
        # "x NOT y" is read as "x AND NOT y"; NOT1 is the unary "everything except"
        self.assertEqual(post, ["deep", "vision", "image", "OR", "AND", "survey", "NOT1", "AND"])
        self.assertEqual(steps[-1]["token"], "end")
        self.assertEqual(infix_to_postfix(["NOT", "a"])[0], ["a", "NOT1"])
        with self.assertRaises(ValueError):
            infix_to_postfix(["(", "a"])

    def test_evaluate_with_posting_lists(self):
        post = {"a": [1, 2, 3, 5], "b": [2, 5, 7], "c": [3]}
        run = lambda q: evaluate_postfix(infix_to_postfix(tokenize_query(q))[0], post.get, list(range(8)))
        self.assertEqual(run("a AND b"), [2, 5])
        self.assertEqual(run("a OR b"), [1, 2, 3, 5, 7])
        self.assertEqual(run("a NOT b"), [1, 3])
        self.assertEqual(run("(a OR c) AND NOT b"), [1, 3])
        self.assertEqual(run("NOT a"), [0, 4, 6, 7])


class MergeTests(unittest.TestCase):
    def test_merge_and_sort(self):
        data = [random.randint(0, 50) for _ in range(300)]
        self.assertEqual(merge_sort(data), sorted(data))
        self.assertEqual(merge_sort(data, reverse=True), sorted(data, reverse=True))
        self.assertEqual(merge([1, 4, 9], [2, 3, 10]), [1, 2, 3, 4, 9, 10])
        self.assertEqual(merge_sort([]), [])

    def test_merge_sort_is_stable(self):
        rows = [(random.randint(0, 5), i) for i in range(200)]
        out = merge_sort(rows, key=lambda r: r[0], reverse=True)
        self.assertEqual(out, sorted(rows, key=lambda r: -r[0]))      # Python's sort is stable too

    def test_set_operations(self):
        a, b = [1, 3, 5, 7], [3, 4, 5, 8]
        self.assertEqual(intersect(a, b), [3, 5])
        self.assertEqual(union(a, b), [1, 3, 4, 5, 7, 8])
        self.assertEqual(difference(a, b), [1, 7])


class LRUTests(unittest.TestCase):
    def test_eviction_order_and_callback(self):
        evicted = []
        c = LRUCache(3, on_evict=lambda k, v: evicted.append(k))
        for k in "abc":
            c.put(k, k.upper())
        c.get("a")                       # a becomes most recent
        c.put("d", "D")                  # b is the least recently used
        self.assertEqual(evicted, ["b"])
        self.assertEqual(c.keys(), ["c", "a", "d"])
        c.touch("c")
        c.put("e", "E")
        self.assertEqual(evicted, ["b", "a"])
        self.assertEqual(c.pop("c"), "C")
        self.assertNotIn("c", c)

    def test_expiry(self):
        now = [0.0]
        c = LRUCache(10, ttl=5, clock=lambda: now[0])
        c.put("q", [1])
        c.put("r", None, ttl=60)          # a cached "not found" is a value too
        now[0] = 4.9
        self.assertEqual(c.get("q"), [1])
        now[0] = 5.0
        self.assertIsNone(c.get("q"))
        self.assertEqual(c.get("r", "miss"), None)
        self.assertEqual(c.get("zzz", "miss"), "miss")
        self.assertEqual(len(c), 1)


class HeapTests(unittest.TestCase):
    def test_heap_order_and_pop(self):
        rng = random.Random(4)
        xs = [rng.randint(0, 50) for _ in range(200)]
        h = MinHeap(xs[:100])                                   # heapify
        for x in xs[100:]:
            h.push(x)
        a = h.items()
        self.assertTrue(all(a[(i - 1) // 2] <= a[i] for i in range(1, len(a))))   # parent <= child
        self.assertEqual([h.pop() for _ in range(len(xs))], sorted(xs))
        with self.assertRaises(IndexError):
            h.pop()

    def test_top_k_equals_sorting(self):
        rng = random.Random(5)
        scores = [rng.choice([0.1, 0.5, 0.5, 0.9]) + rng.random() / 10 for _ in range(300)] + [0.5] * 20
        ids = list(range(len(scores)))
        best = sorted(ids, key=lambda d: (-scores[d], d))       # ties: smaller id first
        for k in (0, 1, 7, 200, 400):
            self.assertEqual(sorted(top_k(scores, ids, k)), sorted(best[:k]))


class HashTableTests(unittest.TestCase):
    def test_collisions_in_one_bucket(self):
        t = HashTable(buckets=1)
        t.MAX_LOAD = 100                    # never rehash: every key shares bucket 0
        for i, k in enumerate(["doi:a", "doi:b", "arxiv:c"]):
            t[k] = i
        self.assertEqual((t["doi:a"], t["doi:b"], t["arxiv:c"], t.longest_chain()), (0, 1, 2, 3))
        del t["doi:a"]
        self.assertNotIn("doi:a", t)
        self.assertEqual((t.get("doi:b"), t.get("doi:a", -1), len(t)), (1, -1, 2))
        with self.assertRaises(KeyError):
            del t["doi:a"]

    def test_rehash_keeps_load_low(self):
        t = HashTable(buckets=4)
        for i in range(1000):
            t["k%d" % i] = i
        self.assertLessEqual(t.load_factor(), HashTable.MAX_LOAD)
        self.assertTrue(all(t["k%d" % i] == i for i in range(1000)))

    def test_matches_a_dict(self):
        rng, t, d = random.Random(3), HashTable(), {}
        for _ in range(5000):
            k, op = "doi:%d" % rng.randrange(400), rng.random()
            if op < 0.5:
                t[k] = d[k] = rng.randrange(99)
            elif op < 0.7:
                self.assertEqual(t.setdefault(k, 7), d.setdefault(k, 7))
            elif k in d:
                del t[k], d[k]
            self.assertEqual((k in t, t.get(k), len(t)), (k in d, d.get(k), len(d)))
        self.assertEqual(sorted(t.items()), sorted(d.items()))


class QueueTests(unittest.TestCase):
    def test_fifo_and_wraparound(self):
        q, out = CircularQueue(capacity=2), []
        for i in range(10):                 # interleaved: the rear wraps past the end, then the array grows
            q.enqueue(i)
            if i % 3 == 2:
                out.append(q.dequeue())
        self.assertEqual(q.items(), [i for i in range(10) if i not in out])
        while not q.is_empty():
            out.append(q.dequeue())
        self.assertEqual(out, list(range(10)))
        with self.assertRaises(QueueUnderflow):
            q.dequeue()


class BFSTests(unittest.TestCase):
    GRAPH = {1: [2, 3], 2: [4], 3: [4, 5], 4: [6], 5: [], 6: [1]}

    def test_level_order_and_depth(self):
        found = bfs(1, self.GRAPH.get, max_depth=3)
        self.assertEqual(found, [(2, 1, 1), (3, 1, 1), (4, 2, 2), (5, 2, 3), (6, 3, 4)])
        self.assertEqual([n for n, _, _ in bfs(1, self.GRAPH.get, max_depth=1)], [2, 3])
        self.assertEqual(len(bfs(1, self.GRAPH.get, max_depth=9, limit=3)), 3)

    def test_hops_are_shortest_paths(self):
        rng = random.Random(8)
        g = {v: rng.sample(range(60), 3) for v in range(60)}
        hops = {n: h for n, h, _ in bfs(0, g.get, max_depth=60)}
        dist, frontier = {0: 0}, [0]       # reference: plain level-by-level expansion
        while frontier:
            nxt = []
            for v in frontier:
                for w in g[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        nxt.append(w)
            frontier = nxt
        del dist[0]
        self.assertEqual(hops, dist)


class YearTreeTests(unittest.TestCase):
    def test_range_matches_a_scan(self):
        rng, tree, papers = random.Random(2), YearTree(), {}
        for d in range(800):
            papers[d] = rng.randrange(1950, 2026)
            tree.insert(papers[d], d)
        for d in rng.sample(range(800), 200):
            tree.remove(papers.pop(d), d)
        self.assertEqual(tree.size, 600)
        for lo, hi in [(1990, 2000), (2025, 2025), (1800, 1949), (1950, 2030), (2001, 1999)]:
            want = sorted((y, d) for d, y in papers.items() if lo <= y <= hi)
            self.assertEqual(tree.range(lo, hi), [d for _, d in want])
        self.assertEqual(sum(n for _, n in tree.years()), 600)
        self.assertLess(tree.height(), 30)

    def test_balanced_build(self):
        pairs = [(y, d) for d, y in enumerate(range(1900, 2027))]      # sorted input: worst case for insert()
        grown, built = YearTree(), YearTree.balanced(pairs)
        for y, d in pairs:
            grown.insert(y, d)
        self.assertEqual((grown.height(), built.height()), (127, 7))
        self.assertEqual(built.range(1950, 1960), grown.range(1950, 1960))


if __name__ == "__main__":
    unittest.main()
