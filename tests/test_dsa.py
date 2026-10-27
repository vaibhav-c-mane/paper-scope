"""The hand-written syllabus structures the engine uses. Run: python -m unittest -v"""
import os
import random
import tempfile
import unittest

from dsa.bst import YearTree
from dsa.graph import bfs, dijkstra
from dsa.hashtable import HashTable
from dsa.heap import MinHeap, top_k
from dsa.expression import evaluate_postfix, infix_to_postfix, tokenize_query
from dsa.logstore import LogStore
from dsa.lru import LRUCache
from dsa.merge import difference, intersect, merge, merge_sort, union
from dsa.priority_queue import IndexedMinPQ
from dsa.queue import CircularQueue, QueueUnderflow
from dsa.stack import Stack, StackUnderflow
from dsa.trie import RadixTrie


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


class PriorityQueueTests(unittest.TestCase):
    def test_pop_order_and_decrease_key(self):
        pq = IndexedMinPQ()
        for k, p in [("a", 5), ("b", 3), ("c", 8), ("d", 1)]:
            pq.push(k, p)
        pq.decrease("c", 0)
        self.assertIn("a", pq)
        self.assertFalse(pq.push_or_decrease("a", 9))                  # not lower: unchanged
        self.assertTrue(pq.push_or_decrease("a", 2))
        self.assertEqual([pq.pop() for _ in range(4)], [("c", 0), ("d", 1), ("a", 2), ("b", 3)])
        with self.assertRaises(ValueError):
            pq.push("x", 1) or pq.decrease("x", 4)
        with self.assertRaises(IndexError):
            IndexedMinPQ().pop()

    def test_matches_sorting_under_random_updates(self):
        rng = random.Random(3)
        pq, best = IndexedMinPQ(), {}
        for _ in range(2000):
            k, p = rng.randrange(300), rng.random()
            if pq.push_or_decrease(k, p):
                best[k] = p
            else:
                self.assertLessEqual(best[k], p)
        out = [pq.pop() for _ in range(len(pq))]
        self.assertEqual([p for _, p in out], sorted(best.values()))
        self.assertEqual(dict(out), best)


class DijkstraTests(unittest.TestCase):
    def test_shortest_paths_beat_fewer_hops(self):
        g = {"a": [("b", 1, "x"), ("c", 4, "y")], "b": [("c", 1, "z"), ("d", 5, "w")], "c": [("d", 1, "v")], "d": []}
        got = dijkstra("a", lambda n: g[n])
        self.assertEqual(got, [("b", 1, "a", "x"), ("c", 2, "b", "z"), ("d", 3, "c", "v")])   # c via b, not direct
        self.assertEqual([n for n, *_ in dijkstra("a", lambda n: g[n], limit=2)], ["b", "c"])
        # equal lengths: the same order as breadth-first search
        chain = {i: [(i + 1, 1, "")] + ([(i - 1, 1, "")] if i else []) for i in range(6)}
        chain[5] = [(4, 1, "")]
        self.assertEqual([n for n, *_ in dijkstra(2, lambda n: chain[n])], [n for n, _, _ in bfs(2, lambda n: [m for m, _, _ in chain[n]], max_depth=9)])


class RadixTrieTests(unittest.TestCase):
    def test_prefix_listing_and_removal(self):
        t = RadixTrie()
        for i, k in enumerate(["attention is all you need", "attention is not explanation", "attention mechanisms",
                               "graph attention networks"]):
            t.insert(k, i)
        self.assertEqual(sorted(k for k, _ in t.with_prefix("attention is")),
                         ["attention is all you need", "attention is not explanation"])
        self.assertEqual(t.with_prefix("attention mech"), [("attention mechanisms", {2})])
        self.assertEqual(t.with_prefix("xyz"), [])
        nodes = t.node_count()
        self.assertTrue(t.remove("attention is not explanation", 1))
        self.assertFalse(t.remove("attention is not explanation", 1))
        self.assertLess(t.node_count(), nodes)                          # the branch point merged away
        self.assertEqual(len(t), 3)

    def test_matches_brute_force(self):
        rng, t, ref = random.Random(1), RadixTrie(), {}
        for _ in range(4000):
            k, v = "".join(rng.choice("ab ") for _ in range(rng.randint(1, 6))), rng.randint(0, 3)
            if rng.random() < 0.6:
                t.insert(k, v)
                ref.setdefault(k, set()).add(v)
            elif t.remove(k, v):
                ref[k].discard(v)
                if not ref[k]:
                    del ref[k]
            p = "".join(rng.choice("ab ") for _ in range(rng.randint(0, 3)))
            self.assertEqual(dict(t.with_prefix(p)), {k: v for k, v in ref.items() if k.startswith(p)})
        # far fewer nodes than characters: single-child chains are merged
        self.assertLess(t.node_count(), 2 * len(ref) + 2)


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


class LogStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sub", "cache.jsonl")
        self.now = [1000.0]

    def tearDown(self):
        self.dir.cleanup()

    def store(self, **kw):
        return LogStore(self.path, clock=lambda: self.now[0], **kw)

    def test_put_get_overwrite_delete_and_reload(self):
        s = self.store()
        s.put("a", [1, 2])
        s.put("b", {"x": "y"})
        s.put("a", [3])                                     # the old line stays in the log as garbage
        s.delete("b")
        self.assertEqual((s.get("a"), s.get("b"), len(s), s.lines), ([3], None, 1, 4))
        s.close()
        again = self.store()                                # a restart rebuilds the hash map from the log
        self.assertEqual((again.get("a"), again.get("b"), len(again)), ([3], None, 1))
        again.close()

    def test_expired_records_are_gone_also_after_a_restart(self):
        s = self.store()
        s.put("short", "v", ttl=10)
        s.put("long", "v", ttl=1000)
        self.now[0] += 11
        self.assertEqual((s.get("short"), s.get("long")), (None, "v"))
        s.close()
        again = self.store()
        self.assertEqual(len(again), 1)
        again.close()

    def test_a_torn_last_line_is_cut_off(self):
        s = self.store()
        s.put("a", "kept")
        s.close()
        with open(self.path, "ab") as f:
            f.write(b'["b",null,"half writ')                # the app stopped in the middle of a write
        s = self.store()
        self.assertEqual((s.get("a"), s.get("b")), ("kept", None))
        s.put("c", "new")                                   # appends after the good part, not after the torn bytes
        s.close()
        s = self.store()
        self.assertEqual((s.get("a"), s.get("c"), s.lines), ("kept", "new", 2))
        s.close()

    def test_compaction_keeps_the_newest_records(self):
        s = self.store()
        for i in range(1500):
            s.put("k%d" % (i % 10), i)                      # 1,500 lines for 10 keys: compaction runs
        self.assertLess(s.lines, 1100)
        self.assertEqual([s.get("k%d" % i) for i in range(10)], [1490 + i for i in range(10)])
        s.compact()
        self.assertEqual((s.lines, os.path.getsize(self.path)), (10, s.size))
        s.close()

    def test_over_the_size_limit_the_oldest_records_go(self):
        s = self.store(max_bytes=2000)
        for i in range(60):
            s.put("key%02d" % i, "x" * 40)
        self.assertLessEqual(os.path.getsize(self.path), 2000)
        self.assertIsNone(s.get("key00"))
        self.assertEqual(s.get("key59"), "x" * 40)
        s.close()


if __name__ == "__main__":
    unittest.main()
