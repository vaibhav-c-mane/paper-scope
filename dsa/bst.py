"""Binary search tree keyed by publication year, for year-range filters.

Each node holds one year and the set of papers published that year; everything
in a node's left subtree is an earlier year, everything in its right subtree a
later one. There are only a few hundred distinct years, so the tree stays small
even when it indexes 20,000 papers.

* insert(year, paper): walk down from the root, left if the year is smaller,
  right if larger, and add the paper to the node of that year (a new leaf if
  the year is new): O(h) for tree height h.
* remove(year, paper): find the node, drop the paper from its set: O(h).
* range(lo, hi): in-order traversal that skips a whole subtree when it cannot
  hold years in [lo, hi] (no left subtree when the node's year < lo, no right
  subtree when > hi): O(h + years visited + papers returned), and the papers
  come out grouped by year in increasing order.

Written iteratively (an explicit stack of nodes), so a tall tree cannot hit
Python's recursion limit. Use in the search engine: the year filter
(engine/pipeline.py), with the tree kept up to date as papers are added and
evicted (engine/memindex.py).
"""


class _Node:
    __slots__ = ("year", "papers", "left", "right")

    def __init__(self, year):
        self.year, self.papers = year, set()
        self.left = self.right = None


class YearTree:
    def __init__(self):
        self.root = None
        self.size = 0                       # papers in the tree

    @classmethod
    def balanced(cls, pairs):
        """A tree of (year, paper) pairs built with the median year at the root, then the
        medians of each half, and so on: height about log2(distinct years). Used when the
        index is compacted, so a tree grown in an unlucky order is rebalanced from time to time."""
        groups = {}
        for year, paper in pairs:
            groups.setdefault(year, []).append(paper)
        years = sorted(groups)
        tree, ranges = cls(), [(0, len(years) - 1)]
        while ranges:                       # each range inserts its middle year, then splits in two
            lo, hi = ranges.pop()
            if lo > hi:
                continue
            mid = (lo + hi) // 2
            for paper in groups[years[mid]]:
                tree.insert(years[mid], paper)
            ranges += [(lo, mid - 1), (mid + 1, hi)]
        return tree

    def insert(self, year, paper):
        if self.root is None:
            self.root = _Node(year)
            node = self.root
        else:
            node = self.root
            while node.year != year:
                side = "left" if year < node.year else "right"
                child = getattr(node, side)
                if child is None:
                    child = _Node(year)
                    setattr(node, side, child)
                node = child
        if paper not in node.papers:
            node.papers.add(paper)
            self.size += 1

    def remove(self, year, paper):
        """Drop a paper; a year that becomes empty keeps its (empty) node."""
        node = self.root
        while node is not None and node.year != year:
            node = node.left if year < node.year else node.right
        if node is not None and paper in node.papers:
            node.papers.discard(paper)
            self.size -= 1

    def range(self, lo, hi):
        """Papers with lo <= year <= hi, by increasing year."""
        out, stack, node = [], [], self.root
        while stack or node is not None:
            if node is not None:
                stack.append(node)
                node = node.left if node.year > lo else None      # earlier years can only be on the left
                continue
            node = stack.pop()
            if lo <= node.year <= hi:
                out.extend(sorted(node.papers))
            node = node.right if node.year < hi else None         # later years can only be on the right
        return out

    def years(self):
        """(year, number of papers) in increasing year order (in-order traversal)."""
        out, stack, node = [], [], self.root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            if node.papers:
                out.append((node.year, len(node.papers)))
            node = node.right
        return out

    def height(self):
        best, stack = 0, [(self.root, 1)] if self.root else []
        while stack:
            node, h = stack.pop()
            best = max(best, h)
            stack += [(c, h + 1) for c in (node.left, node.right) if c is not None]
        return best
