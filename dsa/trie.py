from .stack import Stack


class _Node:
    __slots__ = ("edges", "values")

    def __init__(self):
        self.edges = {}                 # first character -> [label, child node]
        self.values = set()


def _common(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RadixTrie:
    def __init__(self):
        self.root = _Node()
        self.size = 0                   # (key, value) pairs stored

    def __len__(self):
        return self.size

    def insert(self, key, value):
        node, rest = self.root, key
        while rest:
            edge = node.edges.get(rest[0])
            if edge is None:                            # no edge starts with this character: new leaf
                leaf = _Node()
                node.edges[rest[0]] = [rest, leaf]
                node = leaf
                break
            label, child = edge
            k = _common(label, rest)
            if k < len(label):                          # the key leaves the edge midway: split it
                mid = _Node()
                mid.edges[label[k]] = [label[k:], child]
                edge[0], edge[1] = label[:k], mid
                child = mid
            node, rest = child, rest[k:]
        if value not in node.values:
            node.values.add(value)
            self.size += 1

    def remove(self, key, value):
        """Remove one (key, value) pair; False if it was not stored."""
        path = []                                       # (parent, first character of the edge taken)
        node, rest = self.root, key
        while rest:
            edge = node.edges.get(rest[0])
            if edge is None or not rest.startswith(edge[0]):
                return False
            path.append((node, rest[0]))
            node, rest = edge[1], rest[len(edge[0]):]
        if value not in node.values:
            return False
        node.values.discard(value)
        self.size -= 1
        # tidy up from the bottom: drop empty leaves, merge single-child nodes into their edge
        while path:
            parent, c = path.pop()
            label, child = parent.edges[c]
            if child.values:
                break
            if not child.edges:
                del parent.edges[c]
                continue
            if len(child.edges) == 1:
                (sub_label, grandchild), = child.edges.values()
                parent.edges[c] = [label + sub_label, grandchild]
            break
        return True

    def _find(self, prefix):
        """(node, the part of its edge label beyond the prefix) for the subtree of the prefix."""
        node, rest = self.root, prefix
        while rest:
            edge = node.edges.get(rest[0])
            if edge is None:
                return None, ""
            label, child = edge
            k = _common(label, rest)
            if k == len(rest):                          # the prefix ends inside (or at the end of) this edge
                return child, label[k:]
            if k < len(label):
                return None, ""
            node, rest = child, rest[k:]
        return node, ""

    def with_prefix(self, prefix, limit=None):
        """[(key, values)] of the stored keys that start with `prefix`, depth first (at most
        `limit` keys). An iterative walk with an explicit stack: no recursion limit."""
        node, tail = self._find(prefix)
        if node is None:
            return []
        out = []
        stack = Stack()
        stack.push((node, prefix + tail))
        while not stack.is_empty():
            node, key = stack.pop()
            if node.values:
                out.append((key, set(node.values)))
                if limit is not None and len(out) >= limit:
                    break
            for label, child in node.edges.values():
                stack.push((child, key + label))
        return out

    def node_count(self):
        n, stack = 0, Stack()
        stack.push(self.root)
        while not stack.is_empty():
            node = stack.pop()
            n += 1
            for _, child in node.edges.values():
                stack.push(child)
        return n
