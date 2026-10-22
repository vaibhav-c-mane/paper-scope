"""Merging sorted lists, and merge sort built on the same merge.

One routine, two jobs in the search engine:
* posting lists are sorted lists of doc ids, so AND / OR / NOT between query
  words are single merge passes: O(m + n) instead of O(m * n) nested loops;
* merge sort orders the result list (by score, year or citations).
  It is O(n log n) in every case, unlike quicksort's O(n^2) worst case, and it
  is stable: papers with equal keys keep their relevance order, which is what
  "sort by year" should do. Bubble, selection, insertion, heap and the other
  sorts from the syllabus are not needed.
"""


def merge(a, b, key=None, reverse=False):
    """Merge two lists that are already sorted by `key`. On equal keys the item
    from `a` comes first (stability)."""
    key = key or (lambda x: x)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        ka, kb = key(a[i]), key(b[j])
        take_a = ka >= kb if reverse else ka <= kb
        if take_a:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_sort(items, key=None, reverse=False):
    """Bottom-up merge sort: merge runs of width 1, 2, 4, ... (no recursion, so no
    depth limit). Returns a new sorted list."""
    runs = [[x] for x in items]
    while len(runs) > 1:
        runs = [merge(runs[k], runs[k + 1], key, reverse) if k + 1 < len(runs) else runs[k]
                for k in range(0, len(runs), 2)]
    return runs[0] if runs else []


# --- set operations on sorted lists of unique doc ids (boolean queries)
def intersect(a, b):
    """a AND b."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a, b):
    """a OR b: the merge step, keeping one copy of ids in both lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a, b):
    """a AND NOT b."""
    out, i, j = [], 0, 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out
