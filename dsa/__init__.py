"""The syllabus data structures the search engine actually uses, written by hand.

  stack.py       array-based stack                -> parsing AND / OR / NOT queries
  expression.py  infix -> postfix, evaluation     -> boolean queries (on the stack)
  merge.py       merge of sorted lists, merge sort -> AND / OR / NOT on posting lists,
                                                     ordering result lists
  lru.py         hash map + doubly linked list    -> bounded caches (papers in memory,
                                                     online API answers)
  heap.py        binary min-heap on an array      -> the k best candidate scores
  hashtable.py   hash table, separate chaining    -> identity keys of papers (merging duplicates)
  bst.py         binary search tree by year       -> year-range filters
  queue.py       circular queue on an array       -> the frontier of the BFS
  graph.py       breadth-first search             -> related papers through citation links

Everything else from the syllabus (other sorts, other searches, linked-list stacks,
deque, Josephus, address calculation) was left out because the search does not
need it; see docs/PROJECT.md, section 10.
"""
