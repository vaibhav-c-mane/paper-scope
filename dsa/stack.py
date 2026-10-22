"""Stack on an array (Python list used as a dynamic array: push and pop at the end are O(1)).

One implementation is enough: the array version needs no node objects and keeps
the items next to each other in memory, and a query never has more than a few
dozen tokens, so a fixed-size or linked version would add nothing.
"""


class StackUnderflow(IndexError):
    pass


class Stack:
    def __init__(self):
        self._items = []

    def push(self, x):
        self._items.append(x)

    def pop(self):
        if not self._items:
            raise StackUnderflow("stack underflow")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise StackUnderflow("stack empty")
        return self._items[-1]

    def is_empty(self):
        return not self._items

    def __len__(self):
        return len(self._items)

    def items(self):
        """Bottom ... top, for display."""
        return list(self._items)
