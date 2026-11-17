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
