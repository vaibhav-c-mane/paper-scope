"""Stack application: boolean search queries.

    deep AND (vision OR image) NOT survey

The query is tokenized (adjacent words get an implicit AND), converted from
infix to postfix with the shunting-yard algorithm on a stack, then evaluated
with a second stack whose operands are sorted lists of matching doc ids;
AND / OR / NOT are merge passes over those lists (dsa/merge.py).
"""
from . import merge
from .stack import Stack

PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
OPERATORS = set(PRECEDENCE)


def tokenize_query(q, normalize=lambda w: w.lower(), implicit="AND"):
    """Split a query into words, operators and brackets. Operators must be written
    in capitals (AND, OR, NOT); '-word' means NOT word. Adjacent words get the
    `implicit` operator between them."""
    raw = q.replace("(", " ( ").replace(")", " ) ").replace('"', " ").split()
    tokens = []
    for t in raw:
        if t in OPERATORS or t in ("(", ")"):
            tokens.append(t)
        elif t == "-":
            tokens.append("NOT")
        elif t.startswith("-") and len(t) > 1:
            tokens.extend(["NOT", t[1:]])
        else:
            w = normalize(t)
            if w:
                tokens.append(w)
    out = []
    for t in tokens:                       # insert the implicit operator where two operands meet
        if out:
            prev = out[-1]
            prev_ends = prev == ")" or prev not in OPERATORS and prev != "("
            cur_starts = t == "(" or t == "NOT" or (t not in OPERATORS and t != ")")
            if prev_ends and cur_starts:
                out.append("AND" if t == "NOT" else implicit)   # "a NOT b" = "a AND NOT b"
        out.append(t)
    return out


def _prec(op):
    return 4 if op == "NOT1" else PRECEDENCE[op]


def infix_to_postfix(tokens):
    """Shunting-yard. Returns (postfix, steps); steps record the stack after each
    token, for the search trace in the UI. A NOT with no left operand is unary
    (written NOT1: everything except)."""
    stack, out, steps = Stack(), [], []
    for i, t in enumerate(tokens):
        if t == "(":
            stack.push(t)
        elif t == ")":
            while not stack.is_empty() and stack.peek() != "(":
                out.append(stack.pop())
            if stack.is_empty():
                raise ValueError("unbalanced parentheses")
            stack.pop()
        elif t in OPERATORS:
            unary = t == "NOT" and (i == 0 or tokens[i - 1] in OPERATORS or tokens[i - 1] == "(")
            tok, prec = ("NOT1", 4) if unary else (t, PRECEDENCE[t])
            while not unary and not stack.is_empty() and stack.peek() != "(" and _prec(stack.peek()) >= prec:
                out.append(stack.pop())
            stack.push(tok)
        else:
            out.append(t)
        steps.append({"token": t, "stack": stack.items(), "output": list(out)})
    while not stack.is_empty():
        top = stack.pop()
        if top == "(":
            raise ValueError("unbalanced parentheses")
        out.append(top)
    steps.append({"token": "end", "stack": [], "output": list(out)})
    return out, steps


def evaluate_postfix(postfix, lookup, universe):
    """`lookup(word)` -> sorted list of doc ids; `universe` = all doc ids (for unary NOT)."""
    stack = Stack()
    for t in postfix:
        if t == "NOT1":
            stack.push(merge.difference(universe, stack.pop()))
        elif t in OPERATORS:
            if len(stack) < 2:
                raise ValueError("malformed query near " + t)
            b, a = stack.pop(), stack.pop()
            op = {"AND": merge.intersect, "OR": merge.union, "NOT": merge.difference}[t]
            stack.push(op(a, b))
        else:
            stack.push(lookup(t))
    if len(stack) != 1:
        raise ValueError("malformed query")
    return stack.pop()
