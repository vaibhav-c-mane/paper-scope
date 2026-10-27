"""Query understanding: what the user typed -> what the search engine should look for.

People type research questions the way they would ask a librarian:

    "I want research papers on how to make neural networks smaller so they run on phones"
    "nueral netwrok pruning"
    "recent work on diffusion models for image generation"

Searched word for word, the first one matches "want", "research", "make" and
"run" as if they were the topic, the second matches nothing (two typos), and
the third treats "recent" as a topic word. This module turns each query into:

* `query`: the topic words, spelling corrected, without the request around them
  ("make neural networks smaller so they run on phones", "neural network
  pruning", "diffusion models for image generation");
* `recent_from`: a year filter when the user asked for recent work;
* `semantic`: whether the query is written as a sentence or question, so the
  search should also ask OpenAlex's semantic (meaning-based) search, which
  understands paraphrases that share no words with the papers;
* `title`: whether the query could be the title of a paper, so the search also
  looks for papers with those words in their title, most cited first (an exact
  title such as "attention is all you need" is otherwise buried under the many
  papers that reuse it).

Spelling correction is a small statistical model learned from the 600,013 real
papers behind the Lexicon (engine/lexicon.py), the noisy-channel corrector of
Norvig's "How to write a spelling corrector":
  * language model: how many papers use a word (its document frequency);
  * error model: a typo is one edit away (a letter deleted, inserted, replaced,
    or two neighbours swapped) more often than two edits away.
A word is replaced only when it is rare in real papers (fewer than 10 use it)
and a candidate one edit away (or, for a word no paper uses, two) is used
by at least 50 times as many papers. Two edits are tried only for words of 5 to 9
letters that no paper uses: they mean about 100,000 candidates (0.1 to 0.4 s). Candidates are checked with the Lexicon's hash map, O(1)
each: about 54n + 25 candidates for a word of n letters at one edit.
The rules for requests ("find me papers about ..."), quality words ("best",
"top"), and recency ("recent", "latest") are short word lists: written by hand,
shown in the search trace, and undone by searching with exact=True.
"""
import re

from engine.text import normalize

_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Words that ask for papers rather than describe the topic.
REQUEST = set("""please find show give list recommend suggest search searching searched look looking
want wants wanted need needs needed help tell explain get fetch read me us i we you my our
would could can""".split())
QUALITY = set("""good best top great important relevant interesting influential key seminal famous
popular useful notable major main classic""".split())
DOCUMENT = set("""paper papers article articles publication publications literature""".split())
# Words that are a request only next to a preposition or another request word:
# "research on X", "studies showing X", but not "operations research".
CONTEXTUAL = set("""research work works study studies results findings advances developments progress
trends""".split())
RECENT = set("""recent recently latest newest current""".split())
_LINKS = set("""on about into regarding concerning related relating showing that which where
for in of to around discussing describing explaining covering""".split())
_FUNCTION = _LINKS | {"some", "any", "a", "an", "the", "few", "all", "other", "more"}
_ANCHOR = DOCUMENT | CONTEXTUAL | {"me", "us", "i", "we", "you", "please"}     # these show a request is made
_QUESTION = set("""how what why which when where who whom does do did is are was were can could
should will would""".split())

RARE_DF = 10            # a word used by fewer papers than this may be a typo
GAIN = 50               # ... and is corrected only to a word used by 50x as many papers
RECENT_YEARS = 5        # "recent" = published in the last 5 calendar years
MIN_DOCS = 10_000       # spelling is corrected only with a corpus at least this large


def _edits1(w):
    """Every string one edit away from w (deletes, swaps of neighbours, replaces, inserts)."""
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in _LETTERS}
    out |= {a + c + b for a, b in splits for c in _LETTERS}
    out.discard(w)
    return out


class QueryUnderstanding:
    def __init__(self, lexicon):
        self.lex = lexicon

    # ---------------------------------------------------------------- spelling
    def df(self, word):
        """How many papers of the background corpus use this word (0 if none)."""
        t = normalize(word)
        if not t:
            return -1                       # a stop word: never "corrected"
        i = self.lex.term_id(t)
        return int(self.lex.df[i]) if i >= 0 else 0

    def correct(self, word):
        """The intended word if `word` looks like a typo, else None."""
        w = word.lower()
        if len(w) < 4 or not w.isalpha() or (word.isupper() and len(word) <= 6):    # acronyms: GNN, LLM
            return None
        if self.lex.n_docs < MIN_DOCS:      # too few papers to tell a typo from a rare word
            return None
        own = self.df(w)
        if own < 0 or own >= RARE_DF:
            return None
        need = max(GAIN * max(own, 1), 100)
        for candidates in (lambda: _edits1(w), lambda: self._edits2(w) if own == 0 else ()):
            best, best_df = None, need - 1
            for c in candidates():
                d = self.df(c)
                if d > best_df or (d == best_df and best is not None and c < best):
                    best, best_df = c, d
            if best is not None:
                return best
        return None

    def _edits2(self, w):
        if not 5 <= len(w) <= 9:            # two edits of a short word reach almost any short word
            return ()
        return (e2 for e1 in _edits1(w) for e2 in _edits1(e1))

    # ---------------------------------------------------------------- the whole query
    def understand(self, raw, now_year):
        """See the module docstring. `raw` is the query box text without operators or ids."""
        tokens = raw.split()
        bare = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in tokens]
        out = {"original": " ".join(tokens), "corrections": [], "removed": [], "recent_from": None,
               "kind": "keywords", "semantic": False, "title": False}
        if bare and (bare[0] in _QUESTION or tokens[-1].endswith("?")):
            out["kind"] = "question"

        # 1. spelling, word by word (exclusions, prefixes and quoted words are left alone)
        words = []
        for t, b in zip(tokens, bare):
            fixed = None
            if t.isalpha():
                fixed = self.correct(t)
            if fixed:
                out["corrections"].append({"from": t, "to": fixed})
                words.append(fixed)
            else:
                words.append(t)
        bare = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in words]

        # 2. what is a request, not the topic. Request words count only where requests are
        #    made: at the start ("can you find me some recent papers on ..."), or as a pair
        #    such as "best papers" / "latest research", or a trailing "papers". Elsewhere they
        #    can be the topic: "attention is all you need", "best response dynamics", "electric current".
        n = len(words)
        drop = [False] * n

        def asks(j):                        # could word j be part of a request, given the word after it?
            b, nxt = bare[j], (bare[j + 1] if j + 1 < n else "")
            return (b in REQUEST | QUALITY | DOCUMENT | RECENT | _FUNCTION
                    or (b == "new" and nxt in DOCUMENT | CONTEXTUAL)
                    or (b in CONTEXTUAL and (nxt in _LINKS | DOCUMENT | CONTEXTUAL or not nxt)))

        i = 0                               # the run of request-like words at the start ...
        if not (bare and bare[0] in _QUESTION - {"can", "could", "would"}):
            while i < n and asks(i):
                i += 1
        if i >= 2 and any(bare[j] in _ANCHOR for j in range(i)):   # ... is a request if it asks for something
            drop[:i] = [True] * i
        for j in range(i, n):
            b, nxt = bare[j], (bare[j + 1] if j + 1 < n else "")
            if b in QUALITY | RECENT | {"new"} and nxt in DOCUMENT | CONTEXTUAL:
                k = j + 2                   # "the latest papers on X": the pair and the words linking it to X
                while k < n and bare[k] in _LINKS:
                    k += 1
                drop[j:k] = [True] * (k - j)
            elif b in DOCUMENT and (not nxt or nxt in _LINKS) and j:
                drop[j] = True
        for j in range(n):
            if drop[j] and (bare[j] in RECENT or (bare[j] == "new" and j + 1 < n and bare[j + 1] in DOCUMENT | CONTEXTUAL)):
                out["recent_from"] = now_year - RECENT_YEARS + 1
        keep = [w for j, w in enumerate(words) if not drop[j]]
        removed = [w for j, w in enumerate(words) if drop[j]]
        if any(normalize(w) for w in keep):
            if any(bare[j] in REQUEST | QUALITY | DOCUMENT | RECENT for j in range(n) if drop[j]):
                out["kind"] = "request" if out["kind"] == "keywords" else out["kind"]
            out["removed"] = removed
        else:                                # nothing but request words: search what was typed
            keep = words
            out["recent_from"] = None
        out["query"] = " ".join(keep)

        # 3. how to search online
        content = [w for w in keep if normalize(w)]
        out["semantic"] = out["kind"] != "keywords" or len(content) >= 5 or bool(out["corrections"])
        out["title"] = out["kind"] == "keywords" and len(keep) >= 3
        out["changed"] = out["query"].lower() != out["original"].lower() or out["recent_from"] is not None
        return out
