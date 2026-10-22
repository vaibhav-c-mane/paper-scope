"""Turning text into index terms: lowercase, split, drop stop words, light stemming.

The same analyzer is used for papers and for queries, so a query word and a
paper word only need to stem to the same string to match.
"""
import re
from functools import lru_cache

# General English stop words plus words that appear in almost every abstract
# and say nothing about the topic ("we propose a novel approach ...").
STOP = set("""a about above after again against all also although among an and any are as at be
because been before being below between both but by can could did do does doing done
during each either et al etc few for from further had has have having here how however
i if in into is it its itself just less like may might more most much must no nor not
of off on once one only or other our ours out over own per same several should since so
some such than that the their them then there therefore these they this those through
thus to too under until up upon very via was we were what when where whether which
while who whom why will with within without would yet you your
paper papers propose proposed proposes present presents presented show shows shown
approach approaches method methods novel new use used uses using based result results
work study studies demonstrate demonstrates achieve achieves well many often
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")
_VOWELS = set("aeiou")


def _has_vowel(s):
    return any(ch in _VOWELS for ch in s)


def stem(w):
    """Small English stemmer (plurals, -ed, -ing, final -y), so that
    networks/network, trained/training/train and query/queries match."""
    if len(w) <= 3:
        return w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("xes", "ches", "shes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf, min_stem in (("ing", 3), ("ed", 4)):
        if w.endswith(suf) and not w.endswith("eed"):
            base = w[: -len(suf)]
            if len(base) >= min_stem and _has_vowel(base):
                w = base
                if w.endswith(("at", "bl", "iz")):          # creat-ing -> create, optimiz-ed -> optimize
                    w += "e"
                elif len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":   # embedd-ing -> embed
                    w = w[:-1]
            break
    if len(w) > 3 and w.endswith("y") and w[-2] not in _VOWELS:   # query / queries -> queri
        w = w[:-1] + "i"
    return w


@lru_cache(maxsize=300_000)
def normalize(word):
    """One raw word -> its index term, or "" if it carries no meaning (memoised: words repeat a lot)."""
    w = "".join(ch for ch in word.lower() if ch.isalnum())
    if len(w) < 2 or w in STOP or w.isdigit():
        return ""
    return stem(w)


def normalize_query_word(word):
    """Like normalize, but keeps a trailing * for prefix search (trans* -> trans*)."""
    if word.endswith("*") and len(word) > 1:
        w = "".join(ch for ch in word.lower() if ch.isalnum())
        return w + "*" if w else ""
    return normalize(word)


def terms(text):
    """Text -> list of index terms in order (duplicates kept, so counts are term frequencies)."""
    out = []
    for raw in _TOKEN.findall((text or "").lower()):
        t = normalize(raw)
        if t:
            out.append(t)
    return out


def identity(tokens):
    """Analyzer for scikit-learn vectorizers that receive already-tokenized documents."""
    return tokens
