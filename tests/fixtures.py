"""Test fixtures: a small, fully controlled corpus for checking the code's mechanics, and a
fake HTTP transport so no test touches the network.

The corpus below only exercises the code (indexing, scoring, eviction, the
online flow). The models in data/models are trained and evaluated on real
OpenAlex papers with citation labels (ml/train.py), never on this.
"""
import json
import random
import socket

TOPICS = {
    "cs.SI": ["graph", "node", "embedding", "network", "walk", "community", "link", "vertex", "neighbourhood"],
    "cs.CV": ["image", "pixel", "segmentation", "detection", "camera", "object", "convolutional", "visual", "scene"],
    "quant-ph": ["quantum", "qubit", "entanglement", "circuit", "superconducting", "photon", "decoherence", "gate"],
    "q-bio.BM": ["protein", "folding", "structure", "amino", "residue", "enzyme", "binding", "molecular", "sequence"],
}
FILLER = ["method", "results", "model", "learning", "data", "efficient", "large", "scale", "evaluation", "framework"]


def make_corpus(per_topic=20, seed=3):
    rng = random.Random(seed)
    papers, n = [], 0
    for cat, words in TOPICS.items():
        for i in range(per_topic):
            title = " ".join(rng.sample(words, 3)).title() + f" {n}"      # number keeps titles unique
            abstract = " ".join(rng.choice(words) if rng.random() < 0.6 else rng.choice(FILLER) for _ in range(30))
            papers.append({
                "arxiv_id": f"2101.{n:05d}", "title": title, "abstract": abstract.capitalize() + ".",
                "authors": f"Author{n % 7} Lastname{n % 5}, Coauthor{n % 3}", "categories": cat,
                "year": 2010 + n % 14, "venue": "Journal of Tests" if n % 4 == 0 else "",
                "citations": (n * 7) % 90, "published": n % 4 == 0, "source": "test",
            })
            n += 1
    return papers


_LEXICON = None


def lexicon():
    """A Lexicon learned from the fixture corpus (built once per test run)."""
    global _LEXICON
    if _LEXICON is None:
        from engine.lexicon import Lexicon
        _LEXICON = Lexicon.build(make_corpus(per_topic=40, seed=11), dim=16, progress=None)
    return _LEXICON


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.t += s


class FakeTransport:
    """Answers HTTP calls from a routing table: {url substring: handler}. A handler
    returns (status, headers, body-dict) or raises. Unrouted URLs get a 404."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls = []

    def __call__(self, method, url, body, headers, timeout):
        self.calls.append({"method": method, "url": url, "body": json.loads(body) if body else None})
        for part, handler in self.routes.items():
            if part in url:
                status, hdrs, data = handler(method, url, body)
                return status, hdrs, data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
        return 404, {}, b""

    def urls(self, part=""):
        return [c["url"] for c in self.calls if part in c["url"]]


def ok(data):
    return lambda method, url, body: (200, {}, data)


def timeout(method, url, body):
    raise socket.timeout("timed out")


def client_with(routes, order=("openalex", "semanticscholar", "crossref"), **kw):
    from engine.sources import Http, ScholarlyClient
    clock = FakeClock()
    transport = FakeTransport(routes)
    http = Http(transport=transport, timeout=2, retries=1, sleep=clock.sleep, clock=clock)
    kw.setdefault("field_sources", ())         # the free field sources only where a test asks for them
    kw.setdefault("cache_file", None)          # never the cache file from a developer's .env
    return ScholarlyClient(order=order, http=http, **kw), transport, clock


def openalex_work(title, doi=None, year=2021, cites=12, abstract="protein folding structure of amino residue chains",
                  venue=None, listed_in=None, work_type="article", refs=None,
                  topic=("Protein Structure", "Molecular Biology", "Biochemistry, Genetics and Molecular Biology")):
    n = abs(hash(title)) % 10 ** 8
    src = {"display_name": venue, "type": "journal", "listed_in": listed_in or []} if venue else None
    w = {"id": f"https://openalex.org/W{n}", "title": title, "type": work_type,
         "publication_year": year, "cited_by_count": cites,
         "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
         "abstract_inverted_index": {w: [i] for i, w in enumerate(abstract.split())},
         "primary_location": {"landing_page_url": "https://example.org/" + title.replace(" ", "-"), "source": src,
                              "is_published": bool(venue)},
         "primary_topic": {"display_name": topic[0], "subfield": {"display_name": topic[1]},
                           "field": {"display_name": topic[2]}} if topic else None,
         "open_access": {"oa_url": None}}
    if doi:
        w["doi"] = "https://doi.org/" + doi
    if refs is not None:
        w["referenced_works"] = [f"https://openalex.org/W{r}" for r in refs]
    return w


def s2_paper(title, arxiv=None, doi=None, cites=5, abstract="protein structure folding prediction", venue="Nature"):
    return {"title": title, "abstract": abstract, "year": 2020, "citationCount": cites, "venue": venue,
            "authors": [{"name": "Grace Hopper"}], "externalIds": {"ArXiv": arxiv, "DOI": doi}, "url": "https://s2/x"}
