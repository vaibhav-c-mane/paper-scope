"""Online scholarly sources: OpenAlex (first), Semantic Scholar, Crossref.

This is where the papers come from. Nothing is stored on disk: answers are kept
in an in-memory LRU cache with an expiry time (dsa/lru.py), so a repeated query
within the session costs no network call, and the cache never grows past a
fixed number of entries.

What makes it fast and safe to use:
* Provider cascade. OpenAlex first (largest open index, generous limits, venue
  rankings and topics included); only if it fails or finds too little,
  Semantic Scholar, then Crossref.
* Rate limits. One token bucket per provider (OpenAlex 8/s, Semantic Scholar 1/s,
  Crossref 3/s), Retry-After is honoured on HTTP 429, and the polite-pool email
  (OPENALEX_EMAIL) is sent when set.
* Timeouts and a deadline. Each request has a timeout and every search has an
  overall deadline, so a slow API delays a search by a few seconds at most;
  retries use exponential backoff and stop when the deadline is near.
* Cool-down. A provider that just failed (network down, blocked, HTTP 5xx) is
  skipped for a minute, so an outage does not add a timeout to every search.
* Batching. Looking papers up by id asks for 50 per OpenAlex request
  (filter=ids.openalex:W1|W2|...), not one request per paper.
* Missing fields. Every provider's answer is normalised into the same paper
  dict (engine/index.py); records without a title are dropped.
"""
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from dsa.lru import LRUCache
from engine.papers import complete, paper_from_openalex, paper_keys

HOUR, DAY = 3600.0, 86400.0
_MISS = object()


def load_env_file(path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")):
    """Read KEY=value lines from paper-search/.env (e.g. OPENALEX_API_KEY=...) so the key does not
    have to be typed in every new terminal. Variables already set in the terminal win."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


load_env_file()


# ------------------------------------------------------------------ plumbing
class RateLimiter:
    """Token bucket: on average `rate` requests per second, at most `burst` at once.
    Callers reserve a slot and sleep until it is theirs, so concurrent threads queue fairly."""

    def __init__(self, rate, burst=1, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.clock = clock
        self.sleep = sleep
        self.last = clock()
        self.lock = threading.Lock()

    def acquire(self, deadline=None):
        with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            self.tokens -= 1.0
            wait = max(0.0, -self.tokens / self.rate)
            if deadline is not None and now + wait > deadline:
                self.tokens += 1.0              # give the slot back
                return False
        if wait:
            self.sleep(wait)
        return True


TRANSIENT = (429, 500, 502, 503, 504)


def urllib_transport(method, url, body, headers, timeout):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        return e.code, dict(e.headers or {}), raw


class RateLimited(RuntimeError):
    """The API said 'too many requests' and asked us to wait longer than a search can.
    `wait` is how long it asked for, so the client can skip that provider until then."""

    def __init__(self, host, wait, detail=""):
        super().__init__(f"HTTP 429 from {host}: rate limit, retry in {_human(wait)}" + (f" ({detail})" if detail else ""))
        self.wait = wait


def _human(seconds):
    """3600 -> 'about 1 h', 90 -> 'about 2 min', 20 -> '20 s'."""
    if seconds >= 5400:
        return f"about {seconds / 3600:.0f} h"
    if seconds >= 90:
        return f"about {seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _error_detail(raw):
    """A short human-readable reason from an API's JSON error body, if it has one."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return ""
    msg = data.get("message") or data.get("error") or "" if isinstance(data, dict) else ""
    return " ".join(str(msg).split())[:200]


class Http:
    """JSON over HTTP with timeouts, retries with backoff, and Retry-After support.
    `transport` is swappable so tests never touch the network."""

    def __init__(self, transport=urllib_transport, timeout=6.0, retries=2, sleep=time.sleep, clock=time.monotonic):
        self.transport = transport
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.clock = clock
        self.calls = 0

    def json(self, method, url, payload=None, headers=None, limiter=None, deadline=None):
        email = os.environ.get("OPENALEX_EMAIL", "")
        hdrs = {"User-Agent": f"PaperScope/3.1 (student project{'; mailto:' + email if email else ''})",
                "Accept": "application/json"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        last_error = None
        for attempt in range(self.retries + 1):
            if limiter is not None and not limiter.acquire(deadline):
                raise TimeoutError("deadline reached while waiting for the rate limit")
            timeout = self.timeout if deadline is None else min(self.timeout, deadline - self.clock())
            if timeout <= 0.05:
                raise TimeoutError("search deadline reached")
            self.calls += 1
            try:
                status, rh, raw = self.transport(method, url, body, hdrs, timeout)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                status, rh, raw, last_error = None, {}, b"", e
            if status is not None and 200 <= status < 300:
                return json.loads(raw.decode("utf-8")) if raw else None
            if status == 404:
                return None
            if status is not None and status not in TRANSIENT:
                raise RuntimeError(f"HTTP {status} from {urllib.parse.urlsplit(url).netloc}")
            last_error = RuntimeError(f"HTTP {status} from {urllib.parse.urlsplit(url).netloc}") if status else last_error
            wait = 0.5 * (2 ** attempt)
            retry_after = {k.lower(): v for k, v in rh.items()}.get("retry-after")
            if retry_after and str(retry_after).strip().isdigit():
                asked = float(retry_after)
                if asked > 5.0:                 # e.g. a daily budget used up: waiting cannot help this search
                    raise RateLimited(urllib.parse.urlsplit(url).netloc, asked, _error_detail(raw))
                wait = asked
            if attempt == self.retries or (status == 429 and attempt >= 1):   # still "too many requests": give up
                break
            if deadline is not None and self.clock() + wait > deadline:
                break
            self.sleep(wait)
        raise last_error if isinstance(last_error, Exception) else RuntimeError("request failed")


class _Locked:
    """An LRU cache shared by the searches that run at the same time (keyword, by meaning,
    by title): get and put move nodes of its linked list, so one runs at a time."""

    def __init__(self, cache):
        self._cache, self._lock = cache, threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._cache.get(key, default)

    def put(self, key, value, ttl=None):
        with self._lock:
            return self._cache.put(key, value, ttl)

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def __len__(self):
        return len(self._cache)


# ------------------------------------------------------------------ helpers
_TAGS = re.compile(r"<[^>]+>")
_OPENALEX_ID = re.compile(r"(?:https?://openalex\.org/)?(W\d+)$", re.I)
_ARXIV_ID = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)


def arxiv_doi(arxiv_id):
    return ("10.48550/arxiv." + re.sub(r"v\d+$", "", arxiv_id)).lower()


def _clean(s):
    return " ".join(_TAGS.sub(" ", s or "").split())


def api_query(query):
    """Our query syntax -> what the APIs accept: '-word' becomes 'NOT word' and the
    prefix star is dropped (the APIs stem words themselves)."""
    out = []
    for t in query.replace("*", " ").split():
        if t.startswith("-") and len(t) > 1:
            out += ["NOT", t[1:]]
        else:
            out.append(t)
    return " ".join(out)


def openalex_number(work_id):
    """'https://openalex.org/W2741809807' or 'W2741809807' -> 2741809807 (None if not an id)."""
    m = _OPENALEX_ID.search(str(work_id or "").strip())
    return int(m.group(1)[1:]) if m else None


# ------------------------------------------------------------------ providers
class OpenAlex:
    name = "OpenAlex"
    BASE = "https://api.openalex.org"
    SELECT = ("id,doi,title,display_name,publication_year,cited_by_count,authorships,primary_location,"
              "abstract_inverted_index,open_access,primary_topic,type,language")
    BATCH = 50

    def __init__(self, http):
        self.http = http
        self.limiter = RateLimiter(8, burst=4, clock=http.clock, sleep=http.sleep)

    def _url(self, params, refs=False):
        params = dict(params, select=self.SELECT + (",referenced_works" if refs else ""))
        if os.environ.get("OPENALEX_EMAIL"):
            params["mailto"] = os.environ["OPENALEX_EMAIL"]
        if os.environ.get("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        return f"{self.BASE}/works?" + urllib.parse.urlencode(params)

    def normalize(self, w, refs=False):
        p = paper_from_openalex(w, source=self.name)
        if p is not None and refs:
            p["refs"] = [n for n in (openalex_number(r) for r in w.get("referenced_works") or []) if n]
        return p

    def _get(self, params, refs=False, deadline=None):
        data = self.http.json("GET", self._url(params, refs), limiter=self.limiter, deadline=deadline) or {}
        return [p for p in (self.normalize(w, refs) for w in data.get("results", [])) if p], data.get("meta") or {}

    def search(self, query, limit, deadline=None, year_from=None, year_to=None):
        params = {"search": api_query(query), "per_page": min(200, limit)}
        if year_from or year_to:
            params["filter"] = f"publication_year:{year_from or 1900}-{year_to or 2100}"
        return self._get(params, refs=True, deadline=deadline)[0]     # reference lists: the citation graph

    def semantic(self, query, limit, deadline=None, year_from=None, year_to=None):
        """Meaning-based search: OpenAlex embeds the query and returns the works whose
        embeddings are closest, so "make neural networks smaller for phones" finds model
        compression papers that share none of those words. At most 50 results."""
        params = {"search.semantic": " ".join(query.split()), "per_page": min(50, limit)}
        if year_from or year_to:
            params["filter"] = f"publication_year:{year_from or 1900}-{year_to or 2100}"
        return self._get(params, refs=True, deadline=deadline)[0]

    def titles(self, query, limit, deadline=None, year_from=None, year_to=None):
        """Works whose title contains the query words, most cited first: finds the original
        paper behind a title such as "attention is all you need"."""
        words = re.sub(r"[^\w\s-]", " ", query).split()       # , : | would start a new filter
        flt = "title.search:" + " ".join(words)
        if year_from or year_to:
            flt += f",publication_year:{year_from or 1900}-{year_to or 2100}"
        return self._get({"filter": flt, "sort": "cited_by_count:desc", "per_page": min(25, limit)},
                         refs=True, deadline=deadline)[0]

    def lookup(self, dois, deadline=None):
        """DOIs -> {doi: paper}; one request per 50 DOIs."""
        out = {}
        for i in range(0, len(dois), self.BATCH):
            chunk = dois[i:i + self.BATCH]
            got = self._get({"filter": "doi:" + "|".join(chunk), "per_page": len(chunk)}, True, deadline)[0]
            for p in got:
                if p.get("doi"):
                    out[p["doi"].lower()] = p
            # a work can be found by a DOI that is not its main one (arXiv DOIs often are not)
            if len(chunk) == 1 and len(got) == 1 and chunk[0] not in out:
                out[chunk[0]] = got[0]
        return out

    # ---- used by ml/dataset.py to build real relevance labels online
    def by_ids(self, numbers, refs=False, deadline=None):
        """OpenAlex work numbers -> papers; one request per 50 ids."""
        out = []
        for i in range(0, len(numbers), self.BATCH):
            chunk = numbers[i:i + self.BATCH]
            ids = "|".join(f"W{n}" for n in chunk)
            out += self._get({"filter": "ids.openalex:" + ids, "per_page": len(chunk)}, refs, deadline)[0]
        return out

    def sample(self, n, seed=0, filters="has_abstract:true,referenced_works_count:>9", refs=True):
        """A reproducible random sample of works (with their reference lists)."""
        out, page = [], 1
        while len(out) < n:
            per = min(200, n - len(out))
            got, _ = self._get({"sample": n, "seed": seed, "filter": filters, "per_page": per, "page": page}, refs)
            if not got:
                break
            out += got
            page += 1
        return out[:n]

    def iter_works(self, filters="has_abstract:true", limit=100000):
        """Stream works with cursor paging (200 per request), e.g. to learn the Lexicon online."""
        cursor, n = "*", 0
        while cursor and n < limit:
            got, meta = self._get({"filter": filters, "per_page": 200, "cursor": cursor})
            if not got:
                return
            for p in got[:limit - n]:
                yield p
            n += len(got)
            cursor = meta.get("next_cursor")


class SemanticScholar:
    name = "Semantic Scholar"
    BASE = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "title,abstract,authors,year,citationCount,externalIds,url,openAccessPdf,venue,fieldsOfStudy,publicationTypes"
    BATCH = 500

    def __init__(self, http):
        self.http = http
        self.limiter = RateLimiter(1, burst=1, clock=http.clock, sleep=http.sleep)

    def _headers(self):
        key = os.environ.get("S2_API_KEY")
        return {"x-api-key": key} if key else {}

    def search(self, query, limit, deadline=None, year_from=None, year_to=None):
        params = {"query": api_query(query), "limit": min(100, limit), "fields": self.FIELDS}
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        data = self.http.json("GET", f"{self.BASE}/paper/search?" + urllib.parse.urlencode(params),
                              headers=self._headers(), limiter=self.limiter, deadline=deadline)
        return [p for p in (self.normalize(x) for x in (data or {}).get("data", []) or []) if p]

    def lookup(self, ids, deadline=None):
        """ids like 'ARXIV:1706.03762' or 'DOI:10.1/x' -> {id: paper}; one request per 500."""
        out = {}
        for i in range(0, len(ids), self.BATCH):
            chunk = ids[i:i + self.BATCH]
            data = self.http.json("POST", f"{self.BASE}/paper/batch?fields={self.FIELDS}", payload={"ids": chunk},
                                  headers=self._headers(), limiter=self.limiter, deadline=deadline) or []
            for sid, rec in zip(chunk, data):
                p = self.normalize(rec) if rec else None
                if p:
                    out[sid] = p
        return out

    def normalize(self, r):
        title = " ".join((r.get("title") or "").split())
        if not title:
            return None
        ext = r.get("externalIds") or {}
        venue = (r.get("venue") or "").strip()
        preprint_only = not venue or "arxiv" in venue.lower() or "rxiv" in venue.lower()
        return complete({
            "arxiv_id": ext.get("ArXiv"), "title": title,
            "authors": ", ".join(a.get("name", "") for a in (r.get("authors") or []) if a.get("name")),
            "abstract": r.get("abstract") or "",
            "categories": "; ".join(r.get("fieldsOfStudy") or []),
            "year": r.get("year") or 0,
            "venue": "" if preprint_only else venue,
            "published": None if not venue else not preprint_only,
            "doi": (ext.get("DOI") or "").lower() or None, "url": r.get("url"),
            "oa_url": (r.get("openAccessPdf") or {}).get("url"),
            "citations": r.get("citationCount"), "source": self.name,
        })


class Crossref:
    name = "Crossref"
    BASE = "https://api.crossref.org"
    SELECT = "DOI,title,author,container-title,issued,created,is-referenced-by-count,abstract,URL,type"
    PUBLISHED_TYPES = {"journal-article", "proceedings-article", "book-chapter", "monograph", "book"}
    # Crossref also registers figures, peer reviews, grants and whole journal issues; none is a paper
    NOT_PAPERS = {"component", "peer-review", "grant", "journal", "journal-issue", "journal-volume",
                  "proceedings", "proceedings-series", "book-series", "book-set", "report-series", "database"}

    def __init__(self, http):
        self.http = http
        self.limiter = RateLimiter(3, burst=1, clock=http.clock, sleep=http.sleep)

    def search(self, query, limit, deadline=None, year_from=None, year_to=None):
        params = {"query.bibliographic": api_query(query), "rows": min(100, limit), "select": self.SELECT}
        dates = [f"from-pub-date:{year_from}"] * bool(year_from) + [f"until-pub-date:{year_to}"] * bool(year_to)
        if dates:
            params["filter"] = ",".join(dates)
        if os.environ.get("OPENALEX_EMAIL"):
            params["mailto"] = os.environ["OPENALEX_EMAIL"]
        data = self.http.json("GET", f"{self.BASE}/works?" + urllib.parse.urlencode(params),
                              limiter=self.limiter, deadline=deadline)
        items = ((data or {}).get("message") or {}).get("items", [])
        return [p for p in (self.normalize(x) for x in items) if p]

    def lookup(self, doi, deadline=None):
        """One DOI -> paper (or None). Crossref knows journal and conference DOIs, not arXiv ones."""
        params = {"mailto": os.environ["OPENALEX_EMAIL"]} if os.environ.get("OPENALEX_EMAIL") else {}
        url = f"{self.BASE}/works/{urllib.parse.quote(doi, safe='/')}" + ("?" + urllib.parse.urlencode(params) if params else "")
        data = self.http.json("GET", url, limiter=self.limiter, deadline=deadline)
        return self.normalize((data or {}).get("message") or {})

    def normalize(self, r):
        title = " ".join(" ".join(r.get("title") or []).split())
        if not title or r.get("type") in self.NOT_PAPERS:
            return None
        parts = ((r.get("issued") or {}).get("date-parts") or [[None]])[0]
        if not (parts and parts[0]):                    # no publication date: fall back to when the DOI was registered
            parts = ((r.get("created") or {}).get("date-parts") or [[None]])[0]
        kind = r.get("type") or ""
        return complete({
            "title": title,
            "authors": ", ".join(" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in (r.get("author") or [])),
            "abstract": _clean(r.get("abstract")),
            "year": parts[0] if parts and parts[0] else 0,
            "venue": " ".join(r.get("container-title") or []),
            "published": True if kind in self.PUBLISHED_TYPES else False if kind == "posted-content" else None,
            "doi": (r.get("DOI") or "").lower() or None, "url": r.get("URL"),
            "citations": r.get("is-referenced-by-count"), "source": self.name,
        })


PROVIDERS = {"openalex": OpenAlex, "semanticscholar": SemanticScholar, "crossref": Crossref}


# ------------------------------------------------------------------ client
class ScholarlyClient:
    """The search engine's only way to get papers. `cache_size` bounds the number of
    remembered answers (searches and single papers); `*_ttl` say how long an answer
    stays fresh. "Not found" answers are cached too, for a shorter time."""
    COOLDOWN = 60.0
    RESERVE = 1.5          # seconds kept back for each later provider in the cascade

    def __init__(self, order=("openalex", "semanticscholar", "crossref"), http=None, cache_size=2000,
                 search_ttl=DAY, paper_ttl=7 * DAY, negative_ttl=HOUR):
        self.http = http or Http()
        self.cache = _Locked(LRUCache(cache_size, ttl=search_ttl, clock=self.http.clock))
        self.providers = {name: PROVIDERS[name](self.http) for name in order}
        self.order = list(order)
        self.search_ttl, self.paper_ttl, self.negative_ttl = search_ttl, paper_ttl, negative_ttl
        self.down_until = {}                 # provider -> clock time until which it is skipped

    def _wait(self, name):
        return max(0.0, self.down_until.get(name, 0.0) - self.http.clock())

    def _up(self, name):
        return self.http.clock() >= self.down_until.get(name, 0.0)

    def _failed(self, name, error=None):
        """Skip a failed provider for a minute, or for as long as its rate limit asked (at most 6 hours)."""
        pause = self.COOLDOWN
        if isinstance(error, RateLimited):
            pause = min(max(error.wait, self.COOLDOWN), 6 * HOUR)
        self.down_until[name] = self.http.clock() + pause

    # ---- search
    def search(self, query, limit=50, deadline_s=6.0, year_from=None, year_to=None):
        """Ask providers in order until one gives enough results. Returns (papers, report)."""
        deadline = self.http.clock() + deadline_s
        query = " ".join(query.split())
        papers, seen, report = [], set(), []
        for i, name in enumerate(self.order):
            prov = self.providers[name]
            # keep a little time for every provider after this one, so one slow API cannot use the whole deadline
            own_deadline = deadline - self.RESERVE * (len(self.order) - 1 - i)
            own_deadline = max(own_deadline, min(deadline, self.http.clock() + self.RESERVE))
            key = ("search", name, limit, query.lower(), year_from, year_to)
            t0 = time.perf_counter()
            cached = self.cache.get(key, _MISS)
            hit = cached is not _MISS
            entry = {"provider": prov.name, "cached": hit}
            if not hit and not self._up(name):
                entry.update(ok=False, results=0, error=f"skipped: failed recently, retrying in {_human(self._wait(name))}", ms=0)
                report.append(entry)
                continue
            try:
                found = cached if hit else prov.search(query, limit, own_deadline, year_from, year_to)
                if not hit:
                    self.cache.put(key, found, self.search_ttl if found else self.negative_ttl)
                entry.update(ok=True, results=len(found))
            except Exception as e:                      # timeout, HTTP error, bad JSON: try the next provider
                found = []
                self._failed(name, e)
                entry.update(ok=False, results=0, error=f"{type(e).__name__}: {e}"[:160])
            entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            report.append(entry)
            for p in found:
                ks = paper_keys(p)
                if not any(k in seen for k in ks):
                    seen.update(ks)
                    papers.append(dict(p))
            if len(papers) >= max(3, limit // 2) or self.http.clock() >= deadline:
                break
        return papers[:limit], report

    def search_openalex(self, kind, query, limit=25, deadline_s=6.0, year_from=None, year_to=None):
        """An extra OpenAlex search: kind "semantic" (by meaning) or "titles" (by title, most
        cited first). Cached like search(). A failure here is reported but does not pause the
        keyword search, which counts as a separate provider ("openalex-semantic", ...)."""
        name = "openalex-" + kind
        label = {"semantic": "OpenAlex (by meaning)", "titles": "OpenAlex (by title)"}[kind]
        entry = {"provider": label, "cached": False}
        prov = self.providers.get("openalex")
        if prov is None:
            return [], dict(entry, ok=False, results=0, error="OpenAlex not configured", ms=0)
        key = ("search", name, limit, " ".join(query.lower().split()), year_from, year_to)
        t0 = time.perf_counter()
        found = self.cache.get(key, _MISS)
        if found is not _MISS:
            entry.update(cached=True, ok=True, results=len(found))
        elif not self._up(name) or not self._up("openalex"):
            return [], dict(entry, ok=False, results=0, ms=0,
                            error=f"skipped: failed recently, retrying in {_human(max(self._wait(name), self._wait('openalex')))}")
        else:
            try:
                found = getattr(prov, kind)(query, limit, self.http.clock() + deadline_s, year_from, year_to)
                self.cache.put(key, found, self.search_ttl if found else self.negative_ttl)
                entry.update(ok=True, results=len(found))
            except Exception as e:
                found = []
                self._failed(name, e)
                if isinstance(e, RateLimited):          # the daily budget is shared with the keyword search
                    self._failed("openalex", e)
                entry.update(ok=False, results=0, error=f"{type(e).__name__}: {e}"[:160])
        entry["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return [dict(p) for p in found], entry

    # ---- one paper by DOI, arXiv id or OpenAlex id
    def get_paper(self, identifier, deadline_s=6.0):
        ident = identifier.strip()
        m = _ARXIV_ID.match(ident)
        doi = arxiv_doi(m.group(1)) if m else ident.lower().replace("https://doi.org/", "").removeprefix("doi:")
        number = openalex_number(ident)
        key = ("paper", number or doi)
        cached = self.cache.get(key, _MISS)
        if cached is not _MISS:
            return (dict(cached) if cached else None), {"cached": True, "errors": []}
        deadline = self.http.clock() + deadline_s
        report, paper = {"cached": False, "errors": []}, None
        for name in ("openalex", "semanticscholar", "crossref"):    # Crossref has no arXiv or OpenAlex ids
            if name not in self.providers or paper or (name == "crossref" and (number or m)):
                continue
            if not self._up(name):
                report["errors"].append(f"{self.providers[name].name}: skipped, failed recently, retrying in {_human(self._wait(name))}")
                continue
            try:
                prov = self.providers[name]
                if name == "openalex":
                    paper = next(iter(prov.by_ids([number], refs=True, deadline=deadline)), None) if number \
                        else prov.lookup([doi], deadline).get(doi)
                elif name == "semanticscholar" and not number:
                    sid = ("ARXIV:" + doi.split("arxiv.", 1)[1]) if doi.startswith("10.48550/arxiv.") else "DOI:" + doi
                    paper = prov.lookup([sid], deadline).get(sid)
                elif name == "crossref":
                    paper = prov.lookup(doi, deadline)
            except Exception as e:
                self._failed(name, e)
                report["errors"].append(f"{self.providers[name].name}: {type(e).__name__}: {e}"[:160])
        if paper or not report["errors"]:            # only remember "not found" when the APIs answered
            self.cache.put(key, paper, self.paper_ttl if paper else self.negative_ttl)
        return paper, report
