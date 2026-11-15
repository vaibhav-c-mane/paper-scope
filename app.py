"""PaperScope - research paper search over OpenAlex, arXiv, Europe PMC, Semantic Scholar and Crossref,
with an in-memory hybrid index (BM25 + embeddings) and a learned ranker.

Run:  python app.py            then open http://127.0.0.1:5000
      PORT=8000 python app.py
      OPENALEX_EMAIL=you@uni.edu python app.py     (OpenAlex's faster "polite pool")
On a public website use python wsgi.py instead (a production server, see wsgi.py).

Papers fetched from the internet are kept in memory (at most PAPERSCOPE_CAPACITY,
default 20000, least recently used evicted first) and forgotten when the app stops.
Nothing is written to disk unless you turn on the cache file (PAPERSCOPE_CACHE_FILE
in .env), which keeps API answers between runs to save the daily OpenAlex budget.

Visitors get the papers and what the search understood. The engine's trace, the ranking
numbers, the test settings and the OpenAlex budget are only for the admin (admin/,
switched on with python scripts/set_admin.py).
"""
import hashlib
import html
import json
import logging
import math
import os
import re
import threading
import time

from flask import Flask, Response, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from admin import AdminLogin, settings_from_env
from dsa.lru import LRUCache
from engine.domains import DOMAINS
from engine.pipeline import MODES, SearchPipeline
from engine.sources import ScholarlyClient

SORTS = ("relevance", "year", "citations")
UNDERSTOOD_SHOWN = ("changed", "query", "original", "corrections", "recent_from")    # the notice above the results
RANKING_NUMBERS = ("score", "relevance", "features")
MAX_QUERY, MAX_AUTHOR, MAX_KEY = 500, 100, 300      # characters; a pasted title or question fits easily
SEARCHES_PER_MINUTE = 30                            # per visitor address, searches and opened papers together
# Search engines may index the start page; the API stays off limits (a crawler running searches
# would spend the OpenAlex budget). The admin login's address is secret, so it is never named here.
ROBOTS = "User-agent: *\nDisallow: /api/\n"
SITE_RE = re.compile(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
STATIC_MAX_AGE = 365 * 24 * 3600                    # seconds a browser keeps a versioned static file
NOT_FOUND_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>Page not found | PaperScope</title>
<script src="{theme}"></script>
<link rel="stylesheet" href="{style}">
</head>
<body>
<main class="home">
  <h1 class="home-title">This page does not exist</h1>
  <p class="home-lede">The address may be mistyped, or the page has moved. Every search starts from the start page.</p>
  <a class="btn btn-primary" href="/">Go to the start page</a>
</main>
</body>
</html>
"""
# The page's own scripts only (an injected <script> cannot run), styles and fonts from Google Fonts,
# API calls to this site only, and no other site may show the page in a frame.
CONTENT_POLICY = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                  "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                  "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")

HERE = os.path.dirname(os.path.abspath(__file__))


def _int(src, name, default=None):
    """A whole number from the address, or `default` for anything else ("--5", "²", 5,000 digits)."""
    v = str(src.get(name, "") or "").strip()
    return int(v) if re.fullmatch(r"-?[0-9]{1,9}", v) else default


class VisitorLimit:
    """How often one visitor address may search or open a paper: a token bucket each, so
    `per_minute` can come at once and then one every 60/per_minute seconds. A new topic spends
    the OpenAlex budget and the free sources' patience, so one script cannot use them up.
    The buckets live in an LRU cache of at most 10,000 addresses; an entry expires a minute after
    the visitor's last request, when the bucket would be full again anyway."""

    def __init__(self, per_minute=SEARCHES_PER_MINUTE, clock=time.monotonic):
        self.rate = per_minute / 60.0
        self.burst = float(per_minute)
        self.clock = clock
        self.buckets = LRUCache(10000, ttl=60.0, clock=clock)    # address -> (tokens, time, warned)
        self.lock = threading.Lock()

    def wait_s(self, addr):
        """0 when the visitor may go ahead (one token is used), else the seconds to wait."""
        now = self.clock()
        with self.lock:
            tokens, last, warned = self.buckets.get(addr, (self.burst, now, False))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self.buckets.put(addr, (tokens - 1.0, now, False))
                return 0.0
            self.buckets.put(addr, (tokens, now, True))
        if not warned:                                  # once per run of refusals, not for every request
            logging.getLogger("paperscope").warning("search limit: %s made more than %d requests in a minute",
                                                    addr, int(self.burst))
        return (1.0 - tokens) / self.rate


def site_address(value):
    """The site's public address, e.g. paperscope.onrender.com (PAPERSCOPE_SITE): the one address
    search engines should list, in the sitemap and as the start page's canonical address. A scheme
    or path typed with it is dropped; anything that is not an address is ignored ("")."""
    v = (value or "").strip()
    v = re.sub(r"^https?://", "", v, flags=re.I).split("/", 1)[0].lower()
    return v if SITE_RE.fullmatch(v) else ""


def shape(res, admin):
    """The words to mark and what the search understood, for everyone; the trace and the
    ranking numbers only for the admin."""
    trace = res["trace"]
    u = trace.get("understood")
    res["words"] = (trace.get("query") or {}).get("words") or []
    res["understood"] = {k: u[k] for k in UNDERSTOOD_SHOWN} if u else None
    if not admin:
        del res["trace"]
        for p in res["results"]:
            for k in RANKING_NUMBERS:
                p.pop(k, None)
    return res


def create_app(pipeline=None, admin_settings=None, searches_per_minute=None, site=None, google_verification=None):
    """admin_settings: the admin login (path, user, password_hash, secret_key); read from .env when None.
    searches_per_minute: per visitor address (PAPERSCOPE_SEARCHES_PER_MINUTE, default 30; 0 = no limit).
    site: the public address (PAPERSCOPE_SITE); google_verification: the code Google Search Console
    gives for its "HTML tag" check (PAPERSCOPE_GOOGLE_VERIFICATION). Both are read from .env when None."""
    if pipeline is None:
        pipeline = SearchPipeline(models_dir=os.path.join(HERE, "data", "models"), client=ScholarlyClient(),
                                  capacity=int(os.environ.get("PAPERSCOPE_CAPACITY", "20000")))
    if searches_per_minute is None:
        searches_per_minute = int(os.environ.get("PAPERSCOPE_SEARCHES_PER_MINUTE") or SEARCHES_PER_MINUTE)
    site = site_address(os.environ.get("PAPERSCOPE_SITE") if site is None else site)
    if google_verification is None:
        google_verification = os.environ.get("PAPERSCOPE_GOOGLE_VERIFICATION", "")
    google_verification = google_verification.strip()
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
    app.config["pipeline"] = pipeline
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024        # the only form is the admin sign-in
    admin = app.config["admin"] = AdminLogin(app, settings_from_env() if admin_settings is None else admin_settings)
    limit = VisitorLimit(searches_per_minute) if searches_per_minute > 0 else None

    def too_many():
        """None when this visitor may search now, else the answer that asks them to wait."""
        if limit is None or admin.signed_in():
            return None
        wait = limit.wait_s(request.remote_addr or "unknown")
        if not wait:
            return None
        r = jsonify({"error": f"Too many searches from your connection. Wait {math.ceil(wait)} seconds, then try again."})
        r.status_code = 429
        r.headers["Retry-After"] = str(math.ceil(wait))
        return r

    @app.after_request
    def safety_headers(response):
        h = response.headers
        h.setdefault("Content-Security-Policy", CONTENT_POLICY)
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")                 # frame-ancestors, for older browsers
        h.setdefault("Referrer-Policy", "same-origin")          # other sites never see what was searched
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if request.path.startswith("/api/"):
            h.setdefault("Cache-Control", "no-store")           # answers can hold the admin's details
        if app.config.get("HTTPS_ONLY"):
            h.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    def base_url():
        """The start page's full address, for the sitemap and robots.txt."""
        if site:
            return f"https://{site}/"
        return f"https://{request.host}/" if app.config.get("HTTPS_ONLY") else request.host_url

    versions = {}                                       # static file -> (modified time, short hash)

    def asset(name):
        """/static/<name>?v=<hash of the file>: a changed file gets a new address, so a browser can
        keep each version for a year and still never use an old one after a deploy."""
        path = os.path.join(app.static_folder, name)
        mtime = os.path.getmtime(path)
        if versions.get(name, (None,))[0] != mtime:
            with open(path, "rb") as f:
                versions[name] = (mtime, hashlib.sha256(f.read()).hexdigest()[:10])
        return f"/static/{name}?v={versions[name][1]}"

    app.context_processor(lambda: {"asset": asset})     # the admin login page uses it too
    # versioned static files are kept for a year; an address without ?v= is checked every time
    app.get_send_file_max_age = lambda filename: STATIC_MAX_AGE if request.args.get("v") else None

    def page_html(start_page):
        """index.html with versioned file addresses, Google's check code, and on the start page the
        canonical address, the link preview's address and image, and what the site is (JSON-LD)."""
        with open(os.path.join(app.static_folder, "index.html"), encoding="utf-8") as f:
            page = f.read()
        page = re.sub(r'="/static/([\w.-]+)"', lambda m: f'="{asset(m.group(1))}"', page)
        head = []
        if site and start_page:
            url = f"https://{site}/"
            head += [f'<link rel="canonical" href="{url}">', f'<meta property="og:url" content="{url}">',
                     f'<meta property="og:image" content="https://{site}{asset("og-image.png")}">']
            described = re.search(r'<meta name="description" content="([^"]*)"', page)
            about = {"@context": "https://schema.org", "@graph": [
                {"@type": "WebSite", "name": "PaperScope", "url": url},
                {"@type": "WebApplication", "name": "PaperScope", "url": url,
                 "description": html.unescape(described.group(1)) if described else "",
                 "applicationCategory": "ReferenceApplication", "operatingSystem": "Any",
                 "isAccessibleForFree": True, "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}}]}
            # "</" is written as "<\/" so nothing in the data can end the script element early
            head.append('<script type="application/ld+json">' + json.dumps(about).replace("</", "<\\/") + "</script>")
        if google_verification:
            head.append(f'<meta name="google-site-verification" content="{html.escape(google_verification)}">')
        return page.replace("</head>", "".join(h + "\n" for h in head) + "</head>", 1) if head else page

    @app.get("/")
    def index():
        results = bool(request.args.get("q"))
        page = page_html(start_page=not results)
        r = admin.search_page(page) if admin.signed_in() else Response(page, mimetype="text/html")
        if results:                                             # a page of results: only the start page belongs in search engines
            r.headers["X-Robots-Tag"] = "noindex"
        return r

    @app.errorhandler(404)
    def not_found(e):
        """A page with a way back for people, plain JSON for the API. Also what a wrong admin
        address gets, so nothing tells that a hidden page exists."""
        if request.path.startswith("/api/"):
            return jsonify({"error": "not found"}), 404
        return Response(NOT_FOUND_PAGE.format(theme=asset("theme.js"), style=asset("style.css")),
                        status=404, mimetype="text/html")

    @app.get("/robots.txt")
    def robots():
        return app.response_class(ROBOTS + f"Sitemap: {base_url()}sitemap.xml\n", mimetype="text/plain")

    @app.get("/sitemap.xml")
    def sitemap():
        # the start page only: pages of results are not indexed, and the admin login is secret
        xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               f"  <url><loc>{html.escape(base_url())}</loc></url>\n"
               "</urlset>\n")
        return app.response_class(xml, mimetype="application/xml")

    @app.get("/api/search")
    def api_search():
        a = request.args
        if len(a.get("q", "")) > MAX_QUERY or len(a.get("author", "")) > MAX_AUTHOR:
            return jsonify({"error": f"That is too long: a search can have at most {MAX_QUERY} characters, "
                                     f"an author name {MAX_AUTHOR}."}), 400
        is_admin = admin.signed_in()
        # the test settings: for everyone else the defaults, so no one can make every search go online
        online = a.get("online", "auto") if is_admin else "auto"
        mode = a.get("mode", "learned") if is_admin else "learned"
        sort = a.get("sort", "relevance")
        if online not in ("auto", "always", "never") or mode not in MODES or sort not in SORTS:
            return jsonify({"error": "online must be auto, always or never; mode one of " + ", ".join(MODES)
                            + "; sort one of " + ", ".join(SORTS)}), 400
        field = a.get("field") or None
        if field and field not in DOMAINS:
            return jsonify({"error": f"unknown field '{field}'"}), 400
        refused = too_many()
        if refused:
            return refused
        try:
            res = pipeline.search(a.get("q", ""), year_from=_int(a, "year_from"), year_to=_int(a, "year_to"),
                                  author=a.get("author", ""), field=field, strict_field=a.get("strict") == "1",
                                  sort_by=sort, page=max(1, _int(a, "page", 1)),
                                  per_page=10, external=online, mode=mode, exact=a.get("exact") == "1")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(shape(res, is_admin))

    @app.get("/api/suggest")
    def api_suggest():
        return jsonify({"suggestions": pipeline.suggest(request.args.get("prefix", "")[:60]),
                        "titles": pipeline.suggest_titles(request.args.get("q", "")[:MAX_QUERY])})

    @app.get("/api/paper")
    def api_paper():
        key = request.args.get("key", "")
        if len(key) > MAX_KEY:
            return jsonify({"error": "paper not found"}), 404
        refused = too_many()                                    # a paper not in memory is looked up online
        if refused:
            return refused
        p = pipeline.paper(key)
        return (jsonify(p), 200) if p else (jsonify({"error": "paper not found"}), 404)

    @app.get("/api/state")
    def api_state():
        return jsonify(pipeline.state() if admin.signed_in() else {"fields": DOMAINS})

    return app


def for_deployment(app, proxies=1, https=True):
    """Settings for a public website (wsgi.py).

    proxies: how many proxies (the host's, nginx) pass each request on. Each adds the address it
    got the request from to X-Forwarded-For, so the visitor's real address is that many places
    from the end; earlier entries can be made up by the visitor and are ignored. The admin
    login's wrong-password limit counts by this address. 0 when visitors reach the app directly.
    https: the sign-in cookie is only ever sent over HTTPS, and browsers are told to always use
    HTTPS for this site."""
    if proxies:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=proxies, x_proto=proxies)
    if https:
        # "__Host-" makes the browser refuse the cookie unless it is HTTPS-only and for this site alone
        app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_NAME="__Host-session", HTTPS_ONLY=True)
    return app


def describe(app):
    """One line for the terminal when the app starts."""
    s = app.config["pipeline"].state()
    return (f"PaperScope: ranker {s['ranker']['kind']}, lexicon of {s['lexicon'].get('vocabulary', 0):,} words, "
            f"memory for {s['capacity']:,} papers. Searches go to OpenAlex, then arXiv or Europe PMC by field, "
            f"Semantic Scholar and Crossref" + (" (cache file on)" if s["budget"].get("cache_file") else "") + ". " +
            ("Admin login is on." if app.config["admin"].enabled
             else "Admin login is off (python scripts/set_admin.py turns it on)."))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    application = create_app()
    print(describe(application))
    application.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
