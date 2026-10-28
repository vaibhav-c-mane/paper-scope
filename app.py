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
import logging
import os

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from admin import AdminLogin, settings_from_env
from engine.domains import DOMAINS
from engine.pipeline import MODES, SearchPipeline
from engine.sources import ScholarlyClient

SORTS = ("relevance", "year", "citations")
UNDERSTOOD_SHOWN = ("changed", "query", "original", "corrections", "recent_from")    # the notice above the results
RANKING_NUMBERS = ("score", "relevance", "features")
# The page's own scripts only (an injected <script> cannot run), styles and fonts from Google Fonts,
# API calls to this site only, and no other site may show the page in a frame.
CONTENT_POLICY = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                  "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; object-src 'none'; "
                  "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")

HERE = os.path.dirname(os.path.abspath(__file__))


def _int(src, name, default=None):
    v = str(src.get(name, "") or "").strip()
    return int(v) if v.lstrip("-").isdigit() else default


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


def create_app(pipeline=None, admin_settings=None):
    """admin_settings: the admin login (path, user, password_hash, secret_key); read from .env when None."""
    if pipeline is None:
        pipeline = SearchPipeline(models_dir=os.path.join(HERE, "data", "models"), client=ScholarlyClient(),
                                  capacity=int(os.environ.get("PAPERSCOPE_CAPACITY", "20000")))
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"))
    app.config["pipeline"] = pipeline
    admin = app.config["admin"] = AdminLogin(app, settings_from_env() if admin_settings is None else admin_settings)

    @app.after_request
    def safety_headers(response):
        h = response.headers
        h.setdefault("Content-Security-Policy", CONTENT_POLICY)
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("Referrer-Policy", "same-origin")          # other sites never see what was searched
        if app.config.get("HTTPS_ONLY"):
            h.setdefault("Strict-Transport-Security", "max-age=31536000")
        return response

    @app.get("/")
    def index():
        if admin.signed_in():
            return admin.search_page(os.path.join(app.static_folder, "index.html"))
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/search")
    def api_search():
        a = request.args
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
        return jsonify({"suggestions": pipeline.suggest(request.args.get("prefix", "")),
                        "titles": pipeline.suggest_titles(request.args.get("q", ""))})

    @app.get("/api/paper")
    def api_paper():
        p = pipeline.paper(request.args.get("key", ""))
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
