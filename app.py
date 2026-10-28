"""PaperScope - research paper search over OpenAlex, arXiv, Europe PMC, Semantic Scholar and Crossref,
with an in-memory hybrid index (BM25 + embeddings) and a learned ranker.

Run:  python app.py            then open http://127.0.0.1:5000
      PORT=8000 python app.py
      OPENALEX_EMAIL=you@uni.edu python app.py     (OpenAlex's faster "polite pool")

Papers fetched from the internet are kept in memory (at most PAPERSCOPE_CAPACITY,
default 20000, least recently used evicted first) and forgotten when the app stops.
Nothing is written to disk unless you turn on the cache file (PAPERSCOPE_CACHE_FILE
in .env), which keeps API answers between runs to save the daily OpenAlex budget.

Visitors get the papers and what the search understood. The engine's trace, the ranking
numbers, the test settings and the OpenAlex budget are only for the admin (admin/,
switched on with python scripts/set_admin.py).
"""
import os

from flask import Flask, jsonify, request, send_from_directory

from admin import AdminLogin, settings_from_env
from engine.domains import DOMAINS
from engine.pipeline import MODES, SearchPipeline
from engine.sources import ScholarlyClient

SORTS = ("relevance", "year", "citations")
UNDERSTOOD_SHOWN = ("changed", "query", "original", "corrections", "recent_from")    # the notice above the results
RANKING_NUMBERS = ("score", "relevance", "features")

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


if __name__ == "__main__":
    application = create_app()
    s = application.config["pipeline"].state()
    print(f"PaperScope: ranker {s['ranker']['kind']}, lexicon of {s['lexicon'].get('vocabulary', 0):,} words, "
          f"memory for {s['capacity']:,} papers. Searches go to OpenAlex, then arXiv or Europe PMC by field, "
          f"Semantic Scholar and Crossref" + (" (cache file on)" if s["budget"].get("cache_file") else "") + ". " +
          ("Admin login is on." if application.config["admin"].enabled
           else "Admin login is off (python scripts/set_admin.py turns it on)."))
    application.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False, threaded=True)
