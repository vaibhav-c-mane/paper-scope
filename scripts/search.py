"""Search from the command line, with the same pipeline as the web app.

  python scripts/search.py "graph neural networks"
  python scripts/search.py "protein folding" --field "Biology & Medicine" -k 5
  python scripts/search.py "deep AND (vision OR image) NOT survey" --year-from 2018
  python scripts/search.py 10.48550/arXiv.1706.03762
  python scripts/search.py "transformer attention machine translation" "attention transformers for translation"   # several queries, one session

Several queries in one call share the in-memory index, so you can see the second
search answered from memory without asking the internet again.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from engine.domains import DOMAINS                      # noqa: E402
from engine.pipeline import MODES, SearchPipeline       # noqa: E402
from engine.sources import ScholarlyClient              # noqa: E402


def show(res, k):
    t = res["trace"]
    if t.get("field"):
        print("field: " + ", ".join(f"{d['name']} {d['p']:.0%}" for d in t["field"]["detected"]))
    for err in (t.get("identifier") or {}).get("online", {}).get("errors", []):
        print("lookup: " + err)
    on = t.get("online", {})
    if on.get("used"):
        prov = ", ".join(f"{r['provider']}{'' if r.get('ok') else ' (skipped)' if str(r.get('error', '')).startswith('skipped') else ' (failed)'}"
                         f"{' (cached)' if r.get('cached') else ''}"
                         for r in on["providers"])
        print(f"online: {on['reason']}; {on['fetched']} papers from {prov}, {on['new']} new")
    elif on:
        print(f"online: not needed ({on['reason']})")
    print(f"{res['total']} results in {t['total_ms']:.0f} ms\n")
    for i, p in enumerate(res["results"][:k], 1):
        print(f"{i:2d}. {p['title'][:95]}")
        venue = p["venue"] or ("preprint" if p.get("published") is False else "")
        cites = "?" if p["citations"] is None else p["citations"]
        print(f"    {p['year'] or '?'} · {venue[:40]} · {cites} citations · {p['field']} · {p['source']}")
        print("    score {:.3f} · relevance {:.2f} · ".format(p["score"], p["relevance"])
              + " ".join(f"{n}={v:.2f}" for n, v in p["features"].items()))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="+")
    ap.add_argument("-k", type=int, default=10, help="results to show")
    ap.add_argument("--mode", choices=MODES, default="learned")
    ap.add_argument("--online", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--field", choices=DOMAINS, help="favour papers from this research field")
    ap.add_argument("--strict-field", action="store_true", help="only papers from --field")
    ap.add_argument("--year-from", type=int)
    ap.add_argument("--year-to", type=int)
    ap.add_argument("--author", default="")
    ap.add_argument("--sort", choices=("relevance", "year", "citations"), default="relevance")
    args = ap.parse_args()

    pipe = SearchPipeline(models_dir=os.path.join(HERE, "data", "models"), client=ScholarlyClient())
    print(f"ranker: {pipe.ranker.kind}\n")
    for q in args.queries:
        print(f'>>> "{q}"')
        try:
            res = pipe.search(q, args.year_from, args.year_to, args.author, args.field, args.strict_field,
                              sort_by=args.sort, per_page=args.k, external=args.online, mode=args.mode)
        except ValueError as e:
            print("error:", e)
            continue
        show(res, args.k)


if __name__ == "__main__":
    main()
