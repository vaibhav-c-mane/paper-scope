"""Stream real papers with their reference lists from the public OpenAlex snapshot, for training.

Only needed when the OpenAlex API cannot be reached (`python -m ml.train --online`
needs nothing from here). The snapshot is free and needs no account: its part
files are read straight from the internet, decompressed on the fly, and only the
kept papers are written, one JSON per line (~1.6 KB each):

  research works (article, conference paper, preprint, review), English,
  abstract of at least 40 words, published 1990 or later, a research field we
  can map, no retractions, no duplicate titles; Biology & Medicine (half of
  OpenAlex) is capped so the other fields are not drowned out.

  python scripts/fetch_openalex.py pool.jsonl --papers 600000
  python -m ml.train --pool pool.jsonl
  (delete pool.jsonl afterwards if you like: the search engine never reads it)

Licence: OpenAlex data is CC0 (https://docs.openalex.org/download-all-data/openalex-snapshot).
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import zlib
from multiprocessing import Pool

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from engine.domains import weak_label                   # noqa: E402
from engine.papers import paper_from_openalex           # noqa: E402

MANIFEST = "https://openalex.s3.amazonaws.com/data/jsonl/works/manifest.json"
KEEP = {"article", "conference-paper", "preprint", "review"}
FIELDS = ("arxiv_id", "title", "authors", "abstract", "categories", "topic", "year", "venue", "published",
          "venue_quality", "doi", "url", "oa_url", "citations", "openalex_id", "source")


def parse(lines):
    """Raw snapshot lines -> JSON lines of kept papers (runs in worker processes)."""
    out = []
    for line in lines:
        if b'"abstract_inverted_index":null' in line or b'"abstract_inverted_index": null' in line:
            continue                                         # cheap reject before parsing JSON
        try:
            w = json.loads(line)
        except ValueError:
            continue
        if w.get("type") not in KEEP or w.get("language") not in ("en", None) or w.get("is_retracted"):
            continue
        p = paper_from_openalex(w)
        if not p or len(p["abstract"].split()) < 40 or p["year"] < 1990 or not weak_label(p):
            continue
        d = {k: p[k] for k in FIELDS}
        d["refs"] = [int(r.rsplit("W", 1)[1]) for r in (w.get("referenced_works") or []) if "W" in r]
        out.append(json.dumps(d, ensure_ascii=False))
    return out


def stream_lines(url, chunk=1 << 22):
    """Lines of a gzip file on the internet, in batches, without saving the file."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    buf = b""
    with urllib.request.urlopen(url, timeout=60) as r:
        while True:
            data = r.read(chunk)
            if not data:
                break
            buf += d.decompress(data)
            *lines, buf = buf.split(b"\n")
            yield lines
    if buf.strip():
        yield [buf]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output file (JSON lines)")
    ap.add_argument("--papers", type=int, default=600000, help="stop after this many papers")
    ap.add_argument("--bio-cap", type=int, default=60000, help="at most this many Biology & Medicine papers")
    ap.add_argument("--seed", type=int, default=5, help="which part files to read first")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    with urllib.request.urlopen(MANIFEST, timeout=60) as r:
        manifest = json.load(r)
    # large part files of recently updated works: many works per request
    files = [f["url"].replace("s3://openalex/", "https://openalex.s3.amazonaws.com/") for f in manifest["files"]
             if f["meta"]["record_count"] >= 200000 and "updated_date=202" in f["url"]]
    random.Random(a.seed).shuffle(files)
    seen, kept, bio, t0 = set(), 0, 0, time.time()
    with open(a.out, "w", encoding="utf-8") as out, Pool(a.workers) as pool:
        for url in files:
            print("reading", url, flush=True)
            for batch in pool.imap(parse, stream_lines(url)):
                for line in batch:
                    rec = json.loads(line)
                    title = rec["title"].lower()
                    if title in seen:
                        continue
                    if weak_label(rec) == "Biology & Medicine":
                        bio += 1
                        if bio > a.bio_cap:
                            continue
                    seen.add(title)
                    out.write(line + "\n")
                    kept += 1
                if kept >= a.papers:
                    break
            print(f"  {kept:,} papers after {time.time() - t0:.0f}s", flush=True)
            if kept >= a.papers:
                break
    print(f"done: {kept:,} papers in {a.out}")


if __name__ == "__main__":
    main()
