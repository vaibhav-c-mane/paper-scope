"""Paper records: every source's answer becomes one plain dict with the same keys.

    arxiv_id, title, authors, abstract, categories, topic, year, venue, published,
    venue_quality, doi, url, oa_url, citations, openalex_id, source

Missing fields get safe defaults ("" / 0 / None), so the rest of the engine never
has to check whether a key exists. `None` means "not known", which is different
from 0 (e.g. citations=None: the source did not say; citations=0: nobody cites it).
"""
import re

PAPER_DEFAULTS = {
    "arxiv_id": None, "title": "", "authors": "", "abstract": "", "categories": "",
    "topic": "", "year": 0, "venue": "", "published": None, "venue_quality": None,
    "doi": None, "url": None, "oa_url": None, "citations": None, "openalex_id": None,
    "source": "",
}
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+?)(?:v\d+)?(?:\.pdf)?$", re.I)


def complete(paper):
    """Fill in every missing key with its default (used for external papers too)."""
    out = dict(PAPER_DEFAULTS)
    out.update({k: v for k, v in paper.items() if v is not None or k not in out})
    out["year"] = int(out["year"] or 0)
    return out


def abstract_from_inverted_index(inv):
    """OpenAlex ships abstracts as {word: [positions]}; put the words back in order."""
    if not inv:
        return ""
    slots = {}
    for word, positions in inv.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def paper_from_openalex(w, source="OpenAlex"):
    """An OpenAlex work (API response or snapshot line) -> paper dict, or None without a title."""
    from engine.metadata import venue_quality_from_source        # avoids an import cycle

    title = " ".join((w.get("title") or w.get("display_name") or "").split())
    if not title:
        return None
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    peer_venue = src.get("type") in ("journal", "conference", "book series")
    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower() or None
    arxiv_id = None
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv_id = doi[len("10.48550/arxiv."):]
    else:
        m = _ARXIV_URL.search(loc.get("landing_page_url") or "")
        arxiv_id = m.group(1) if m else None
    # publication status: preprint (repository / submitted version) or published (journal, conference)
    if w.get("type") == "preprint" or src.get("type") == "repository":
        published = False
    elif peer_venue or loc.get("is_published"):
        published = True
    else:
        published = False if loc.get("is_published") is False else None
    topic = w.get("primary_topic") or {}
    parts = [topic.get("display_name"), (topic.get("subfield") or {}).get("display_name"),
             (topic.get("field") or {}).get("display_name")]
    return complete({
        "arxiv_id": arxiv_id, "title": title,
        "authors": ", ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])
                             if (a.get("author") or {}).get("display_name")),
        "abstract": abstract_from_inverted_index(w.get("abstract_inverted_index")),
        "topic": " | ".join(p or "" for p in parts) if any(parts) else "",
        "year": w.get("publication_year") or 0,
        "venue": (src.get("display_name") or "") if peer_venue else "",
        "published": published,
        "venue_quality": venue_quality_from_source(src, published),
        "doi": doi,
        "url": loc.get("landing_page_url") or w.get("id"),
        "oa_url": (w.get("open_access") or {}).get("oa_url"),
        "citations": w.get("cited_by_count"),
        "openalex_id": w.get("id"), "source": source,
    })


_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)


def paper_keys(p):
    """Identity keys used to spot the same paper coming from two sources.
    arXiv's own DOI (10.48550/arXiv.<id>) is turned back into the arXiv id."""
    keys = []
    doi = (p.get("doi") or "").lower().strip()
    m = _ARXIV_DOI.match(doi)
    arxiv_id = (p.get("arxiv_id") or (m.group(1) if m else "") or "").lower()
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    if arxiv_id:
        keys.append("arxiv:" + arxiv_id)
    if doi and not m:
        keys.append("doi:" + doi)
    title = re.sub(r"[^a-z0-9]+", " ", (p.get("title") or "").lower()).strip()
    if title:
        keys.append("title:" + title)
    return keys
