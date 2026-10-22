"""Publication metadata as ranking signals: citations, recency, publication status, venue quality.

Each signal is a number in 0..1, or -1 when it is not known. The gradient-boosted
ranker can split on "-1" separately, so "unknown" is never confused with "bad".

  citations      log(1 + citations per year since publication) / log(101), capped at 1.
                 Per year, so a 2-year-old paper with 200 citations is not beaten
                 by a 20-year-old one with 300.
  recency        exp(-age / 8 years): 1 this year, 0.29 after 10 years.
  peer_reviewed  1 published in a journal / conference, 0 preprint only.
  venue_quality  1.0 top venue, 0.75 strong, 0.5 recognised, 0.25 published but
                 not ranked, 0 preprint server. Two sources:
                 * OpenAlex venue listings (`listed_in`): the Norwegian and Finnish
                   national rankings (level 2 / level 3 = top of their field), the
                   CCF computer-science ranking (A/B/C), CWTS core journals, ...
                   These arrive with papers from OpenAlex (snapshot, search or
                   enrichment), so they cover every field.
                 * For arXiv journal references, which have no listings, a short
                   list of flagship venues (below), so offline data still has a signal.
"""
import datetime
import math
import re

import numpy as np

METADATA_FEATURES = ["citations", "recency", "peer_reviewed", "venue_quality"]
RECENCY_SCALE = 8.0
UNKNOWN = -1.0

_TOP_LISTS = {"norway-2", "jufo-3", "ccf-a", "abdc-a-star", "ft50", "utd24", "fnege-1-star", "ki-jl-3"}
_STRONG_LISTS = {"jufo-2", "ccf-b", "anvur-class-a", "abdc-a", "fnege-1", "ki-jl-2"}
_RECOGNISED_LISTS = {"cwts-core", "norway-1", "jufo-1", "ccf-c", "abdc-b", "ki-jl-1"}

# Flagship venues by name (lower case). Matched against arXiv journal references
# such as "Phys. Rev. Lett. 101, 123456 (2008)" or "NeurIPS 2019".
_TOP_NAMES = re.compile("|".join([
    r"neurips", r"neural information processing systems", r"\bnips\b", r"\bicml\b",
    r"international conference on machine learning", r"\biclr\b", r"learning representations",
    r"\bcvpr\b", r"computer vision and pattern recognition", r"\biccv\b", r"\beccv\b",
    r"international conference on computer vision", r"european conference on computer vision",
    r"\bacl\b", r"association for computational linguistics", r"\bemnlp\b", r"\bnaacl\b",
    r"\baaai\b", r"\bijcai\b", r"\bkdd\b", r"knowledge discovery and data mining", r"\bsigir\b",
    r"\bsigmod\b", r"\bvldb\b", r"\bosdi\b", r"\bsosp\b", r"\bnsdi\b", r"\bsigcomm\b",
    r"computer and communications security", r"usenix security", r"symposium on security and privacy",
    r"\bstoc\b", r"symposium on theory of computing", r"\bfocs\b", r"foundations of computer science",
    r"\bsoda\b", r"\bicra\b", r"\biros\b", r"robotics: science and systems",
    r"journal of machine learning research", r"\bjmlr\b", r"pattern analysis and machine intelligence",
    r"\bt-?pami\b", r"international journal of computer vision", r"journal of the acm",
    r"^nature\b", r"^science( advances)?[\s,.(\d]", r"^science$", r"physical review letters",
    r"phys\.? ?rev\.? ?lett", r"physical review x\b", r"phys\.? ?rev\.? ?x\b", r"\bpnas\b",
    r"proceedings of the national academy of sciences", r"^cell\b", r"lancet",
    r"new england journal of medicine", r"annals of statistics", r"ann\.? statist",
    r"annals of mathematics", r"ann\.? of math", r"journal of the american statistical association",
    r"econometrica", r"american economic review", r"quarterly journal of economics", r"journal of finance\b",
]))
_STRONG_NAMES = re.compile("|".join([
    r"physical review [a-e]\b", r"phys\.? ?rev\.? ?[a-e]\b", r"astrophysical journal", r"\bapj\b",
    r"monthly notices of the royal astronomical society", r"\bmnras\b", r"astronomy (and|&) astrophysics",
    r"journal of high energy physics", r"\bjhep\b", r"\bieee trans", r"\bacm trans",
    r"machine learning\s*\d", r"artificial intelligence\s*\d", r"\bcolt\b", r"\buai\b", r"\baistats\b",
    r"\bwww\b", r"the web conference", r"\bwsdm\b", r"\bicde\b", r"\bbmvc\b", r"\bcorl\b",
]))


def flagship_tier(venue):
    """1.0 / 0.75 for venues on the flagship lists, else None."""
    v = " ".join((venue or "").lower().split())
    if not v:
        return None
    if _TOP_NAMES.search(v):
        return 1.0
    if _STRONG_NAMES.search(v):
        return 0.75
    return None


def venue_quality_from_source(src, published=None):
    """Venue quality from an OpenAlex `primary_location.source` object."""
    if not src:
        return 0.0 if published is False else None
    if src.get("type") == "repository":
        return 0.0
    listed = set(src.get("listed_in") or [])
    if listed & _TOP_LISTS:
        q = 1.0
    elif listed & _STRONG_LISTS:
        q = 0.75
    elif listed & _RECOGNISED_LISTS or src.get("is_core"):
        q = 0.5
    elif src.get("type") in ("journal", "conference", "book series"):
        q = 0.25
    else:
        return None
    return max(q, flagship_tier(src.get("display_name")) or 0.0)


def published_flag(paper):
    """1 published, 0 preprint only, -1 unknown."""
    if paper.get("published") is True or (paper.get("venue") or "").strip():
        return 1
    if paper.get("published") is False:
        return 0
    return -1


def venue_quality(paper):
    q = paper.get("venue_quality")
    tier = flagship_tier(paper.get("venue"))
    if q is not None and q >= 0:
        return max(float(q), tier or 0.0)
    if tier:
        return tier
    status = published_flag(paper)
    return 0.25 if status == 1 else 0.0 if status == 0 else UNKNOWN


def citation_count(paper):
    c = paper.get("citations")
    try:
        return -1 if c is None else max(0, int(c))
    except (TypeError, ValueError):
        return -1


def current_year():
    return datetime.date.today().year


# ------------------------------------------------------------------ vectorised, at query time
def citation_score(citations, years, now_year):
    citations = np.asarray(citations, np.float32)
    years = np.asarray(years, np.float32)
    age = np.where(years > 0, np.maximum(1.0, now_year - years + 1.0), 5.0)
    s = np.minimum(1.0, np.log1p(np.maximum(citations, 0) / age) / math.log1p(100))
    return np.where(citations < 0, UNKNOWN, s).astype(np.float32)


def recency_score(years, now_year):
    years = np.asarray(years, np.float32)
    s = np.exp(-np.maximum(0.0, now_year - years) / RECENCY_SCALE)
    return np.where(years > 0, s, UNKNOWN).astype(np.float32)


def metadata_matrix(years, citations, published, venue_q, now_year=None):
    """(n, 4) matrix in METADATA_FEATURES order from the stored metadata columns.
    `now_year` is when the search happens (for recency); training sets it to the year
    the query paper was written. Citation counts are always today's, so citations
    per year are always counted up to today."""
    out = np.empty((len(years), len(METADATA_FEATURES)), np.float32)
    out[:, 0] = citation_score(citations, years, current_year())
    out[:, 1] = recency_score(years, now_year or current_year())
    out[:, 2] = np.asarray(published, np.float32)
    out[:, 3] = np.asarray(venue_q, np.float32)
    return out


def columns_of(paper):
    """The four stored metadata values of one paper: (year, citations, published, venue_quality)."""
    return int(paper.get("year") or 0), citation_count(paper), published_flag(paper), venue_quality(paper)
