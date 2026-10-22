"""Ranking features: what the ranker looks at for every (query, paper) pair.

Relevance (all 0..1, comparable across queries):
  bm25            BM25 score / the best score this query could reach (lexical match)
  semantic        cosine similarity of query and paper embeddings (same meaning,
                  even with different words)
  title_coverage  share of the query words that appear in the title
  term_coverage   share of the query words that appear anywhere in the paper
  domain_match    P(domain | query) . P(domain | paper): both about the same field

Publication metadata (0..1, -1 = unknown; engine/metadata.py):
  citations, recency, peer_reviewed, venue_quality

The learned ranker sees all nine. The hand-set fallback (no model trained yet)
uses the relevance features and multiplies in a small metadata prior.
"""
import numpy as np

from engine.metadata import METADATA_FEATURES, metadata_matrix

RELEVANCE_FEATURES = ["bm25", "semantic", "title_coverage", "term_coverage", "domain_match"]
FEATURES = RELEVANCE_FEATURES + METADATA_FEATURES
PRIOR_WEIGHTS = {"citations": 0.4, "recency": 0.2, "peer_reviewed": 0.2, "venue_quality": 0.2}


def feature_matrix(index, docs, groups, ids, weights, qvec, p_domain, now_year=None):
    """(features (n, 9), raw BM25 scores) for papers `docs` of a PaperIndex."""
    docs = np.asarray(docs, np.int64)
    x = np.zeros((len(docs), len(FEATURES)), np.float32)
    raw = np.zeros(len(docs), np.float32)
    if not len(docs):
        return x, raw
    if len(ids):
        raw = index.bm25(ids, weights)[docs]
        x[:, 0] = np.clip(raw / index.ceiling(ids, weights), 0, 1)
    if qvec is not None:
        x[:, 1] = np.clip(index.vectors.view[docs] @ qvec, 0, 1)
    x[:, 2], x[:, 3] = index.coverage(groups, docs)
    if p_domain is not None:
        x[:, 4] = index.domains.view[docs] @ p_domain
    m = index.meta.view[docs]
    x[:, 5:] = metadata_matrix(m[:, 0], m[:, 1], m[:, 2], m[:, 3], now_year)
    return x, raw


def metadata_prior(x):
    """Fixed 0..1 quality prior from the metadata columns of a feature matrix (unknown = 0)."""
    m = np.maximum(np.asarray(x)[:, len(RELEVANCE_FEATURES):], 0)
    w = np.asarray([PRIOR_WEIGHTS[f] for f in METADATA_FEATURES], np.float32)
    return m @ w
