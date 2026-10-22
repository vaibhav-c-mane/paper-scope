"""Research domains: a fixed taxonomy, free "weak" labels, and a learned topic classifier.

Labels come with the data:
* arXiv papers carry author-chosen categories (cs.CV, quant-ph, ...);
* OpenAlex works carry a primary topic with its subfield and field
  ("Domain Adaptation and Few-Shot Learning | Artificial Intelligence | Computer Science").
Both map to one of the 13 domains below. Why a classifier as well?
* queries have no category: the classifier says "pedestrian detection at night"
  is Computer Vision, so Computer Vision papers can be favoured;
* papers from Semantic Scholar / Crossref, and some OpenAlex works, have no usable
  category, so their domain is predicted from title + abstract;
* it gives a probability over domains instead of one hard label, so the
  "same domain as the query" feature is smooth.

Model: logistic regression (one-vs-rest, trained with SGD, balanced classes) on the
TF-IDF vectors the index already computes. It is stored as two plain arrays
(weights and biases), so it loads without pickle and in any library version.
"""
import json
import os
import re

import numpy as np

DOMAINS = [
    "Machine Learning",
    "Computer Vision",
    "Language & Information Retrieval",
    "Robotics & Control",
    "Security & Cryptography",
    "Theory & Algorithms",
    "Systems & Software",
    "Networks & Data Mining",
    "Quantum Physics",
    "Physics & Astronomy",
    "Mathematics & Statistics",
    "Biology & Medicine",
    "Economics & Finance",
]

# ------------------------------------------------------------------ arXiv categories
_EXACT = {
    "cs.LG": "Machine Learning", "stat.ML": "Machine Learning", "cs.NE": "Machine Learning",
    "cs.AI": "Machine Learning", "cs.MA": "Machine Learning",
    "cs.CV": "Computer Vision", "eess.IV": "Computer Vision", "cs.GR": "Computer Vision", "cs.MM": "Computer Vision",
    "cs.CL": "Language & Information Retrieval", "cs.IR": "Language & Information Retrieval",
    "cs.DL": "Language & Information Retrieval",
    "cs.RO": "Robotics & Control", "cs.SY": "Robotics & Control", "eess.SY": "Robotics & Control",
    "math.OC": "Robotics & Control",
    "cs.CR": "Security & Cryptography",
    "cs.DS": "Theory & Algorithms", "cs.CC": "Theory & Algorithms", "cs.DM": "Theory & Algorithms",
    "cs.LO": "Theory & Algorithms", "cs.FL": "Theory & Algorithms", "cs.GT": "Theory & Algorithms",
    "cs.CG": "Theory & Algorithms", "cs.IT": "Theory & Algorithms", "math.IT": "Theory & Algorithms",
    "cs.SI": "Networks & Data Mining", "cs.DB": "Networks & Data Mining", "physics.soc-ph": "Networks & Data Mining",
    "quant-ph": "Quantum Physics",
    "physics.med-ph": "Biology & Medicine", "physics.bio-ph": "Biology & Medicine",
}
_PREFIX = [
    ("cs.", "Systems & Software"), ("eess.", "Systems & Software"),
    ("q-bio", "Biology & Medicine"), ("q-fin", "Economics & Finance"), ("econ", "Economics & Finance"),
    ("math", "Mathematics & Statistics"), ("stat", "Mathematics & Statistics"),
    ("astro-ph", "Physics & Astronomy"), ("hep-", "Physics & Astronomy"), ("gr-qc", "Physics & Astronomy"),
    ("nucl-", "Physics & Astronomy"), ("cond-mat", "Physics & Astronomy"), ("physics", "Physics & Astronomy"),
    ("nlin", "Physics & Astronomy"), ("math-ph", "Physics & Astronomy"),
]
_GENERIC = {"Machine Learning"}   # cs.LG + cs.CL usually means an NLP paper, so prefer the specific one


def domain_of_category(cat):
    if cat in _EXACT:
        return _EXACT[cat]
    for prefix, dom in _PREFIX:
        if cat.startswith(prefix):
            return dom
    return None


# ------------------------------------------------------------------ OpenAlex topics
# Topic names decide first inside the technical fields (an "Artificial Intelligence"
# subfield topic about speech is Language, not generic ML), then the subfield, then the field.
_TOPIC_RULES = [
    (re.compile(r"quantum"), "Quantum Physics"),
    (re.compile(r"natural language|\btext|speech|language model|information retrieval|recommend|sentiment|"
                r"translation|question answering|topic model|semantic web|knowledge graph"),
     "Language & Information Retrieval"),
    (re.compile(r"robot|autonomous vehicle|motion planning|uav|unmanned|manipulat|locomotion"), "Robotics & Control"),
    (re.compile(r"secur|cryptograph|privacy|malware|intrusion|authentication|blockchain|steganograph"),
     "Security & Cryptography"),
    (re.compile(r"image|vision|video|visual|face |facial|object detection|point cloud|3d |graphics|"
                r"remote sensing|segmentation"), "Computer Vision"),
]
_TOPIC_FIELDS = {"Computer Science", "Engineering", "Mathematics", "Physics and Astronomy", "Decision Sciences"}
_SUBFIELDS = {
    "Artificial Intelligence": "Machine Learning",
    "Computer Vision and Pattern Recognition": "Computer Vision",
    "Computer Graphics and Computer-Aided Design": "Computer Vision",
    "Computational Theory and Mathematics": "Theory & Algorithms",
    "Theoretical Computer Science": "Theory & Algorithms",
    "Information Systems": "Networks & Data Mining",
    "Computer Networks and Communications": "Systems & Software",
    "Hardware and Architecture": "Systems & Software",
    "Software": "Systems & Software",
    "Signal Processing": "Systems & Software",
    "Human-Computer Interaction": "Systems & Software",
    "Computer Science Applications": "Systems & Software",
    "Electrical and Electronic Engineering": "Systems & Software",
    "Control and Systems Engineering": "Robotics & Control",
    "Statistics and Probability": "Mathematics & Statistics",
    "Statistics, Probability and Uncertainty": "Mathematics & Statistics",
    "Economics and Econometrics": "Economics & Finance",
    "Finance": "Economics & Finance",
}
_FIELDS = {
    "Computer Science": "Systems & Software",
    "Mathematics": "Mathematics & Statistics",
    "Physics and Astronomy": "Physics & Astronomy",
    "Earth and Planetary Sciences": "Physics & Astronomy",
    "Materials Science": "Physics & Astronomy",
    "Medicine": "Biology & Medicine",
    "Biochemistry, Genetics and Molecular Biology": "Biology & Medicine",
    "Neuroscience": "Biology & Medicine",
    "Immunology and Microbiology": "Biology & Medicine",
    "Agricultural and Biological Sciences": "Biology & Medicine",
    "Pharmacology, Toxicology and Pharmaceutics": "Biology & Medicine",
    "Health Professions": "Biology & Medicine",
    "Economics, Econometrics and Finance": "Economics & Finance",
}


def domain_of_topic(topic):
    """'Topic | Subfield | Field' (OpenAlex primary topic) -> domain, or None."""
    parts = [p.strip() for p in (topic or "").split("|")] + ["", "", ""]
    name, subfield, field = parts[0], parts[1], parts[2]
    if field in _TOPIC_FIELDS:
        low = name.lower()
        for rx, dom in _TOPIC_RULES:
            if rx.search(low):
                return dom
    if subfield in _SUBFIELDS:
        return _SUBFIELDS[subfield]
    return _FIELDS.get(field)


def weak_label(paper):
    """Domain from the paper's own subject labels: arXiv categories (primary category,
    unless it is a generic ML category and a more specific one is listed), else the
    OpenAlex topic. None when there is no usable label."""
    cats = (paper.get("categories") or "").split()
    doms = [d for d in (domain_of_category(c) for c in cats) if d]
    if doms:
        if doms[0] in _GENERIC:
            for d in doms[1:]:
                if d not in _GENERIC:
                    return d
        return doms[0]
    return domain_of_topic(paper.get("topic"))


def weak_label_index(paper):
    lab = weak_label(paper)
    return DOMAINS.index(lab) if lab else -1


# ------------------------------------------------------------------ classifier
class DomainClassifier:
    """Linear one-vs-rest logistic regression over TF-IDF vectors (any scipy sparse
    or dense matrix whose columns are the index's TF-IDF columns)."""

    def __init__(self, coef=None, bias=None):
        self.coef = coef            # (n_tfidf_columns, n_domains) float32
        self.bias = bias            # (n_domains,) float32

    @property
    def ready(self):
        return self.coef is not None

    def fit(self, X, y, alpha=5e-5, seed=0):
        """X: TF-IDF rows of labelled papers; y: domain indices."""
        from sklearn.linear_model import SGDClassifier

        clf = SGDClassifier(loss="log_loss", alpha=alpha, class_weight="balanced", max_iter=30, tol=None,
                            random_state=seed).fit(X, y)
        coef = np.zeros((X.shape[1], len(DOMAINS)), np.float32)
        bias = np.full(len(DOMAINS), -30.0, np.float32)          # domains never seen: probability ~0
        if len(clf.classes_) == 2:                                # sklearn stores one row for two classes
            rows = np.vstack([-clf.coef_[0], clf.coef_[0]])
            inter = np.array([-clf.intercept_[0], clf.intercept_[0]])
        else:
            rows, inter = clf.coef_, clf.intercept_
        for j, c in enumerate(clf.classes_):
            coef[:, int(c)] = rows[j]
            bias[int(c)] = inter[j]
        self.coef, self.bias = coef, bias
        return self

    def predict_proba(self, X):
        """(n, 13) probabilities; uniform without a model."""
        n = X.shape[0]
        if not self.ready:
            return np.full((n, len(DOMAINS)), 1.0 / len(DOMAINS), np.float32)
        z = np.asarray(X @ self.coef, np.float32) + self.bias
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))           # one-vs-rest, then normalised
        return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)

    def save(self, folder):
        if self.ready:
            np.save(os.path.join(folder, "domain_coef.npy"), self.coef)
            np.save(os.path.join(folder, "domain_bias.npy"), self.bias)
            with open(os.path.join(folder, "domains.json"), "w", encoding="utf-8") as f:
                json.dump(DOMAINS, f)

    @classmethod
    def load(cls, folder, mmap=True):
        path = os.path.join(folder, "domain_coef.npy")
        if not os.path.isfile(path):
            return cls()
        with open(os.path.join(folder, "domains.json"), encoding="utf-8") as f:
            if json.load(f) != DOMAINS:
                raise ValueError("the domain list changed since this index was built: rebuild it")
        return cls(np.load(path, mmap_mode="r" if mmap else None), np.load(os.path.join(folder, "domain_bias.npy")))


def blend_with_labels(probs, label_idx, label_weight=0.6):
    """P(domain | paper): a paper's own category is better evidence than the classifier's
    guess, so it gets `label_weight` of the probability and the classifier the rest
    (cross-listed papers keep some probability elsewhere). label_idx -1 = no label."""
    probs = np.array(probs, np.float32, copy=True)
    has = np.asarray(label_idx) >= 0
    probs[has] *= 1.0 - label_weight
    probs[np.flatnonzero(has), np.asarray(label_idx)[has]] += label_weight
    return probs
