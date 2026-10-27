"""Check how well everyday-language searches work, live against OpenAlex.

Each case is a query typed the way a person would, and a pattern that the titles of
fitting papers match (a rough, hand-written check, not a trained label). For each
query a fresh, empty memory is used, so every search goes online, and the script
prints how many of the top 10 titles match and where the first match is.

  OPENALEX_API_KEY=... python scripts/check_queries.py            with query understanding
  OPENALEX_API_KEY=... python scripts/check_queries.py --exact    words as typed (the old behaviour)
  OPENALEX_API_KEY=... python scripts/check_queries.py --saver    saver mode: free sources first, no extra searches

About 3 OpenAlex searches per query ($0.003 with a key); 1 in saver mode. The script
prints what the OpenAlex answers say it cost.
"""
import argparse
import os
import re
import statistics
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from engine.lexicon import Lexicon, lexicon_dir          # noqa: E402
from engine.pipeline import SearchPipeline              # noqa: E402
from engine.sources import ScholarlyClient              # noqa: E402

CASES = [
    ('I want research papers on how to make neural networks smaller so they run on phones', 'compress|prun|quantiz|mobile|efficient|distill|lightweight|compact|binar|low-bit|edge'),
    ('attention is all you need', '^attention is all you need$'),
    ('nueral netwrok pruning', 'prun'),
    ('papers about detecting fake news using AI', 'fake news|misinformation|disinformation|rumou?r'),
    ('how does climate change affect crop yields', 'crop|yield|agricultur|wheat|maize|rice'),
    ('can machine learning predict heart disease', 'heart|cardi|coronary'),
    ('what causes alzheimers disease', 'alzheimer|dementia'),
    ('deep residual learning for image recognition', '^deep residual learning for image recognition$'),
    ('BERT pre-training of deep bidirectional transformers', '^bert: pre-training of deep bidirectional transformers'),
    ('reinforcment lerning for robot control', 'reinforcement|robot'),
    ('teaching computers to recognise human emotions from faces', 'emotion|facial|expression|affect'),
    ('how do vaccines train the immune system', 'vaccin|immun'),
    ('using satellites to watch deforestation', 'deforest|forest|tree cover|land cover'),
    ('recent work on diffusion models for image generation', 'diffusion'),
    ('generative adversarial networks', 'adversarial|gan'),
    ('graph neural networks', 'graph'),
    ('quantum error correction', 'error|correct|code|fault'),
    ('i need studies showing that social media harms teenage mental health', 'social media|adolescen|teen|youth|young|mental|depress|wellbeing|well-being'),
    ('self-driving cars seeing pedestrians at night', 'pedestrian|night|low.light|thermal|infrared|dark'),
    ('why do lithium batteries lose capacity over time', 'batter|degrad|aging|ageing|lithium|capacity fade'),
    ('imagenet classification with deep convolutional neural networks', '^imagenet classification with deep convolutional neural networks$'),
    ('computer programs that write code', 'code|program'),
    ('materials that capture carbon dioxide from air', 'carbon|co2|capture|adsorb'),
    ('translating languages without parallel data', 'unsupervised|translation|monolingual'),
    ('machne lerning for protien structure predicton', 'protein|structure|fold'),
    ('show me the best papers about crispr gene editing', 'crispr|cas9|genome|gene editing'),
    ('How does reinforcement learning improve autonomous vehicle navigation at night?', 'reinforcement|navigat|autonomous|driving|night'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exact", action="store_true", help="search the words as typed")
    ap.add_argument("--saver", action="store_true", help="saver mode, as when the day's budget is nearly used up")
    args = ap.parse_args()
    os.environ["PAPERSCOPE_SAVER"] = "on" if args.saver else "off"
    models = os.path.join(HERE, "data", "models")
    lex = Lexicon.load(lexicon_dir(models))
    p10s, firsts, times, spent = [], [], [], 0.0
    for q, pattern in CASES:
        pipe = SearchPipeline(models_dir=models, client=ScholarlyClient(cache_file=None), lexicon=lex)
        t0 = time.perf_counter()
        res = pipe.search(q, per_page=10, exact=args.exact)
        times.append(time.perf_counter() - t0)
        spent += pipe.client.budget()["spent_usd"]
        rx = re.compile(pattern, re.I)
        hits = [bool(rx.search(" ".join(p["title"].lower().split()))) for p in res["results"]]
        first = next((i + 1 for i, h in enumerate(hits) if h), None)
        p10s.append(sum(hits) / 10)
        firsts.append(first)
        print(f"{sum(hits):2d}/10  first match: {first or '-':>2}  {times[-1]:.1f} s  {q}")
    print(f"\nmatching titles in the top 10: {statistics.mean(p10s):.0%}; first result matches in "
          f"{sum(1 for f in firsts if f == 1)} of {len(CASES)} queries; median time {statistics.median(times):.1f} s; "
          f"OpenAlex cost ${spent:.3f}")


if __name__ == "__main__":
    main()
