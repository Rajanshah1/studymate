#!/usr/bin/env python3
import argparse, pickle, json, os
from sm.utils import read_config
from scripts.generate_questions import generate_from_chunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='data/index/hybrid.pkl')
    ap.add_argument('--topn', type=int, default=5, help='Use top-N retrieved chunks as seeds')
    ap.add_argument('--query', default='', help='Optional query to focus retrieval before QG')
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--per-chunk', type=int, default=2, help='Questions per seed chunk')
    args = ap.parse_args()

    cfg = read_config(args.config) or {}
    alpha = (cfg.get('ir', {}) or {}).get('alpha_bm25', 1.0)

    if not os.path.exists(args.index) or os.path.getsize(args.index) == 0:
        raise FileNotFoundError(f"Index not found or empty: {args.index}")

    with open(args.index, 'rb') as f:
        payload = pickle.load(f)
    retriever = payload['retriever']
    df = payload['df']

    # Choose seeds
    if args.query:
        hits = retriever.search(args.query, k=args.topn, alpha_bm25=alpha)
        seeds = [str(df.iloc[int(i)]['text']) for i, _ in hits]
    else:
        seeds = df['text'].head(args.topn).astype(str).tolist()

    # Generate deterministic template questions
    out = []
    for s in seeds:
        for q, ans in generate_from_chunk(s, n=args.per_chunk):
            out.append({"question": q, "answer": ans})

    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
