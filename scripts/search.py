#!/usr/bin/env python3
import argparse, pickle, json, os
from sm.utils import read_config

def _resolve_cols(df):
    """Map to actual column names (case-insensitive) with fallbacks."""
    cols = {c.lower(): c for c in df.columns}

    def pick(*candidates):
        for c in candidates:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    col_id        = pick("id", "doc_id", "source", "filename", "file")
    col_chunk_idx = pick("chunk_idx", "chunk_id", "chunk", "idx")
    col_text      = pick("text", "content", "chunk_text", "body")

    if col_text is None:
        raise ValueError(f"Couldn't find text column in {list(df.columns)}")

    if col_id is None:
        df["__doc_id__"] = "doc"
        col_id = "__doc_id__"
    if col_chunk_idx is None:
        df["__chunk_idx__"] = range(len(df))
        col_chunk_idx = "__chunk_idx__"

    return col_id, col_chunk_idx, col_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-q', '--query', required=True)
    ap.add_argument('-k', '--topk', type=int, default=5)
    ap.add_argument('--index', default='data/index/hybrid.pkl')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = read_config(args.config) or {}
    alpha = (cfg.get("ir", {}) or {}).get("alpha_bm25", 1.0)

    if not os.path.exists(args.index) or os.path.getsize(args.index) == 0:
        raise FileNotFoundError(f"Index not found or empty: {args.index}")

    with open(args.index, 'rb') as f:
        payload = pickle.load(f)
    retriever = payload['retriever']
    df = payload['df']

    ID_COL, CHUNK_COL, TEXT_COL = _resolve_cols(df)

    results = retriever.search(args.query, k=args.topk, alpha_bm25=alpha)
    out = []
    for idx, score in results:
        row = df.iloc[int(idx)]
        out.append({
            "id":        str(row[ID_COL]),
            "chunk_idx": int(row[CHUNK_COL]),
            "score":     float(score),
            "text":      str(row[TEXT_COL])[:300]
        })
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()

