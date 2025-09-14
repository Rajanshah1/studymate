#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(s: str):
    return TOKEN_RE.findall(str(s).lower())

def choose_col(existing, candidates, required=False, what=""):
    for c in candidates:
        if c in existing:
            return c
    if required:
        raise ValueError(
            f"Required column not found for {what}. "
            f"Looked for any of: {', '.join(candidates)}. "
            f"Available: {list(existing)}"
        )
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required=True, help="Search query")
    ap.add_argument("-k", "--topk", type=int, default=5, help="Top-K results")
    ap.add_argument("--config", default="config.yaml", help="YAML config path")
    ap.add_argument("--chunks", default=None, help="Override chunks CSV path")
    args = ap.parse_args()

    # Load config (optional)
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    chunks_csv = args.chunks or (cfg.get("paths", {}) or {}).get("chunks_csv", "data/processed/chunks.csv")
    chunks_csv = str(chunks_csv)

    if not Path(chunks_csv).exists():
        raise FileNotFoundError(f"Chunks CSV not found: {chunks_csv}")

    # Read as strings to avoid dtype surprises; fillna afterwards
    df = pd.read_csv(chunks_csv, dtype=str).fillna("")

    # Resolve column names
    text_col = choose_col(df.columns, ["text", "content", "chunk", "body"], required=True, what="chunk text")
    id_col   = choose_col(df.columns, ["chunk_id", "id", "stable_id"], required=False, what="chunk id")
    src_col  = choose_col(df.columns, ["source", "file", "path"], required=False, what="source path")

    # Build BM25 over the text column
    corpus = df[text_col].tolist()
    corpus_tokens = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(corpus_tokens)

    # Score query
    q_tokens = tokenize(args.query)
    scores = bm25.get_scores(q_tokens)

    # Choose top-k (guard if k > n)
    n = len(df)
    k = max(0, min(args.topk, n))
    top_idx = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]

    for rank, i in enumerate(top_idx, 1):
        row = df.iloc[i]
        score = float(scores[i])

        # Robust fields
        cid = row[id_col] if id_col else (row.name if hasattr(row, "name") else "n/a")
        src = row[src_col] if src_col else "unknown"

        text = row[text_col]
        snippet = text.replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:240] + "…"

        print(f"{rank}. [{score:.3f}] {cid} — {src}")
        print(f"    {snippet}\n")

if __name__ == "__main__":
    main()

