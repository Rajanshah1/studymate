#!/usr/bin/env python3
import os
import argparse
import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:
    raise SystemExit("Missing dependency: pypdf. Install with `pip install pypdf`.")

# Local utilities
from sm.utils import clean_text, chunk_text, stable_id, read_config


def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        # extract_text() may return None; coalesce and strip
        t = (p.extract_text() or "").strip()
        parts.append(t)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to PDF or TXT")
    ap.add_argument("--output", default="data/processed/chunks.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Drop chunks shorter than this number of tokens (default: 0, keep all)")
    args = ap.parse_args()

    # Load config with sane fallbacks
    cfg = read_config(args.config) if args.config and os.path.exists(args.config) else {}
    ir_cfg = (cfg.get("ir") or {})
    chunk_size = int(ir_cfg.get("chunk_size", 800))
    overlap = int(ir_cfg.get("chunk_overlap", 120))

    # Read source text
    src_path = os.path.abspath(args.input)
    if args.input.lower().endswith(".pdf"):
        text = pdf_to_text(src_path)
    else:
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    # Clean + chunk
    text = clean_text(text or "")
    chunks = chunk_text(text, chunk_size, overlap)

    # Stable IDs: one per document, chunk_id per chunk
    doc_id = stable_id(src_path)  # deterministic per file path
    rows = []
    for i, ch in enumerate(chunks):
        rows.append({
            "chunk_id": f"{doc_id}-{i:04d}",
            "doc_id": doc_id,
            "chunk_idx": i,
            "source": src_path,
            "text": ch,
        })

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Apply min-tokens filter if set
    if args.min_tokens > 0:
        before = len(df)
        df = df[df["text"].fillna("").str.split().str.len() >= args.min_tokens]
        after = len(df)
        print(f"Filtered chunks by --min-tokens={args.min_tokens}: kept {after}/{before}")

    # Write CSV
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} chunks -> {args.output}")


if __name__ == "__main__":
    main()

