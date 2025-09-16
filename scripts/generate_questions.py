#!/usr/bin/env python3
import os, json, argparse, re, math, yaml
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

# ---- optional small hybrid rerank (no FAISS full-scan) ----
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ---- T5-small for QG ----
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(s): return TOKEN_RE.findall(str(s).lower())
def first_sentence(t):
    t = re.split(r'(?<=[.!?])\s+', t.strip(), 1)[0]
    return t.strip()

def load_cfg(path="config.yaml"):
    if path and Path(path).exists():
        with open(path, "r") as f: return yaml.safe_load(f) or {}
    return {}

def bm25_search(df, query, k=20, text_col="text"):
    corpus = df[text_col].fillna("").tolist()
    bm = BM25Okapi([tokenize(x) for x in corpus])
    scores = bm.get_scores(tokenize(query))
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return idx, [float(scores[i]) for i in idx]

def small_hybrid_rerank(df, cand_idx, query, text_col="text"):
    """Rerank a small candidate set by embedding cosine, average with BM25."""
    if not _HAS_EMB:
        return cand_idx, [0.0]*len(cand_idx)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cand_texts = [df.iloc[i][text_col] for i in cand_idx]
    qt = model.encode([query], normalize_embeddings=True)[0]
    embs = model.encode(cand_texts, normalize_embeddings=True)
    sims = np.dot(embs, qt)  # cosine (since normalized)
    # return indices re-ordered by sims desc
    order = list(range(len(cand_idx)))
    order.sort(key=lambda j: sims[j], reverse=True)
    return [cand_idx[j] for j in order], [float(sims[j]) for j in order]

def load_t5(model_name="t5-small"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl

def t5_generate_question(tok, mdl, context, max_new=48, beams=4, no_repeat=3):
    # not fine-tuned for QG, but we nudge with a prompt
    prompt = f"generate question: {context}"
    inpt = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
    out = mdl.generate(
        **inpt,
        max_new_tokens=max_new,
        num_beams=beams,
        no_repeat_ngram_size=no_repeat,
        early_stopping=True,
    )
    q = tok.decode(out[0], skip_special_tokens=True).strip()
    return q

# very small template fallback
def template_fallback(context):
    sent = first_sentence(context)[:240]
    m = re.search(r"\b([A-Z][A-Za-z0-9\- ]{2,40})\s+is\b", sent)
    if m:
        subj = m.group(1).strip()
        return f"What is {subj}?"
    # fallback generic
    return f"According to the text, what does this describe?\n“{sent}”"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="(Optional) path to index; not required for this script")
    ap.add_argument("--chunks", default=None, help="Override chunks CSV")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--query", required=True)
    ap.add_argument("--topn", type=int, default=8)
    ap.add_argument("--pool", type=int, default=80, help="BM25 candidate pool for rerank")
    ap.add_argument("--out", default="-", help="Output JSON path or '-' for stdout")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    chunks_csv = args.chunks or (cfg.get("paths", {}) or {}).get("chunks_csv", "data/processed/chunks.csv")
    qg_cfg = cfg.get("qg") or {}
    model_name = qg_cfg.get("model_name", "t5-small")
    max_new = int(qg_cfg.get("max_new_tokens", 48))
    beams = int(qg_cfg.get("num_beams", 4))
    no_repeat = int(qg_cfg.get("no_repeat_ngram_size", 3))

    df = pd.read_csv(chunks_csv, dtype=str).fillna("")
    # 1) BM25 retrieve a candidate pool
    pool_k = max(args.topn*5, args.pool)
    cand_idx, bm_scores = bm25_search(df, args.query, k=min(pool_k, len(df)))
    # 2) small hybrid rerank (optional)
    if _HAS_EMB:
        cand_idx, emb_scores = small_hybrid_rerank(df, cand_idx, args.query)
    else:
        emb_scores = [0.0]*len(cand_idx)

    # 3) pick topn and form contexts
    chosen = cand_idx[:args.topn]
    tok, mdl = load_t5(model_name)

    out_rows = []
    for j, i in enumerate(chosen, 1):
        row = df.iloc[i]
        text = row["text"]
        context = first_sentence(text)
        try:
            q = t5_generate_question(tok, mdl, context, max_new, beams, no_repeat)
            if not q or q.lower().startswith(("generate", "question")):
                raise ValueError("weak question")
        except Exception:
            q = template_fallback(context)

        out_rows.append({
            "id": f"q{j:04d}",
            "chunk_id": row.get("chunk_id", f"row-{i}"),
            "source": row.get("source", ""),
            "question": q,
            "answer": "",              # Week 3: optional/blank; fill later if desired
            "context": context,
            "query": args.query,
        })

    js = json.dumps(out_rows, ensure_ascii=False, indent=2)
    if args.out == "-" or not args.out:
        print(js)
    else:
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"Wrote {len(out_rows)} questions -> {args.out}")

if __name__ == "__main__":
    main()

