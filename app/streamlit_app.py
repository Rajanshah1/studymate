#!/usr/bin/env python3
# ---- keep your existing sys.path fix at the very top ----
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re, json
import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

# local utils
from sm.utils import read_config

# QG helpers
from scripts.generate_questions import (
    first_sentence,
    template_fallback,
    load_t5,
    t5_generate_question,
)

# Optional embeddings (for hybrid rerank of a small candidate pool)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(s: str):
    return TOKEN_RE.findall(str(s).lower())

def _minmax_norm(arr):
    arr = np.asarray(arr, dtype="float32")
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx <= mn + 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

@st.cache_resource(show_spinner=False)
def load_df(chunks_csv: str):
    df = pd.read_csv(chunks_csv, dtype=str).fillna("")
    # ensure required columns exist
    for c in ["chunk_id", "doc_id", "chunk_idx", "source", "text"]:
        if c not in df.columns:
            if c == "chunk_id":
                df["chunk_id"] = [f"row-{i}" for i in range(len(df))]
            elif c == "doc_id":
                df["doc_id"] = "doc"
            elif c == "chunk_idx":
                df["chunk_idx"] = np.arange(len(df))
            elif c == "source":
                df["source"] = ""
            elif c == "text":
                df["text"] = ""
    return df

@st.cache_resource(show_spinner=False)
def build_bm25(df):
    corpus_tokens = [tokenize(t) for t in df["text"].tolist()]
    return BM25Okapi(corpus_tokens)

@st.cache_resource(show_spinner=False)
def get_embed_model(name="sentence-transformers/all-MiniLM-L6-v2"):
    if not _HAS_EMB:
        return None
    return SentenceTransformer(name)

class HybridRetriever:
    def __init__(self, df: pd.DataFrame, embed_model_name: str | None):
        self.df = df
        self.bm25 = build_bm25(df)
        self.embed = get_embed_model(embed_model_name) if embed_model_name else None

    def search(self, query: str, k: int = 12, alpha_bm25: float = 1.0):
        qtok = tokenize(query)
        bm_scores_all = self.bm25.get_scores(qtok)
        n = len(self.df)
        if alpha_bm25 >= 0.999 or self.embed is None:
            top = sorted(range(n), key=lambda i: bm_scores_all[i], reverse=True)[:k]
            return [(i, float(bm_scores_all[i])) for i in top]

        pool = max(k * 5, k + 50)
        cand = sorted(range(n), key=lambda i: bm_scores_all[i], reverse=True)[:min(pool, n)]

        qt = self.embed.encode([query], normalize_embeddings=True)[0]
        texts = self.df.iloc[cand]["text"].tolist()
        embs = self.embed.encode(texts, normalize_embeddings=True)
        sims = np.dot(embs, qt)  # cosine (normalized)

        bmn = _minmax_norm([bm_scores_all[i] for i in cand])
        emn = _minmax_norm(sims)
        combo = alpha_bm25 * bmn + (1.0 - alpha_bm25) * emn

        order = list(range(len(cand)))
        order.sort(key=lambda j: combo[j], reverse=True)
        chosen = [cand[j] for j in order[:k]]
        return [(i, float(combo[order[idx]])) for idx, i in enumerate(chosen)]

def _filter_results_by_doc(results, df, doc_id):
    if not doc_id:
        return results
    allowed = set(df.loc[df["doc_id"] == doc_id, "chunk_id"])
    out = []
    for (i, s) in results:
        cid = str(df.iloc[int(i)]["chunk_id"])
        if cid in allowed:
            out.append((i, s))
    return out

# ---------- phrase/overlap/highlight helpers ----------
def _parse_phrase(q: str) -> str | None:
    m = re.search(r'"([^"]+)"', q)
    return m.group(1).strip().lower() if m else None

def _apply_text_filters(results, df, q_tokens, phrase: str | None, min_overlap: int):
    out = []
    qset = set(q_tokens)
    for (i, s) in results:
        txt = str(df.iloc[int(i)]["text"]).lower()
        if phrase and phrase not in txt:
            continue
        if min_overlap > 0:
            overlap = sum(1 for t in qset if t in txt)
            if overlap < min_overlap:
                continue
        out.append((i, s))
    return out

def _highlight(text: str, tokens: list[str], phrase: str | None):
    t = text
    if phrase:
        t = re.sub(re.escape(phrase), f"**{phrase}**", t, flags=re.IGNORECASE)
    for tok in set(tokens):
        t = re.sub(rf"\b{re.escape(tok)}\b", f"**{tok}**", t, flags=re.IGNORECASE)
    return t
# -----------------------------------------------------------

# ---------------------------- UI ----------------------------
st.set_page_config(page_title="StudyMate — Automated Exam Question Generator", layout="wide")
st.title("StudyMate — Automated Exam Question Generator")

# Load config + data
CFG = read_config(os.path.join(ROOT, "config.yaml")) or {}
paths = (CFG.get("paths") or {})
chunks_csv = paths.get("chunks_csv", os.path.join(ROOT, "data/processed/chunks.csv"))
embedding_cfg = (CFG.get("embedding") or {})
embed_model_name = embedding_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists(chunks_csv):
    st.error(f"Chunks file not found: {chunks_csv}")
    st.stop()

df = load_df(chunks_csv)
retriever = HybridRetriever(df, embed_model_name if (embedding_cfg.get("use_faiss", True)) else None)

# Sidebar controls
st.sidebar.header("Search Settings")
ir_cfg = (CFG.get("ir") or {})
top_k = st.sidebar.number_input("Top-K results", 1, 50, int(ir_cfg.get("top_k", 12)), 1)
alpha = st.sidebar.slider("BM25 α (0 = Embeddings, 1 = BM25)", 0.0, 1.0, float(ir_cfg.get("alpha_bm25", 1.0)), 0.05)

# stricter matching toggles
require_phrase = st.sidebar.checkbox("Require exact phrase if you use quotes", value=False)
min_overlap = st.sidebar.slider("Min token overlap", min_value=0, max_value=3, value=0)

# Optional doc filter
doc_opts = ["All documents"] + sorted(df["doc_id"].unique().tolist())
doc_choice = st.sidebar.selectbox("Filter by document", doc_opts, index=0)
selected_doc_id = None if doc_choice == "All documents" else doc_choice

# Main input
query = st.text_input("Enter your query", "")

# ----------------------- State & Actions -----------------------
# init session state
if "results" not in st.session_state:
    st.session_state["results"] = []
if "questions" not in st.session_state:
    st.session_state["questions"] = []
if "last_params" not in st.session_state:
    st.session_state["last_params"] = {}

def run_search_and_store():
    k_fetch = max(int(top_k) * 3, int(top_k) + 10)
    results = retriever.search(query, k=k_fetch, alpha_bm25=float(alpha))
    results = _filter_results_by_doc(results, df, selected_doc_id)

    q_tokens = tokenize(query)
    phrase = _parse_phrase(query) if require_phrase else None
    results = _apply_text_filters(results, df, q_tokens, phrase, int(min_overlap))
    results = results[:int(top_k)]

    st.session_state["results"] = results
    st.session_state["questions"] = []  # reset questions when a new search runs
    st.session_state["last_params"] = {
        "query": query,
        "top_k": int(top_k),
        "alpha": float(alpha),
        "selected_doc_id": selected_doc_id,
        "q_tokens": q_tokens,
        "phrase": phrase,
        "min_overlap": int(min_overlap),
    }

# Search button
if st.button("Search", key="btn_search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        run_search_and_store()

# Render results (persisted)
results = st.session_state["results"]
if results:
    st.subheader("Results")
    params = st.session_state["last_params"]
    q_tokens = params.get("q_tokens", [])
    phrase = params.get("phrase")

    for rank, (i, score) in enumerate(results, 1):
        row = df.iloc[int(i)]
        snippet = str(row["text"])
        if len(snippet) > 600:
            snippet = snippet[:600] + "…"
        shown = _highlight(snippet, q_tokens if int(min_overlap) > 0 else [], phrase)
        st.markdown(
            f"**{rank}.** score={score:.4f} • doc_id={row.get('doc_id','?')} • "
            f"chunk={int(row.get('chunk_idx', -1))}"
        )
        st.write(shown)

    # ------------------ Question Generation ------------------
    st.subheader("Question Generation")

    @st.cache_resource(show_spinner=False)
    def get_t5_cached(model_name: str):
        tok, mdl = load_t5(model_name)
        return tok, mdl

    if st.button("Retrieve & Generate (T5-small)", key="btn_qg"):
        qg_cfg = (CFG.get("qg") or {})
        use_tx = bool(qg_cfg.get("use_transformers", False))
        model_name = qg_cfg.get("model_name", "t5-small")
        max_new = int(qg_cfg.get("max_new_tokens", 48))
        beams = int(qg_cfg.get("num_beams", 4))
        no_repeat = int(qg_cfg.get("no_repeat_ngram_size", 3))

        tok = mdl = None
        if use_tx:
            tok, mdl = get_t5_cached(model_name)

        rows = []
        for i, _ in results[:int(top_k)]:
            r = df.iloc[int(i)]
            text = str(r["text"])
            source = str(r.get("source", ""))
            chunk_id = str(r.get("chunk_id", f"{r.get('doc_id','doc')}-{int(r.get('chunk_idx', -1))}"))
            rows.append({"text": text, "source": source, "chunk_id": chunk_id})

        qs = []
        for idx, r in enumerate(rows, 1):
            ctx = first_sentence(r["text"])
            q = None
            if use_tx and tok and mdl:
                try:
                    q = t5_generate_question(
                        tok, mdl, ctx,
                        max_new=max_new, beams=beams, no_repeat_ngram_size=no_repeat
                    )
                    if not q or q.lower().startswith(("generate", "question")):
                        q = None
                except Exception:
                    q = None
            if not q:
                q = template_fallback(ctx)
            qs.append({
                "id": f"q{idx:04d}",
                "question": q,
                "answer": "",
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "context": ctx
            })

        st.session_state["questions"] = qs

    # Show generated questions (persist across reruns)
    qs = st.session_state["questions"]
    if qs:
        st.success(f"Generated {len(qs)} questions.")
        st.write("---")
        for i, q in enumerate(qs, 1):
            st.markdown(f"**Q{i}.** {q['question']}")
            st.caption(f"Source: {q['source']} • {q['chunk_id']}")
        st.download_button(
            "Download JSON",
            data=json.dumps(qs, ensure_ascii=False, indent=2),
            file_name="questions.json",
            mime="application/json",
        )

