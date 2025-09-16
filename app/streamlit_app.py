#!/usr/bin/env python3
# --- ensure project root is importable so "scripts" can be imported ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pickle, subprocess, time, yaml
import streamlit as st

# ---------- Config ----------
def load_cfg(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_cfg()
INDEX_DIR = CFG.get("paths", {}).get("index_dir") or CFG.get("app", {}).get("index_dir") or "data/index"
CHUNKS_CSV = CFG.get("paths", {}).get("chunks_csv") or CFG.get("app", {}).get("processed_csv") or "data/processed/chunks.csv"
ALPHA = (CFG.get("ir", {}) or {}).get("alpha_bm25", 1.0)

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHUNKS_CSV), exist_ok=True)

# ---------- Helpers ----------
@st.cache_resource
def load_index(pkl_path: str):
    if os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        return payload
    return None

def _lecture_options(df):
    if "doc_id" not in df.columns:
        return ["All lectures"], None
    # show "doc_id — filename.pdf"
    meta = (
        df[["doc_id", "source"]]
        .drop_duplicates()
        .assign(filename=lambda x: x["source"].map(lambda p: os.path.basename(str(p))))
        .sort_values(by="filename")
    )
    labels = ["All lectures"] + [f"{r.doc_id} — {r.filename}" for r in meta.itertuples()]
    return labels, meta

def _filter_results_by_doc(results, df, selected_doc_id):
    if selected_doc_id is None:
        return results
    return [(i, s) for (i, s) in results if str(df.iloc[int(i)]["doc_id"]) == str(selected_doc_id)]

from scripts.generate_questions import generate_from_chunk

# ---------- UI ----------
st.set_page_config(page_title="StudyMate — Automated Exam Question Generator", layout="wide")
st.title("StudyMate — Automated Exam Question Generator")

pkl_path = os.path.join(INDEX_DIR, "hybrid.pkl")
payload = load_index(pkl_path)

if not payload:
    st.warning("No index found. Build it first:")
    st.code("python scripts/ingest.py --input 'data/raw/*.pdf' --output data/processed/chunks.csv\n"
            "python scripts/index_local.py --data data/processed/chunks.csv --index-dir data/index", language="bash")
    st.stop()

retriever = payload["retriever"]
df = payload["df"]

# Controls
query = st.text_input("Query", value="graph traversal vs shortest path")
colA, colB, colC = st.columns(3)
with colA:
    top_k = st.slider("Top-K", 1, 30, 8)
with colB:
    alpha = st.slider("BM25 α (0.0 = embeddings, 1.0 = BM25)", 0.0, 1.0, float(ALPHA), 0.05)
with colC:
    labels, meta = _lecture_options(df)
    choice = st.selectbox("Lecture filter", labels, index=0)
    selected_doc_id = None if choice == "All lectures" else choice.split(" — ", 1)[0]

# Search
if st.button("Search"):
    k_fetch = max(top_k * 3, top_k + 10)  # over-fetch, then filter
    results = retriever.search(query, k=k_fetch, alpha_bm25=alpha)
    results = _filter_results_by_doc(results, df, selected_doc_id)
    results = results[:top_k]

    if not results:
        st.info("No results.")
    else:
        st.subheader("Results")
        for rank, (i, score) in enumerate(results, 1):
            row = df.iloc[int(i)]
            st.markdown(f"**{rank}.** score={score:.4f} • doc_id={row.get('doc_id','?')} • chunk={int(row.get('chunk_idx', -1))}")
            st.write(str(row["text"])[:600] + ("…" if len(str(row["text"])) > 600 else ""))

        # QG button
        if st.button("Generate template questions from top-K"):
            qs = []
            for i, _ in results:
                txt = str(df.iloc[int(i)]["text"])
                qs.extend(generate_from_chunk(txt, n=2))  # 2 per chunk
            st.subheader("Questions")
            for j, (q, ans) in enumerate(qs, 1):
                st.markdown(f"**Q{j}.** {q}")
                if ans:
                    with st.expander("Show answer"):
                        st.write(ans)

