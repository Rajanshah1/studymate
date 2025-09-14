import re, pandas as pd, yaml, streamlit as st
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="StudyMate — Week 1 BM25", layout="wide")

@st.cache_resource
def load_index(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    chunks_csv = cfg.get("paths", {}).get("chunks_csv", "data/processed/chunks.csv")
    df = pd.read_csv(chunks_csv)
    def tok(s): return re.findall(r"[a-z0-9]+", str(s).lower())
    docs = df["text"].fillna("").tolist()
    tokens = [tok(t) for t in docs]
    bm25 = BM25Okapi(tokens)
    return df, bm25, tok, cfg

df, bm25, tok, cfg = load_index()

st.sidebar.header("Config")
st.sidebar.write(f"**chunk_size:** {cfg.get('ir',{}).get('chunk_size',500)}")
st.sidebar.write(f"**alpha_bm25:** {cfg.get('ir',{}).get('alpha_bm25',1.0)} (BM25-only)")
st.sidebar.write(f"**chunks_csv:** {cfg.get('paths',{}).get('chunks_csv')}")

st.title("StudyMate — Week 1: BM25 Search")
q = st.text_input("Search your lecture chunks (BM25-only):", value="what is gradient descent?")
k = st.slider("Top-K", 1, 20, 5)

if q.strip():
    scores = bm25.get_scores(tok(q))
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    st.caption(f"Results from **{len(df)}** chunks")
    for rank, i in enumerate(idxs, 1):
        row = df.iloc[i]
        st.markdown(f"**{rank}.** `{row['chunk_id']}` — *{row['source']}*  \nScore: `{scores[i]:.3f}`")
        st.write(row["text"][:600])
        st.divider()
else:
    st.info("Type a query to search.")

