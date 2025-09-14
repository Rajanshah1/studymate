import streamlit as st
import pandas as pd
import os, io, pickle, json, tempfile
from pypdf import PdfReader
from sm.utils import clean_text, chunk_text, read_config
from sm.ir import HybridRetriever
from sm.nlp import extract_key_terms, qg_templates, qg_transformers
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='StudyMate', layout='wide')
st.title('📚 StudyMate — Automated Exam Question Generator')

cfg = read_config('config.yaml')

# Upload
st.header('1) Upload Lecture Notes (PDF or TXT)')
uploaded = st.file_uploader('Upload a PDF or .txt', type=['pdf','txt'])

if uploaded is not None:
    if uploaded.name.lower().endswith('.pdf'):
        reader = PdfReader(io.BytesIO(uploaded.read()))
        pages = [p.extract_text() or '' for p in reader.pages]
        text = '\n'.join(pages)
    else:
        text = uploaded.read().decode('utf-8', errors='ignore')
    text = clean_text(text)

    st.write('Characters:', len(text))
    # Chunk
    chunks = chunk_text(text, cfg['ir']['chunk_size'], cfg['ir']['chunk_overlap'])
    st.success(f'Created {len(chunks)} chunks')

    # Build in-memory index
    with st.spinner('Embedding & indexing...'):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        retriever = HybridRetriever(chunks, embedder=embedder)

    st.header('2) Search / Generate Questions')
    q = st.text_input('Focus query (optional)', placeholder='e.g., backpropagation vs gradient descent')
    topk = st.slider('Top-K seeds', 3, 20, 8)
    alpha = st.slider('Hybrid weight (BM25 α)', 0.0, 1.0, float(cfg['ir']['alpha_bm25']))

    if st.button('🔎 Retrieve & Generate'):
        hits = retriever.search(q if q else 'exam important', k=topk, alpha_bm25=alpha)
        seed_chunks = [chunks[i] for i,_ in hits] if hits else chunks[:topk]
        rows = []
        for s in seed_chunks:
            terms = extract_key_terms(s, cfg['nlp']['spacy_model'])
            generated = []
            if cfg['qg']['use_transformers']:
                generated = qg_transformers(s, cfg['qg']['model_name'])
            if not generated:
                generated = qg_templates(s, terms)
            for g in generated:
                g['terms'] = terms
            rows.extend(generated)

        df = pd.DataFrame(rows)
        st.subheader('Questions')
        st.dataframe(df[['question','type','source']])
        st.download_button('⬇️ Download CSV', data=df.to_csv(index=False), file_name='mock_quiz.csv', mime='text/csv')
        st.download_button('⬇️ Download JSON', data=df.to_json(orient='records', indent=2), file_name='mock_quiz.json', mime='application/json')

st.caption('Hybrid BM25 + embeddings • spaCy concept extraction • T5-small QG fallback to templates')
