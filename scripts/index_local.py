import os, argparse, pickle, pandas as pd
from sentence_transformers import SentenceTransformer
from sm.ir import HybridRetriever
from sm.utils import read_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/chunks.csv')
    ap.add_argument('--index-dir', default='data/index')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = read_config(args.config)
    df = pd.read_csv(args.data)
    texts = df['text'].fillna('').astype(str).tolist()
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    retriever = HybridRetriever(texts, embedder=embedder)
    os.makedirs(args.index_dir, exist_ok=True)

    # Save pickle with DataFrame and retriever essentials (BM25 tokenized corpus) for lightweight reload
    payload = {
        'df': df,
        'texts': texts,
        'retriever': retriever
    }
    with open(os.path.join(args.index_dir, 'hybrid.pkl'), 'wb') as f:
        pickle.dump(payload, f)
    print(f"Built hybrid index for {len(texts)} chunks -> {args.index_dir}/hybrid.pkl")

if __name__ == '__main__':
    main()
