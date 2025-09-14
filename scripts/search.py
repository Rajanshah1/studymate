import argparse, pickle, json
from sm.utils import read_config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-q', '--query', required=True)
    ap.add_argument('-k', '--topk', type=int, default=5)
    ap.add_argument('--index', default='data/index/hybrid.pkl')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = read_config(args.config)
    with open(args.index, 'rb') as f:
        payload = pickle.load(f)
    retriever = payload['retriever']
    df = payload['df']

    results = retriever.search(args.query, k=args.topk, alpha_bm25=cfg['ir']['alpha_bm25'])
    out = []
    for idx, score in results:
        row = df.iloc[int(idx)]
        out.append({'id': row['id'], 'chunk_idx': int(row['chunk_idx']), 'score': float(score), 'text': row['text'][:300]})
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
