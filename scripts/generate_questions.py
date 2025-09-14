import argparse, pickle, json
from sm.utils import read_config
from sm.nlp import extract_key_terms, qg_templates, qg_transformers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default='data/index/hybrid.pkl')
    ap.add_argument('--topn', type=int, default=5, help='Use top-N retrieved chunks as seeds')
    ap.add_argument('--query', default='', help='Optional query to focus retrieval before QG')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = read_config(args.config)
    with open(args.index, 'rb') as f:
        payload = pickle.load(f)
    retriever = payload['retriever']; df = payload['df']

    # choose seeds
    if args.query:
        hits = retriever.search(args.query, k=args.topn, alpha_bm25=cfg['ir']['alpha_bm25'])
        seeds = [df.iloc[i]['text'] for i,_ in hits]
    else:
        seeds = df['text'].head(args.topn).tolist()

    questions = []
    for s in seeds:
        terms = extract_key_terms(s, cfg['nlp']['spacy_model'])
        # Try transformers QG first (if enabled), otherwise use templates
        generated = []
        if cfg['qg']['use_transformers']:
            generated = qg_transformers(s, cfg['qg']['model_name'])
        if not generated:
            generated = qg_templates(s, terms)
        for g in generated:
            g['terms'] = terms
        questions.extend(generated)

    print(json.dumps(questions, indent=2))

if __name__ == '__main__':
    main()
