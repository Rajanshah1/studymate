import os, argparse, pandas as pd
from pypdf import PdfReader
from sm.utils import clean_text, chunk_text, stable_id, read_config

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or '' for p in reader.pages]
    return '\n'.join(pages)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to PDF or TXT')
    ap.add_argument('--output', default='data/processed/chunks.csv')
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    cfg = read_config(args.config)
    chunk_size = cfg['ir']['chunk_size']
    overlap = cfg['ir']['chunk_overlap']

    if args.input.lower().endswith('.pdf'):
        text = pdf_to_text(args.input)
    else:
        with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

    text = clean_text(text)
    chunks = chunk_text(text, chunk_size, overlap)
    rows = []
    for i, ch in enumerate(chunks):
        rows.append({
            'id': f"{stable_id(ch)}",
            'chunk_idx': i,
            'text': ch
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} chunks -> {args.output}")

if __name__ == '__main__':
    main()
