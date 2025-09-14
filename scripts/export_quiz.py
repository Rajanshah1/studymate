import argparse, json, csv, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions-json', required=True, help='JSON file or raw JSON string')
    ap.add_argument('--out-prefix', default='data/processed/mock_quiz')
    args = ap.parse_args()

    # Read JSON
    if os.path.exists(args.questions_json):
        with open(args.questions_json, 'r') as f:
            qs = json.load(f)
    else:
        qs = json.loads(args.questions_json)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # CSV
    csv_path = args.out_prefix + '.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['question','type','source'])
        w.writeheader()
        for q in qs:
            w.writerow({'question': q.get('question',''), 'type': q.get('type',''), 'source': q.get('source','')})
    # Markdown
    md_path = args.out_prefix + '.md'
    with open(md_path, 'w') as f:
        f.write('# Mock Quiz\n\n')
        for i, q in enumerate(qs, 1):
            f.write(f'{i}. {q.get("question","")}\n\n')
    print(f"Wrote {csv_path} and {md_path}")

if __name__ == '__main__':
    main()
