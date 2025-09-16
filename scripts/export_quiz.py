#!/usr/bin/env python3
import argparse, json, csv, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-json", required=True)
    ap.add_argument("--out-prefix", required=True, help="prefix without extension")
    args = ap.parse_args()

    with open(args.questions_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    # CSV
    csv_path = args.out_prefix + ".csv"
    Path(os.path.dirname(csv_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","question","answer","source","chunk_id","context"])
        for r in rows:
            w.writerow([r.get("id",""), r.get("question",""), r.get("answer",""),
                        r.get("source",""), r.get("chunk_id",""), r.get("context","")])

    # Markdown
    md_path = args.out_prefix + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Mock Quiz\n\n")
        for i, r in enumerate(rows, 1):
            f.write(f"**Q{i}.** {r.get('question','')}\n\n")
            ans = r.get("answer","").strip()
            if ans:
                f.write(f"<details><summary>Answer</summary>\n\n{ans}\n\n</details>\n\n")
            src = r.get("source","")
            if src:
                f.write(f"<sub>Source: {src} (chunk {r.get('chunk_id','')})</sub>\n\n")

    print(f"Wrote {csv_path} and {md_path}")

if __name__ == "__main__":
    main()
