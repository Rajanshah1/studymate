#!/usr/bin/env python3
import re, argparse

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def pick_terms(text, max_terms=3):
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")
    seen, terms = set(), []
    for t in sorted(tokens, key=len, reverse=True):
        k = t.lower()
        if k in seen: 
            continue
        if k.endswith(("ly", "ing", "tion", "sion")):
            continue
        seen.add(k); terms.append(t)
        if len(terms) >= max_terms:
            break
    return terms

def cloze(text):
    terms = pick_terms(text, 1)
    if not terms: return None
    t = terms[0]
    return f"Fill in the blank: {text.replace(t, '_____')}", t

def define(text):
    m = re.search(r"\b([A-Z][A-Za-z0-9\- ]{2,20})\s+is\b", text)
    if not m: return None
    subj = m.group(1).strip()
    return f"What is {subj}?", subj

def why(text):
    if "because" in (text or "").lower():
        return "Why does the text say this occurs?", None
    return None

def generate_from_chunk(chunk_text, n=3):
    text = (chunk_text or "").strip()
    sent = SENT_SPLIT.split(text)[0][:300] if text else ""
    qs = []
    for maker in (define, cloze, why):
        q = maker(sent)
        if q: qs.append(q)
        if len(qs) >= n: break
    while len(qs) < n:
        qs.append((f"According to the text, what does this describe?\n“{sent}”", None))
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Chunk text to turn into questions")
    ap.add_argument("-n", type=int, default=3, help="How many questions")
    args = ap.parse_args()
    for i, (q, ans) in enumerate(generate_from_chunk(args.text, args.n), 1):
        print(f"{i}. {q}")
        if ans: print(f"   (answer: {ans})")

if __name__ == "__main__":
    main()

