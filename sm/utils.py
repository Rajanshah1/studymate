import os, re, yaml, hashlib
from typing import List, Dict

def read_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def clean_text(txt: str) -> str:
    txt = txt.replace('\x00', ' ')
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - chunk_overlap)
    return chunks

def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]
