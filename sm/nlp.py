from typing import List, Dict
import re

try:
    import spacy
    _NLP = None
except Exception:
    spacy = None
    _NLP = None

def _load_spacy(model: str = 'en_core_web_sm'):
    global _NLP
    if spacy is None:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load(model)
        except Exception:
            # Model not downloaded; return None to allow graceful fallback
            _NLP = None
    return _NLP

def extract_key_terms(text: str, spacy_model: str = 'en_core_web_sm', top_n: int = 10) -> List[str]:
    nlp = _load_spacy(spacy_model)
    if nlp is None:
        # Simple fallback: heuristics using capitalized words and noun-like tokens
        toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text)
        # naive top terms
        return list(dict.fromkeys([t for t in toks if t[0].isupper()]))[:top_n]
    doc = nlp(text)
    terms = [ent.text for ent in doc.ents]
    # add noun chunks if needed
    terms += [nc.text for nc in doc.noun_chunks if len(nc.text.split()) <= 4]
    # dedupe while preserving order
    seen = set(); out = []
    for t in terms:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:top_n]

# Question generation
def qg_templates(text: str, terms: List[str]) -> List[Dict]:
    qs = []
    for term in terms[:5]:
        qs.append({
            "question": f"What is {term}?",
            "type": "definition",
            "source": text.strip()[:240]
        })
    if len(terms) >= 2:
        qs.append({
            "question": f"Compare {terms[0]} and {terms[1]}. How are they similar and different?",
            "type": "comparison",
            "source": text.strip()[:240]
        })
    qs.append({
        "question": "Explain why this concept is important in the context of the lecture.",
        "type": "explain",
        "source": text.strip()[:240]
    })
    return qs

def qg_transformers(snippet: str, model_name: str = 't5-small'):
    # Lightweight prompt for t5-small; many setups will fallback to templates
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        prompt = "generate question: " + snippet.strip()
        inputs = tok(prompt, return_tensors='pt', max_length=512, truncation=True)
        out = mdl.generate(**inputs, max_new_tokens=48, num_beams=4)
        q = tok.decode(out[0], skip_special_tokens=True)
        return [{"question": q, "type": "generated", "source": snippet.strip()[:240]}]
    except Exception:
        return []
