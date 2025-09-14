from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize

try:
    import faiss  # type: ignore
except Exception as e:
    faiss = None

class HybridRetriever:
    def __init__(self, texts: List[str], embedder=None):
        self.texts = texts
        # BM25
        self.tokenized_corpus = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        # Embeddings
        self.embedder = embedder
        self.faiss_index = None
        self.embeddings = None
        if embedder is not None and faiss is not None:
            self._build_faiss()

    def _build_faiss(self):
        vecs = self.embedder.encode(self.texts, show_progress_bar=False, convert_to_numpy=True)
        vecs = normalize(vecs)
        self.embeddings = vecs.astype('float32')
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings)

    def search(self, query: str, k: int = 10, alpha_bm25: float = 0.5) -> List[Tuple[int, float]]:
        # BM25
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()), dtype=float)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / (bm25_scores.max() + 1e-9)
        # Embedding sim
        emb_scores = np.zeros(len(self.texts), dtype=float)
        if self.embedder is not None and self.faiss_index is not None:
            qv = self.embedder.encode([query], convert_to_numpy=True)
            qv = normalize(qv).astype('float32')
            sim, idx = self.faiss_index.search(qv, len(self.texts))
            emb_scores[idx[0]] = (sim[0] - sim[0].min()) / (sim[0].ptp() + 1e-9) if sim[0].ptp() > 0 else sim[0]
        # Hybrid
        scores = alpha_bm25 * bm25_scores + (1 - alpha_bm25) * emb_scores
        top_idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[int(i)])) for i in top_idx if scores[int(i)] > 0]
