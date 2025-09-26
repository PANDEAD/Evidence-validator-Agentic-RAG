from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.core.schemas import EvidenceSpan

# ---- Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "data" / "indices"
FAISS_PATH = INDEX_DIR / "faiss.index"
SQLITE_PATH = INDEX_DIR / "meta.sqlite"

# ---- Embedding model
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"   # use base for laptop stability
TOP_K_FUSION = 50
TOP_K_FINAL = 12
SIM_THRESHOLD = 0.95  # cosine sim threshold for dedup


# --------------------
# Utility functions
# --------------------
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------
# Core retrieval
# --------------------
def load_faiss_and_sqlite() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load FAISS + SQLite spans into memory."""
    if not FAISS_PATH.exists() or not SQLITE_PATH.exists():
        raise FileNotFoundError("Run ingestion first: FAISS/SQLite not found.")

    # Load FAISS
    index = faiss.read_index(str(FAISS_PATH))

    # Load spans from SQLite
    conn = sqlite3.connect(str(SQLITE_PATH))
    cur = conn.cursor()
    rows = cur.execute("SELECT id, paper_id, section, text, doi, page FROM spans").fetchall()
    conn.close()

    spans: List[Dict[str, Any]] = [
        {
            "id": rid,
            "paper_id": pid,
            "section": sec,
            "text": txt,
            "doi": doi,
            "page": pg,
        }
        for (rid, pid, sec, txt, doi, pg) in rows
    ]

    return index, spans


def bm25_search(query: str, spans: List[Dict[str, Any]], top_k: int = 50) -> List[str]:
    """Return top span_ids via BM25 lexical matching."""
    tokenized_corpus = [s["text"].split() for s in spans]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    top_ids = np.argsort(scores)[::-1][:top_k]
    return [spans[i]["id"] for i in top_ids]


def vector_search(query: str, index: faiss.Index, model: SentenceTransformer, spans: List[Dict[str, Any]], top_k: int = 50) -> List[str]:
    """Return top span_ids via FAISS vector search."""
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    D, I = index.search(q_emb, top_k)
    top_ids = [spans[i]["id"] for i in I[0]]
    return top_ids


def hybrid_retrieve(query: str, top_k_fusion: int = TOP_K_FUSION, top_k_final: int = TOP_K_FINAL) -> List[EvidenceSpan]:
    """Hybrid retrieval: BM25 ∪ FAISS → rerank → dedup/diversity → top_k_final."""
    index, spans = load_faiss_and_sqlite()
    model = SentenceTransformer(EMB_MODEL_NAME)

    # 1. Run BM25 + Vector search
    bm25_ids = bm25_search(query, spans, top_k=top_k_fusion)
    vec_ids = vector_search(query, index, model, spans, top_k=top_k_fusion)

    # 2. Union results
    candidate_ids = list(set(bm25_ids) | set(vec_ids))
    id_to_span = {s["id"]: s for s in spans if s["id"] in candidate_ids}

    # 3. Build embeddings for dedup/diversity
    cand_texts = [id_to_span[sid]["text"] for sid in candidate_ids]
    cand_embs = model.encode(cand_texts, normalize_embeddings=True)
    cand_embs = np.asarray(cand_embs, dtype=np.float32)

    # 4. Deduplicate near-identical chunks
    selected: List[str] = []
    selected_embs: List[np.ndarray] = []
    for sid, emb in zip(candidate_ids, cand_embs):
        if all(cosine_sim(emb, e2) < SIM_THRESHOLD for e2 in selected_embs):
            selected.append(sid)
            selected_embs.append(emb)
        if len(selected) >= top_k_final:
            break

    # 5. Convert to EvidenceSpan models
    results: List[EvidenceSpan] = []
    for sid in selected:
        s = id_to_span[sid]
        results.append(EvidenceSpan(**s))

    return results


if __name__ == "__main__":
    q = "What is the first phase of insulin secretion proportional to?"
    spans = hybrid_retrieve(q)
    print(f"Top {len(spans)} spans:")
    for s in spans:
        print(f"- [{s.paper_id} | {s.section}] {s.text[:120]}…")
