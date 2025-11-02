# src/services/retrieval.py
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
EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"   # base: CPU-friendly
TOP_K_FUSION = 50
TOP_K_FINAL = 12
SIM_THRESHOLD = 0.97  # NEW: tighter near-duplicate cutoff for safety

# Cache model and data
_cached_model = None
_cached_index = None
_cached_spans = None


def get_model() -> SentenceTransformer:
    global _cached_model
    if _cached_model is None:
        print(f"Loading model: {EMB_MODEL_NAME}")
        _cached_model = SentenceTransformer(EMB_MODEL_NAME)
    return _cached_model


def load_faiss_and_sqlite() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not FAISS_PATH.exists() or not SQLITE_PATH.exists():
        raise FileNotFoundError("Run ingestion first: FAISS/SQLite not found.")

    index = faiss.read_index(str(FAISS_PATH))

    conn = sqlite3.connect(str(SQLITE_PATH))
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, paper_id, section, text, doi, page FROM spans"
    ).fetchall()
    conn.close()

    spans: List[Dict[str, Any]] = [
        {"id": rid, "paper_id": pid, "section": sec, "text": txt, "doi": doi, "page": pg}
        for (rid, pid, sec, txt, doi, pg) in rows
    ]
    return index, spans


def get_index_and_spans() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    global _cached_index, _cached_spans
    if _cached_index is None or _cached_spans is None:
        print("Loading FAISS index and spans...")
        _cached_index, _cached_spans = load_faiss_and_sqlite()
        print(f"Loaded {len(_cached_spans)} spans")
    return _cached_index, _cached_spans


def bm25_search(query: str, spans: List[Dict[str, Any]], top_k: int = 50) -> List[str]:
    tokenized_corpus = [s["text"].split() for s in spans]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_q = query.split()
    scores = bm25.get_scores(tokenized_q)
    top_ids = np.argsort(scores)[::-1][:top_k]
    return [spans[i]["id"] for i in top_ids]


def vector_search(query: str, index: faiss.Index, model: SentenceTransformer, spans: List[Dict[str, Any]], top_k: int = 50) -> List[str]:
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    D, I = index.search(q_emb, top_k)
    top_ids = [spans[i]["id"] for i in I[0]]
    return top_ids


# NEW: MMR-style diversity selection
def _mmr_select(ids: List[str], spans_map: Dict[str, Dict[str, Any]], model: SentenceTransformer, top_k: int, lambda_div: float = 0.7) -> List[str]:
    """
    Lightweight MMR-like selection to increase diversity.
    """
    if not ids:
        return []
    
    try:
        texts = [spans_map[sid]["text"] for sid in ids]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

        selected: List[int] = [0]   # start with the first candidate
        cand = list(range(1, len(ids)))

        while len(selected) < min(top_k, len(ids)) and cand:
            # diversity term: farthest from already selected
            best_i, best_score = None, -1e9
            for i in cand:
                # diversity: min cosine similarity to any selected (prefer smaller sim)
                sim_to_sel = max(float(np.dot(embs[i], embs[j])) for j in selected)
                # blended score (favor lower sim)
                score = (1.0 - lambda_div) * (1.0 - len(selected)/len(ids)) + lambda_div * (1 - sim_to_sel)
                if score > best_score:
                    best_score = score
                    best_i = i
            selected.append(best_i)
            cand.remove(best_i)

        return [ids[i] for i in selected]
    except Exception as e:
        print(f"⚠️  MMR selection failed: {e}, falling back to simple selection")
        return ids[:top_k]


def hybrid_retrieve(query: str, top_k_fusion: int = TOP_K_FUSION, top_k_final: int = TOP_K_FINAL) -> List[EvidenceSpan]:
    """
    Enhanced hybrid retrieval with MMR diversity selection.
    """
    index, spans = get_index_and_spans()
    model = get_model()

    bm25_ids = bm25_search(query, spans, top_k=top_k_fusion)
    vec_ids  = vector_search(query, index, model, spans, top_k=top_k_fusion)

    candidate_ids = list(dict.fromkeys(list(bm25_ids) + list(vec_ids)))  # stable union
    spans_map = {s["id"]: s for s in spans if s["id"] in candidate_ids}

    # quick near-duplicate filter
    pruned = []
    seen_texts = set()
    for sid in candidate_ids:
        txt = spans_map[sid]["text"].strip()
        key = txt[:200]
        if key in seen_texts:
            continue
        seen_texts.add(key)
        pruned.append(sid)

    # NEW: Apply MMR diversity selection
    diverse_ids = _mmr_select(pruned, spans_map, model, top_k=top_k_final, lambda_div=0.7)

    results: List[EvidenceSpan] = [EvidenceSpan(**spans_map[sid]) for sid in diverse_ids]
    return results


# NEW: Counter-retrieval for contested claims
_NEGATION_TRIGGERS = [" no ", " not ", " no effect ", " decreases ", " reduction ", " inhibit ", " suppress "]
_ANTONYM = {"increase": "decrease", "decrease": "increase", "improve": "worsen", "worsen": "improve"}

def build_counter_query(claim_text: str) -> str:
    """Build a query to find contradicting evidence."""
    q = claim_text.lower()
    for k, v in _ANTONYM.items():
        if k in q:
            q = q.replace(k, v)
    # gently add negation token
    if " no " not in q and " not " not in q:
        q = "not " + q
    return q

def counter_retrieve(claim_text: str, k: int = TOP_K_FINAL) -> List[EvidenceSpan]:
    """
    Retrieve evidence that might contradict the claim.
    """
    try:
        alt = build_counter_query(claim_text)
        print(f"   🔄 Counter-query: {alt[:80]}...")
        return hybrid_retrieve(alt, top_k_fusion=TOP_K_FUSION, top_k_final=k)
    except Exception as e:
        print(f"   ⚠️ Counter-retrieval failed: {e}")
        return []


if __name__ == "__main__":
    q = "What is the first phase of insulin secretion proportional to?"
    spans = hybrid_retrieve(q)
    print(f"Top {len(spans)} spans:")
    for s in spans:
        print(f"- [{s.paper_id} | {s.section}] {s.text[:120]}…")