from __future__ import annotations
import os
import uuid
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np
import fitz  
from sentence_transformers import SentenceTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = REPO_ROOT / "data" / "pdfs"
INDEX_DIR = REPO_ROOT / "data" / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_PATH = INDEX_DIR / "faiss.index"
SQLITE_PATH = INDEX_DIR / "meta.sqlite"


EMB_MODEL_NAME = "BAAI/bge-base-en-v1.5"     
BATCH_SIZE = 16                              
CHUNK_SIZE = 3000                             
CHUNK_OVERLAP = 500
MAX_PDFS = None                               
MAX_PAGES_PER_PDF = None                     

def read_pdf_texts(pdf_path: Path) -> List[Tuple[int, str]]:
    texts: List[Tuple[int, str]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            if MAX_PAGES_PER_PDF is not None and i >= MAX_PAGES_PER_PDF:
                break
            text = page.get_text("text")
            if text:
                texts.append((i + 1, text))
    return texts

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def yield_pdf_spans(pdf_path: Path) -> Iterable[Tuple[str, str, str, int]]:
    paper_id = str(uuid.uuid5(uuid.NAMESPACE_URL, pdf_path.name))
    for page_num, page_text in read_pdf_texts(pdf_path):
        for chunk in chunk_text(page_text):
            yield (paper_id, f"page-{page_num}", chunk, page_num)

def ensure_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS spans (
            id TEXT PRIMARY KEY,
            paper_id TEXT,
            section TEXT,
            text TEXT,
            doi TEXT,
            page INTEGER
        );
        """
    )
    conn.commit()
    return conn

def build_or_load_index(dim: int) -> faiss.Index:
    if FAISS_PATH.exists():
        return faiss.read_index(str(FAISS_PATH))
    return faiss.IndexFlatIP(dim)  

def ingest_pdfs() -> None:
    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if MAX_PDFS is not None:
        pdfs = pdfs[:MAX_PDFS]
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Add 1–3 to start.")
        return

    print(f"Found {len(pdfs)} PDFs. Loading model '{EMB_MODEL_NAME}' …")
    model = SentenceTransformer(EMB_MODEL_NAME)

    conn = ensure_sqlite()
    cur = conn.cursor()

    probe_vec = model.encode(["probe"], normalize_embeddings=True)
    dim = int(probe_vec.shape[1])
    index = build_or_load_index(dim)

    total_spans = 0
    batch_texts: List[str] = []
    batch_rows: List[Tuple[str, str, str, str, int]] = []

    def flush_batch():
        nonlocal batch_texts, batch_rows, index, total_spans
        if not batch_texts:
            return
        embs = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)
        index.add(embs)
        faiss.write_index(index, str(FAISS_PATH))  
        cur.executemany(
            "INSERT INTO spans (id, paper_id, section, text, doi, page) VALUES (?, ?, ?, ?, ?, ?)",
            [(sid, pid, sec, txt, None, pg) for (sid, pid, sec, txt, pg) in batch_rows],
        )
        conn.commit()
        total_spans += len(batch_texts)
        print(f"…added {len(batch_texts)} spans (total {total_spans})")
        batch_texts = []
        batch_rows = []

    try:
        for pdf in pdfs:
            print(f"Processing {pdf.name} …")
            for paper_id, section, chunk, page in yield_pdf_spans(pdf):
                span_id = str(uuid.uuid4())
                batch_rows.append((span_id, paper_id, section, chunk, page))
                batch_texts.append(chunk)
                if len(batch_texts) >= 128:   
                    flush_batch()
        
        flush_batch()
    finally:
        conn.close()

    print(f"Ingestion complete → {FAISS_PATH.name}, {SQLITE_PATH.name} | spans: {total_spans}")

if __name__ == "__main__":
    ingest_pdfs()
