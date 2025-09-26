# app.py  (run from repo root)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.services.retrieval import hybrid_retrieve
from src.core.schemas import EvidenceSpan

app = FastAPI(title="Multi-Agent RAG Backend")

class QueryRequest(BaseModel):
    question: str

@app.post("/retrieve", response_model=List[EvidenceSpan])
def retrieve(request: QueryRequest):
    return hybrid_retrieve(request.question)
