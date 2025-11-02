# src/core/schemas.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# ---- Thresholds
TAU_SUPPORT: float = 0.65
TAU_CONTRADICT: float = 0.40

class Label(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTESTED = "CONTESTED"
    UNCERTAIN = "UNCERTAIN"

class EvidenceSpan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    id: str
    paper_id: str
    section: Optional[str] = None
    text: str = Field(..., min_length=10)
    doi: Optional[str] = None
    score: Optional[float] = None
    page: Optional[int] = None

class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(..., min_length=10, max_length=600)
    evidence_ids: List[str] = Field(..., min_items=2, max_items=8)

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    label: Label
    support_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)
    rationale: Optional[str] = None

class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: List[str] = Field(..., min_items=1, max_items=3)
    section_bias: Optional[List[str]] = None
    k_cap: int = Field(ge=10, le=100, default=50)

class RunCaps(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_revision: int = 2
    max_counter_retrieval: int = 1  # NEW: allow one contradiction-aware re-retrieval

class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    plan: Optional[RetrievalPlan] = None
    evidence_spans: List[EvidenceSpan] = []
    claims: List[Claim] = []
    verdicts: List[Verdict] = []
    logs: List[Dict[str, Any]] = []
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    final_answer: Optional[str] = None  # NEW: human-readable conclusion
    caps: RunCaps = Field(default_factory=RunCaps)
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "tau_support": TAU_SUPPORT,
        "tau_contradict": TAU_CONTRADICT,
    })

if __name__ == "__main__":
    es = EvidenceSpan(id="s1", paper_id="p1", text="This is a sample evidence span text.")
    print("Schemas OK ✅")