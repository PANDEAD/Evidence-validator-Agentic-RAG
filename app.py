import sys
import traceback
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Set environment variables BEFORE any torch imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from src.agents.orchestrator import run_full_pipeline
from src.services.retrieval import hybrid_retrieve, get_model, get_index_and_spans
from src.core.schemas import EvidenceSpan, RunState, Claim, Verdict
from src.services.validation import validate_claims_against_spans

app = FastAPI(title="Multi-Agent RAG Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RunRequest(BaseModel):
    question: str
    max_claims: int = 2
    retrieval_k: int = 12

# Global flag to prevent multiple model loads
_models_loaded = False

class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 12

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    try:
        spans = hybrid_retrieve(req.question, top_k_final=req.top_k)
        # Return plain dicts for the UI
        return [s.model_dump() for s in spans]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    global _models_loaded
    
    if _models_loaded:
        print("⚠️  Models already loaded, skipping...")
        return
    
    print("🚀 Preloading model and index…")
    try:
        print("   Loading retrieval model...")
        _ = get_model()
        print("   ✓ Retrieval model loaded")
        
        print("   Loading index and spans...")
        _ = get_index_and_spans()
        print("   ✓ Index loaded")
        
        _models_loaded = True
        print("✅ Preload complete")
    except Exception as e:
        print(f"❌ Preload failed: {e}")
        traceback.print_exc()
        # Don't raise - allow app to start, models will load on first request

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _models_loaded,
        "environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            "OMP_NUM_THREADS": os.environ.get('OMP_NUM_THREADS'),
        }
    }

@app.post("/run")
def run_pipeline(req: RunRequest):
    print(f"\n{'='*60}")
    print(f"📥 Received request: {req.question}")
    print(f"   max_claims={req.max_claims}, retrieval_k={req.retrieval_k}")
    print(f"{'='*60}\n")
    
    try:
        print("🔍 Step 1: Starting pipeline...")
        result = run_full_pipeline(
            question=req.question,
            max_claims=req.max_claims,
            retrieval_k=req.retrieval_k
        )
        print("✅ Step 1: Pipeline completed successfully")
        
        print("📦 Step 2: Converting result to dict...")
        result_dict = result.dict()
        print("✅ Step 2: Conversion complete")
        
        print(f"📊 Result summary:")
        print(f"   Status: {result_dict.get('status')}")
        print(f"   Evidence spans: {len(result_dict.get('evidence_spans', []))}")
        print(f"   Claims: {len(result_dict.get('claims', []))}")
        print(f"   Verdicts: {len(result_dict.get('verdicts', []))}")
        
        return result_dict
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"\n{'='*60}")
        print(f"❌ ERROR: {error_type}")
        print(f"❌ Message: {error_msg}")
        print(f"{'='*60}")
        print(f"Stack trace:\n{error_trace}")
        print(f"{'='*60}\n")
        
        # Return error details to frontend
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_type,
                "message": error_msg,
                "trace": error_trace
            }
        )

if __name__ == "__main__":
    import uvicorn
    # Force single worker on macOS
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)