# src/agents/orchestrator.py
from typing import List, Dict, Any
from datetime import datetime
import traceback

from src.core.schemas import Claim, Verdict, RunState, Label, EvidenceSpan
from src.services.retrieval import hybrid_retrieve
from src.agents.claim_synthesizer import synthesize_claims
from src.services.validation import validate_claims_against_spans
from src.services.answer import compose_final_answer

class PipelineOrchestrator:
    def __init__(self):
        self.run_history: List[RunState] = []

    def run_pipeline(self, question: str, max_claims: int = 2, retrieval_k: int = 12) -> RunState:
        print(f"\n{'='*80}")
        print(f"Orchestrator: Starting ENHANCED pipeline")
        print(f"   Question: '{question}'")
        print(f"   Config: max_claims={max_claims}, retrieval_k={retrieval_k}")
        print(f"{'='*80}\n")
        
        run_state = RunState(
            question=question,
            evidence_spans=[],
            claims=[],
            verdicts=[],
            logs=[],
            status="running",
            started_at=datetime.now()
        )

        try:
            print(" Phase 1: INITIAL RETRIEVAL")
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "retrieval",
                "message": "Starting evidence retrieval with MMR diversity"
            })
            
            spans = hybrid_retrieve(question, top_k_final=retrieval_k)
            run_state.evidence_spans = spans
            
            print(f"   ✓ Retrieved {len(spans)} diverse evidence spans")
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "retrieval",
                "message": f"Retrieved {len(spans)} evidence spans"
            })

            if not spans:
                print("    No evidence found")
                run_state.status = "completed_no_evidence"
                run_state.final_answer = "No relevant evidence retrieved for the question."
                run_state.completed_at = datetime.now()
                self.run_history.append(run_state)
                return run_state

            print("\n Phase 2: CLAIM SYNTHESIS (LLM+fallback)")
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "synthesis",
                "message": "Synthesizing claims from evidence (LLM-first, heuristic fallback)"
            })
            
            claims = synthesize_claims(spans, question, max_claims=max_claims)
            run_state.claims = claims
            
            print(f"   ✓ Generated {len(claims)} claims")
            for i, claim in enumerate(claims, 1):
                print(f"     Claim {i}: {claim.text[:80]}...")
                print(f"              Evidence IDs: {len(claim.evidence_ids)} spans")
            
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "synthesis",
                "message": f"Generated {len(claims)} claims"
            })

            if not claims:
                print("    No claims generated")
                run_state.status = "completed_no_claims"
                run_state.final_answer = "Evidence found, but insufficient to form reliable, testable claims."
                run_state.completed_at = datetime.now()
                self.run_history.append(run_state)
                return run_state

            print("\n Phase 3: VALIDATION")
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "validation",
                "message": "Validating claims with enhanced NLI"
            })
            
            verdicts = validate_claims_against_spans(
                claims,
                run_state.evidence_spans,
                tau_support=run_state.thresholds["tau_support"],
                tau_contradict=run_state.thresholds["tau_contradict"],
            )
            run_state.verdicts = verdicts
            
            print(f"\n   ✓ Validation complete:")
            for i, verdict in enumerate(verdicts, 1):
                print(f"     Verdict {i}: {verdict.label.value} "
                      f"(support={verdict.support_score:.3f}, contra={verdict.contradiction_score:.3f})")
            print("\nPhase 4: COMPOSING FINAL ANSWER")
            try:
                run_state.final_answer = compose_final_answer(
                    question, 
                    run_state.claims, 
                    run_state.verdicts
                )
                print(" ✓ Final answer composed")
                print(f"\n{run_state.final_answer}\n")
            except Exception as answer_error:
                print(f" Failed to compose final answer: {answer_error}")
                run_state.final_answer = "Analysis complete. See detailed claims and verdicts below."
            
            run_state.status = "completed_success"
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "complete",
                "message": "Pipeline completed successfully"
            })
            
            print(f"{'='*80}")
            print(" Enhanced pipeline completed successfully")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR in orchestrator: {type(e).__name__}: {e}")
            print(f"{'='*80}")
            traceback.print_exc()
            
            run_state.status = "failed"
            run_state.logs.append({
                "timestamp": datetime.now().isoformat(),
                "step": "error",
                "message": f"Pipeline failed: {str(e)}"
            })
            raise
        finally:
            run_state.completed_at = datetime.now()
            self.run_history.append(run_state)

        return run_state

    def get_run_history(self) -> List[Dict[str, Any]]:
        history = []
        for rs in self.run_history:
            history.append({
                "question": rs.question,
                "status": rs.status,
                "evidence_count": len(rs.evidence_spans),
                "claim_count": len(rs.claims),
                "verdict_counts": {
                    "supported": sum(1 for v in rs.verdicts if v.label == Label.SUPPORTED),
                    "contested": sum(1 for v in rs.verdicts if v.label == Label.CONTESTED),
                    "uncertain": sum(1 for v in rs.verdicts if v.label == Label.UNCERTAIN),
                },
                "started_at": rs.started_at.isoformat() if rs.started_at else None,
                "completed_at": rs.completed_at.isoformat() if rs.completed_at else None,
                "has_final_answer": rs.final_answer is not None,
            })
        return history

orchestrator = PipelineOrchestrator()


def run_full_pipeline(question: str, max_claims: int = 2, retrieval_k: int = 12) -> RunState:
    return orchestrator.run_pipeline(question, max_claims, retrieval_k)