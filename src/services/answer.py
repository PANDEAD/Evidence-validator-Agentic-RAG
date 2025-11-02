# src/services/answer.py
"""
Final answer composer that turns claim verdicts into a concise summary.
No LLM needed - just simple string formatting.
"""
from __future__ import annotations
from typing import List
from src.core.schemas import Claim, Verdict, Label


def compose_final_answer(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str:
    """
    Compose a human-readable final answer from claims and verdicts.
    
    Returns:
        A concise paragraph summary with bulleted claim details.
    """
    if not claims or not verdicts:
        return "Insufficient evidence to form a conclusion for this question."

    # Summarize tallies
    lbl = [v.label for v in verdicts]
    n_sup = sum(1 for x in lbl if x == Label.SUPPORTED)
    n_con = sum(1 for x in lbl if x == Label.CONTESTED)
    n_unc = sum(1 for x in lbl if x == Label.UNCERTAIN)

    # Create headline based on verdict distribution
    if n_sup > 0 and n_con == 0:
        head = "Overall, the retrieved evidence supports the proposed claims."
    elif n_con > 0 and n_sup == 0:
        head = "Overall, the retrieved evidence challenges the proposed claims."
    elif n_sup > 0 and n_con > 0:
        head = "Overall, the retrieved evidence is mixed, with both supporting and contradicting findings."
    else:
        head = "Overall, the retrieved evidence is inconclusive for the proposed claims."

    # Build detailed bullets
    bullets = []
    vmap = {v.claim_id: v for v in verdicts}
    
    for i, cl in enumerate(claims, 1):
        v = vmap.get(cl.id)
        if not v:
            continue
        
        # Format: Claim number, verdict label, scores, claim text
        bullet = (
            f"• **Claim {i}**: {v.label.value} "
            f"(support={v.support_score:.2f}, contradict={v.contradiction_score:.2f})\n"
            f"  ↳ {cl.text}"
        )
        bullets.append(bullet)

    # Add summary statistics
    summary = f"\n\n**Summary**: {n_sup} supported, {n_con} contested, {n_unc} uncertain out of {len(verdicts)} total claims."

    return f"{head}\n\n" + "\n\n".join(bullets) + summary


if __name__ == "__main__":
    # Test
    from src.core.schemas import Claim, Verdict, Label
    
    claims = [
        Claim(id="c1", text="Test claim 1", evidence_ids=["e1", "e2"]),
        Claim(id="c2", text="Test claim 2", evidence_ids=["e3", "e4"]),
    ]
    
    verdicts = [
        Verdict(claim_id="c1", label=Label.SUPPORTED, support_score=0.8, contradiction_score=0.1, rationale="Good"),
        Verdict(claim_id="c2", label=Label.CONTESTED, support_score=0.2, contradiction_score=0.7, rationale="Bad"),
    ]
    
    answer = compose_final_answer("Test question", claims, verdicts)
    print(answer)
    print("\n Answer service test passed")