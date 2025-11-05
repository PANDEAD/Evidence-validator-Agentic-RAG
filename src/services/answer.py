# src/services/answer.py
"""
Final answer composer:
- Prefer LLM summary (Claude) to produce a clear headline + verdict + short markdown answer.
- Fall back to deterministic template if LLM unavailable.
"""
from __future__ import annotations
from typing import List
from src.core.schemas import Claim, Verdict, Label
from .llm import llm_json

# ---------- LLM first ----------

def compose_final_answer_llm(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str | None:
    if not claims or not verdicts:
        return None

    packed = []
    for c in claims:
        v = next((v for v in verdicts if v.claim_id == c.id), None)
        packed.append({
            "claim": c.text,
            "support_score": getattr(v, "support_score", 0.0),
            "contradiction_score": getattr(v, "contradiction_score", 0.0),
            "verdict": getattr(v, "label", Label.UNCERTAIN).value if v else "UNCERTAIN",
            "evidence_ids": c.evidence_ids,
        })

    prompt = f"""
Return STRICT JSON shaped as:
{{"headline":"", "verdict":"SUPPORTED|CONTESTED|UNCERTAIN|MIXED", "answer_md": ""}}

- "headline": one concise sentence that states the overall outcome relative to the user question.
- "verdict": choose SUPPORTED if most claims are supported with margin, CONTESTED if mostly contradicted, UNCERTAIN if weak/mixed, MIXED if truly split.
- "answer_md": 4–6 sentences of markdown using ONLY the provided claims/verdicts (no new knowledge), pointing to the decisive facts.

User question:
{question}

Claims with verdicts (use only these):
{packed}
"""
    js = llm_json(prompt, system="Summarize faithfully. Output JSON only.", max_retries=1)
    if not js:
        return None

    headline = (js.get("headline") or "").strip()
    verdict  = (js.get("verdict")  or "UNCERTAIN").strip()
    answer_md = (js.get("answer_md") or "").strip()

    if not answer_md and not headline:
        return None

    out = ""
    if headline:
        out += f"**{headline}**\n\n"
    out += f"**Overall verdict:** {verdict}\n\n"
    if answer_md:
        out += answer_md
    return out

# ---------- Deterministic fallback ----------

def compose_final_answer(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str:
    """
    Compose a human-readable final answer from claims and verdicts (no LLM).
    """
    if not claims or not verdicts:
        return "Insufficient evidence to form a conclusion for this question."

    lbl = [v.label for v in verdicts]
    n_sup = sum(1 for x in lbl if x == Label.SUPPORTED)
    n_con = sum(1 for x in lbl if x == Label.CONTESTED)
    n_unc = sum(1 for x in lbl if x == Label.UNCERTAIN)

    if n_sup > 0 and n_con == 0:
        head = "Overall, the retrieved evidence supports the proposed claims."
    elif n_con > 0 and n_sup == 0:
        head = "Overall, the retrieved evidence challenges the proposed claims."
    elif n_sup > 0 and n_con > 0:
        head = "Overall, the retrieved evidence is mixed, with both supporting and contradicting findings."
    else:
        head = "Overall, the retrieved evidence is inconclusive for the proposed claims."

    bullets = []
    vmap = {v.claim_id: v for v in verdicts}
    for i, cl in enumerate(claims, 1):
        v = vmap.get(cl.id)
        if not v:
            continue
        bullets.append(
            f"• **Claim {i}**: {v.label.value} "
            f"(support={v.support_score:.2f}, contradict={v.contradiction_score:.2f})\n"
            f"  ↳ {cl.text}"
        )

    summary = f"\n\n**Summary**: {n_sup} supported, {n_con} contested, {n_unc} uncertain out of {len(verdicts)} total claims."
    return f"{head}\n\n" + "\n\n".join(bullets) + summary
