
from __future__ import annotations
from typing import List, Dict, Any
from src.core.schemas import Claim, Verdict, Label
from .llm import llm_json
import logging

logger = logging.getLogger(__name__)


def compose_final_answer(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str:
   
    try:
        adjusted_verdicts = _adjust_verdicts_for_opposition(question, claims, verdicts)
        llm_answer = _compose_llm_answer(question, claims, adjusted_verdicts)
        if llm_answer:
            return llm_answer
        return _compose_template_answer(question, claims, adjusted_verdicts)
    
    except Exception as e:
        logger.error(f"[Answer] Error composing final answer: {e}", exc_info=True)
        return f" Error composing answer: {str(e)}\n\nRaw verdicts: {verdicts}"


def _adjust_verdicts_for_opposition(
    question: str, 
    claims: List[Claim], 
    verdicts: List[Verdict]
) -> List[Verdict]:
    """
    Logic:
    - If claim OPPOSES question AND verdict is SUPPORTED → User's question is REFUTED
    - If claim OPPOSES question AND verdict is CONTESTED → User's question is SUPPORTED
    - If claim ALIGNS with question → Keep original verdict
    """
    adjusted = []
    
    for claim, verdict in zip(claims, verdicts):
        opposition_result = _detect_opposition(question, claim.text)
        opposes = opposition_result["opposes"]
        reasoning = opposition_result["reasoning"]
        
        logger.info(f"[Opposition Check] Claim opposes question: {opposes} - {reasoning[:100]}")
        
        if opposes:
            if verdict.label == Label.SUPPORTED:
                adjusted_verdict = Verdict(
                    claim_id=verdict.claim_id,
                    label=Label.CONTESTED,  
                    support_score=verdict.support_score,
                    contradiction_score=verdict.contradiction_score,
                    rationale=f"[OPPOSES QUESTION: {reasoning}] {verdict.rationale}"
                )
                adjusted.append(adjusted_verdict)
                logger.info(f"[Opposition] SUPPORTED claim opposes question → verdict = CONTESTED")
                
            elif verdict.label == Label.CONTESTED:
                adjusted_verdict = Verdict(
                    claim_id=verdict.claim_id,
                    label=Label.SUPPORTED,  
                    support_score=verdict.support_score,
                    contradiction_score=verdict.contradiction_score,
                    rationale=f"[OPPOSES QUESTION: {reasoning}] {verdict.rationale}"
                )
                adjusted.append(adjusted_verdict)
                logger.info(f"[Opposition] CONTESTED claim opposes question → verdict = SUPPORTED")
                
            else:
                
                adjusted.append(verdict)
        else:
            
            adjusted.append(verdict)
            logger.info(f"[Alignment] Claim aligns with question → verdict unchanged")
    
    return adjusted


def _detect_opposition(question: str, claim: str) -> Dict[str, Any]:
    """
    Detect if a claim takes the OPPOSITE semantic position from the question.
    
    Returns:
        Dict with keys:
        - "opposes": bool (True if opposition detected)
        - "reasoning": str (explanation)
    """
    prompt = f"""Determine if the CLAIM takes the OPPOSITE semantic position from the QUESTION.

**QUESTION**: {question}

**CLAIM**: {claim}

**CRITICAL DISTINCTION**:
- OPPOSITE means they assert CONTRADICTORY positions (one says YES, other says NO)
- SAME means they assert the SAME position (both YES or both NO), even if wording differs

**EXAMPLES OF OPPOSITION** (contradictory positions):
1. Question: "Legal frameworks ARE sufficient" 
   Claim: "Legal frameworks ARE NOT sufficient / ARE insufficient"
   → OPPOSITE ✓ (one affirms, other denies)

2. Question: "X is ethical"
   Claim: "X is unethical"
   → OPPOSITE ✓ (ethical vs unethical)

3. Question: "X should be restricted"
   Claim: "X should not be restricted / restrictions are unnecessary"
   → OPPOSITE ✓ (for vs against restrictions)

**EXAMPLES OF ALIGNMENT** (same position):
1. Question: "Deepfake creation IS unethical"
   Claim: "Deepfakes ENABLE unethical manipulation"
   → SAME ✓ (both express negative ethical stance)

2. Question: "Legal frameworks ARE insufficient"
   Claim: "Current laws FAIL to adequately address deepfakes"
   → SAME ✓ (both express inadequacy)

3. Question: "X is harmful"
   Claim: "X causes significant damage"
   → SAME ✓ (both negative - harm/damage)

**TASK**: 
Does the claim take the OPPOSITE semantic position from the question?

Answer ONLY with valid JSON:
{{
  "opposition": "OPPOSITE" or "SAME",
  "reasoning": "Brief explanation (1-2 sentences)"
}}

Generate JSON now:"""

    js = llm_json(
        prompt,
        system="You are a semantic logic analyzer. Focus on whether positions CONTRADICT (opposite) or ALIGN (same). Output ONLY valid JSON.",
        max_retries=2,
        max_tokens=200,
        temperature=0.0
    )
    
    if not js:
        logger.warning("[Opposition] LLM failed, using keyword fallback")
        opposes = _keyword_based_opposition(question, claim)
        return {
            "opposes": opposes,
            "reasoning": "Keyword-based detection (LLM unavailable)"
        }
    
    opposition = js.get("opposition", "SAME").strip().upper()
    reasoning = js.get("reasoning", "").strip()
    
    opposes = (opposition == "OPPOSITE")
    
    return {
        "opposes": opposes,
        "reasoning": reasoning
    }


def _keyword_based_opposition(question: str, claim: str) -> bool:
    q_lower = question.lower()
    c_lower = claim.lower()
    
    opposition_pairs = [
        ("sufficient", "insufficient"),
        ("sufficient", "inadequate"),
        ("adequate", "inadequate"),
        ("ethical", "unethical"),
        ("possible", "impossible"),
        ("legal", "illegal"),
        ("effective", "ineffective"),
        ("necessary", "unnecessary"),
        ("appropriate", "inappropriate"),
        ("should be", "should not be"),
        ("can be", "cannot be"),
        ("will", "will not"),
        ("is", "is not"),
        ("are", "are not"),
        ("has", "has not"),
        ("have", "have not"),
    ]
    
    for positive, negative in opposition_pairs:
        if (positive in q_lower and negative in c_lower) or \
           (negative in q_lower and positive in c_lower):
            logger.info(f"[Keyword] Found opposition: '{positive}' vs '{negative}'")
            return True
    
    return False


def _compose_llm_answer(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str | None:
   
    if not claims or not verdicts:
        return None
    
    packed = []
    for c in claims:
        v = next((v for v in verdicts if v.claim_id == c.id), None)
        if v:
            opposes = "[OPPOSES QUESTION" in v.rationale
            opposition_note = "⚠️ [Opposes question]" if opposes else "✓ [Aligns with question]"
            
            packed.append({
                "claim": c.text,
                "verdict": v.label.value,
                "support": round(v.support_score, 2),
                "contradiction": round(v.contradiction_score, 2),
                "opposition_note": opposition_note,
                "rationale": v.rationale[:200]
            })
    
    if not packed:
        return None

    prompt = f"""You are synthesizing research findings to answer a specific question.

**USER'S QUESTION**:
{question}

**VALIDATED CLAIMS** (verdicts adjusted for semantic opposition):
{packed}

**IMPORTANT**: 
- Claims marked ⚠️ OPPOSE the user's question (assert the opposite)
- Claims marked ✓ ALIGN with the user's question
- If an opposing claim is CONTESTED, it means user's question is REFUTED

**TASK**:
Generate a concise answer in this JSON format:
{{
  "headline": "One-sentence direct answer to the user's question",
  "verdict": "SUPPORTED|REFUTED|MIXED|UNCERTAIN",
  "answer": "2-3 sentences explaining what evidence says"
}}

**VERDICT RULES**:
- SUPPORTED: Evidence supports user's question
- REFUTED: Evidence refutes user's question (opposite is proven)
- MIXED: Evidence divided
- UNCERTAIN: Evidence unclear

Generate JSON now:"""

    js = llm_json(
        prompt,
        system="You are a precise research synthesizer. Output ONLY valid JSON.",
        max_retries=2,
        max_tokens=600,
        temperature=0.1
    )
    
    if not js:
        return None
    
    headline = (js.get("headline") or "").strip()
    verdict = (js.get("verdict") or "UNCERTAIN").strip().upper()
    answer = (js.get("answer") or "").strip()
    
    if not headline and not answer:
        return None
    
    out = f"**{headline}**\n\n"
    out += f"**Overall Assessment:** {verdict}\n\n"
    out += f"{answer}\n\n"
    
    out += "**Detailed Findings:**\n"
    for i, item in enumerate(packed, 1):
        out += f"{i}. **{item['verdict']}** {item['opposition_note']}\n"
        out += f"   📊 Evidence: support={item['support']}, contra={item['contradiction']}\n"
        out += f"   ↳ {item['claim']}\n\n"
    
    return out


def _compose_template_answer(question: str, claims: List[Claim], verdicts: List[Verdict]) -> str:
    
    if not claims or not verdicts:
        return "❌ Insufficient evidence to answer this question."
   
    lbl = [v.label for v in verdicts]
    n_sup = sum(1 for x in lbl if x == Label.SUPPORTED)
    n_con = sum(1 for x in lbl if x == Label.CONTESTED)
    n_unc = sum(1 for x in lbl if x == Label.UNCERTAIN)
    
   
    if n_sup > 0 and n_con == 0:
        headline = "✅ Evidence SUPPORTS the user's question"
        verdict = "SUPPORTED"
    elif n_con > 0 and n_sup == 0:
        headline = "❌ Evidence REFUTES the user's question"
        verdict = "REFUTED"
    elif n_sup > 0 and n_con > 0:
        headline = "⚠️ Evidence is MIXED"
        verdict = "MIXED"
    else:
        headline = "❓ Evidence is INCONCLUSIVE"
        verdict = "UNCERTAIN"
    
   
    out = f"**{headline}**\n\n"
    out += f"**Overall Assessment:** {verdict}\n\n"
    
    
    if n_sup > n_con:
        out += f"The majority of claims ({n_sup}/{len(verdicts)}) support the user's question. "
    elif n_con > n_sup:
        out += f"The majority of claims ({n_con}/{len(verdicts)}) refute the user's question. "
    else:
        out += f"Evidence is divided: {n_sup} supporting, {n_con} refuting, {n_unc} uncertain. "
    
    out += "\n\n**Detailed Findings:**\n\n"
    
    vmap = {v.claim_id: v for v in verdicts}
    for i, cl in enumerate(claims, 1):
        v = vmap.get(cl.id)
        if not v:
            continue
        
        
        opposes = "[OPPOSES QUESTION" in v.rationale
        opposition_marker = " ⚠️ [Opposes question]" if opposes else ""
        
        emoji = {
            Label.SUPPORTED: "✅",
            Label.CONTESTED: "❌",
            Label.UNCERTAIN: "❓"
        }.get(v.label, "•")
        
        out += f"{i}. {emoji} **{v.label.value}**{opposition_marker}\n"
        out += f"   📊 Evidence: support={v.support_score:.2f}, contra={v.contradiction_score:.2f}\n"
        out += f"   ↳ {cl.text}\n\n"
    
   
    out += f"\n**Summary:** {n_sup} supporting, {n_con} refuting, {n_unc} uncertain (total: {len(verdicts)})\n"
    
    return out