# src/agents/claim_synthesizer.py
"""
Phase I2: LLM Claim Synthesizer (Claude) with strict JSON + heuristic fallback.

Public API:
    synthesize_claims(evidence_spans, question, max_claims=2) -> List[Claim]

Behavior:
- Try LLM first (uses src.services.llm.llm_json). If JSON/IDs are valid → use those claims.
- If LLM unavailable / invalid JSON / bad evidence_ids → fall back to the heuristic builder.
- Emits clear logs so you can see which path was taken.
"""
from __future__ import annotations
from typing import List, Dict, Set
from collections import Counter
import uuid
import re
import logging

from ..core.schemas import EvidenceSpan, Claim
from ..services.llm import llm_json

logger = logging.getLogger(__name__)

MAX_EVIDENCE_TO_FEED = 8  # keep prompt compact and focused

# ------------------------ Public entry ------------------------

def synthesize_claims(evidence_spans: List[EvidenceSpan], question: str, max_claims: int = 2) -> List[Claim]:
    """
    Try LLM synthesis first. If unavailable/invalid → heuristic fallback.
    """
    logger.info("[Synth] Attempting LLM claim synthesis (max_claims=%d)", max_claims)
    claims_llm = _llm_claims(evidence_spans, question, max_claims)
    if claims_llm:
        logger.info("[Synth] ✅ Using %d claim(s) from LLM", len(claims_llm))
        return claims_llm[:max_claims]

    logger.warning("[Synth] ⚠️ LLM produced no usable claims → falling back to heuristic")
    claims_heur = synthesize_claims_heuristic(evidence_spans, max_claims=max_claims)
    if claims_heur:
        logger.info("[Synth] ✅ Heuristic produced %d claim(s)", len(claims_heur))
    else:
        logger.warning("[Synth] ⚠️ Heuristic also produced no claims")
    return claims_heur[:max_claims]

# ------------------------ LLM path ----------------------------

def _llm_claims(evidence_spans: List[EvidenceSpan], question: str, max_claims: int) -> List[Claim]:
    if len(evidence_spans) < 2:
        logger.warning("[Synth] Not enough evidence spans for LLM (%d < 2)", len(evidence_spans))
        return []

    # choose up to MAX_EVIDENCE_TO_FEED evidence spans (retrieval already diversified)
    spans = evidence_spans[:MAX_EVIDENCE_TO_FEED]
    
    # Build evidence list with clear IDs
    evidence_list = []
    for i, s in enumerate(spans, 1):
        evidence_list.append(
            f'[{s.id}] (Paper: {s.paper_id}, Section: {s.section})\n"{s.text}"'
        )
    evidence_str = "\n\n".join(evidence_list)

    prompt = f"""You are analyzing scientific evidence to generate falsifiable claims.

USER QUESTION:
{question}

EVIDENCE (use ONLY these IDs in your citations):
{evidence_str}

TASK:
Generate {max_claims} specific, falsifiable claim(s) that are DIRECTLY supported by the evidence above.

CRITICAL REQUIREMENTS:
1. Each claim MUST cite AT LEAST 2 evidence IDs (e.g., ["id1", "id2", "id3"])
2. Only use IDs from the evidence list above
3. Claims should be specific and measurable, not vague
4. One claim per object in the array
5. Claims should be 20-150 characters long

OUTPUT FORMAT (strict JSON, no markdown, no explanations):
{{
  "claims": [
    {{
      "text": "Specific falsifiable claim based on evidence",
      "evidence_ids": ["id1", "id2", "id3"]
    }}
  ]
}}

Generate the JSON now:"""

    js = llm_json(
        prompt,
        system="You are a scientific claim synthesizer. Output ONLY valid JSON matching the schema. Each claim MUST cite at least 2 evidence_ids.",
        max_retries=2,
        max_tokens=800,
        temperature=0.1,
    )

    if not js or "claims" not in js or not isinstance(js.get("claims"), list):
        logger.warning("[Synth] LLM returned None/invalid JSON structure")
        return []

    out: List[Claim] = []
    seen_text_hash: Set[str] = set()
    valid_ids = {s.id for s in spans}

    logger.info("[Synth] LLM returned %d claim candidates", len(js.get("claims", [])))

    for idx, item in enumerate(js.get("claims", [])[:max_claims], 1):
        try:
            text = str(item.get("text", "")).strip()
            ids_raw = item.get("evidence_ids", []) or []
            
            # Validate evidence IDs
            ids = [eid for eid in ids_raw if eid in valid_ids]
            
            logger.debug(
                "[Synth] Claim %d: text_len=%d, raw_ids=%d, valid_ids=%d",
                idx, len(text), len(ids_raw), len(ids)
            )

            # Relaxed validation: allow 1+ evidence_ids initially, prefer 2+
            if len(text) < 20:
                logger.warning("[Synth] Claim %d: Text too short (%d chars)", idx, len(text))
                continue
            
            if len(text) > 300:
                logger.warning("[Synth] Claim %d: Text too long (%d chars)", idx, len(text))
                continue
                
            if len(ids) < 1:
                logger.warning("[Synth] Claim %d: No valid evidence IDs (raw=%s)", idx, ids_raw)
                continue
            
            # Warn if less than 2 citations but still accept
            if len(ids) < 2:
                logger.warning("[Synth] Claim %d: Only %d citation(s), should be 2+", idx, len(ids))
            
            # Check for duplicates
            key = text.lower()[:200]
            if key in seen_text_hash:
                logger.warning("[Synth] Claim %d: Duplicate detected, skipping", idx)
                continue
            
            seen_text_hash.add(key)
            
            # Create claim with up to 4 evidence IDs
            claim = Claim(
                id=str(uuid.uuid4()),
                text=text,
                evidence_ids=ids[:4]
            )
            out.append(claim)
            logger.info("[Synth] ✓ Claim %d accepted: %s (evidence: %d)", idx, text[:80], len(ids))
            
        except Exception as e:
            logger.warning("[Synth] Error validating claim %d: %s", idx, e)

    if not out:
        logger.warning("[Synth] LLM produced claims but NONE passed validation")
        logger.warning("[Synth] This usually means evidence_ids were missing or invalid")
    
    return out

# ------------------------ Heuristic fallback ------------------

def synthesize_claims_heuristic(evidence_spans: List[EvidenceSpan], max_claims: int = 2) -> List[Claim]:
    if len(evidence_spans) < 2:
        return []

    quality_spans = [span for span in evidence_spans if len(span.text.strip()) >= 50]
    if len(quality_spans) < 2:
        return []

    claims: List[Claim] = []

    if len(quality_spans) >= 4:
        mid = len(quality_spans) // 2
        g1 = quality_spans[:mid]
        g2 = quality_spans[mid:]
        c1 = _create_claim_from_group(g1, 1)
        if c1: claims.append(c1)
        c2 = _create_claim_from_group(g2, 2)
        if c2: claims.append(c2)
    else:
        c = _create_claim_from_group(quality_spans, 1)
        if c: claims.append(c)

    return claims[:max_claims]

def _create_claim_from_group(evidence_group: List[EvidenceSpan], claim_num: int) -> Claim | None:
    if len(evidence_group) < 2:
        return None
    themes = _extract_themes(evidence_group)
    papers = _get_paper_diversity(evidence_group)
    text = _generate_claim_text(themes, papers, claim_num)
    evidence_ids = [span.id for span in evidence_group[:4]]
    return Claim(id=str(uuid.uuid4()), text=text, evidence_ids=evidence_ids)

def _extract_themes(evidence_spans: List[EvidenceSpan]) -> Dict[str, int]:
    combined = " ".join([s.text for s in evidence_spans])
    stop = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were',
        'be','been','have','has','had','do','does','did','will','would','could','should',
        'this','that','these','those','i','you','he','she','it','we','they'
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined.lower())
    filtered = [w for w in words if w not in stop]
    return dict(Counter(filtered).most_common(5))

def _get_paper_diversity(evidence_spans: List[EvidenceSpan]) -> Set[str]:
    return {s.paper_id for s in evidence_spans}

def _generate_claim_text(themes: Dict[str, int], papers: Set[str], claim_num: int) -> str:
    paper_count = len(papers)
    top = list(themes.keys())[:2]
    topic = ", ".join(top) if top else "the available data"
    templates = [
        f"Based on evidence from {paper_count} sources, research suggests patterns related to {topic} that warrant further investigation.",
        f"Analysis of {paper_count} papers indicates potential relationships involving {topic}, though additional research may be needed for confirmation.",
        f"Evidence from multiple sources ({paper_count} papers) points to considerations around {topic} that merit scientific attention."
    ]
    return templates[(claim_num - 1) % len(templates)]