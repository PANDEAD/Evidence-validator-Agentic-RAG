# src/agents/claim_synthesizer.py

from __future__ import annotations
from typing import List, Dict, Set
from collections import Counter
import uuid
import re
import logging

from ..core.schemas import EvidenceSpan, Claim
from ..services.llm import llm_json

logger = logging.getLogger(__name__)

MAX_EVIDENCE_TO_FEED = 10
MIN_EVIDENCE_FOR_LLM = 3

def synthesize_claims(evidence_spans: List[EvidenceSpan], question: str, max_claims: int = 2) -> List[Claim]:
    
    logger.info("[Synth] Starting claim synthesis (max_claims=%d)", max_claims)
    
 
    is_short_technical = len(question.split()) <= 12
   
    claims_llm = _llm_claims(evidence_spans, question, max_claims, is_short_technical)
    if claims_llm:
        logger.info("[Synth]  LLM produced %d claims", len(claims_llm))
        return claims_llm[:max_claims]

    logger.warning("[Synth]  LLM failed → smart heuristic fallback")
    
    claims_smart = _smart_heuristic_claims(evidence_spans, question, max_claims)
    if claims_smart:
        logger.info("[Synth]  Smart heuristic produced %d claims", len(claims_smart))
        return claims_smart[:max_claims]
    claims_heur = synthesize_claims_heuristic(evidence_spans, max_claims=max_claims)
    return claims_heur[:max_claims]


def _llm_claims(
    evidence_spans: List[EvidenceSpan], 
    question: str, 
    max_claims: int,
    is_short_technical: bool = False
) -> List[Claim]:

    if len(evidence_spans) < MIN_EVIDENCE_FOR_LLM:
        logger.warning("[Synth] Not enough evidence (%d < %d)", len(evidence_spans), MIN_EVIDENCE_FOR_LLM)
        return []
    quality_spans = [
        s for s in evidence_spans[:MAX_EVIDENCE_TO_FEED]
        if len(s.text.strip()) >= 100
    ]
    
    if len(quality_spans) < MIN_EVIDENCE_FOR_LLM:
        logger.warning("[Synth] Not enough quality spans after filtering")
        return []

    evidence_list = []
    for i, s in enumerate(quality_spans, 1):
        evidence_list.append(
            f'[{s.id}] (Paper: {s.paper_id}, Section: {s.section})\n"{s.text[:500]}..."'
        )
    evidence_str = "\n\n".join(evidence_list)
    if is_short_technical:
        prompt = f"""You are a scientific claim synthesizer analyzing evidence to answer a TECHNICAL research question.

USER QUESTION:
"{question}"

EVIDENCE SPANS (use ONLY these IDs in citations):
{evidence_str}

TASK:
Generate {max_claims} specific claims that DIRECTLY answer the technical question.

REQUIREMENTS:
1. **Relevance**: Claims MUST directly address the technical concept in the question
2. **Precision**: State what the evidence shows about the concept (existence, absence, properties, limitations)
3. **Citations**: Each claim MUST cite AT LEAST 2 evidence IDs
4. **Accuracy**: Only state what the evidence explicitly supports or contradicts
5. **Technical clarity**: Use precise technical language from the domain

GOOD EXAMPLES (for technical questions):
✓ "Self-reflection methods lack formal convergence proofs in current literature, relying primarily on empirical validation"
✓ "Existing research demonstrates practical effectiveness of self-reflection without establishing theoretical convergence guarantees"
✓ "Current RAG architectures require external vector stores, increasing memory update complexity compared to parameter-based approaches"
✓ "Fine-tuning enables direct parameter updates, while RAG systems necessitate separate index maintenance and retrieval overhead"

BAD EXAMPLES:
❌ "Evidence from papers indicates quantifiable relationships involving fake, passport" (irrelevant to question)
❌ "Research suggests various patterns" (too vague)
❌ "Studies show considerations" (doesn't answer question)

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "claims": [
    {{
      "text": "Precise technical claim that directly answers the question",
      "evidence_ids": ["id1", "id2", "id3"],
      "confidence": "high|medium|low"
    }}
  ]
}}

Generate JSON now:"""
    else:
        prompt = f"""You are a scientific claim synthesizer analyzing evidence to answer a research question.

USER QUESTION:
"{question}"

EVIDENCE SPANS (use ONLY these IDs in citations):
{evidence_str}

TASK:
Generate {max_claims} specific, falsifiable claims that DIRECTLY answer the user's question.

CRITICAL REQUIREMENTS:
1. **Relevance**: Claims MUST directly address the question - no tangential information
2. **Specificity**: Include numbers, percentages, or concrete facts when available
3. **Citations**: Each claim MUST cite AT LEAST 2 evidence IDs (preferably 3-4)
4. **Falsifiability**: Claims must be testable/verifiable, not vague statements
5. **Accuracy**: Only state what the evidence explicitly supports

GOOD EXAMPLES (for quantitative questions):
✓ "The top 1% income share in France was approximately 20% in 1900, declining to 7-8% by the 1990s"
✓ "Historical data shows France's top percentile captured one-fifth of total income at the century's start"

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "claims": [
    {{
      "text": "Specific, factual claim with details that directly answers the question",
      "evidence_ids": ["id1", "id2", "id3"],
      "confidence": "high|medium|low"
    }}
  ]
}}

Generate JSON now:"""

    js = llm_json(
        prompt,
        system="You are a precise scientific claim synthesizer. Output ONLY valid JSON. Be specific and accurate.",
        max_retries=2,
        max_tokens=1200,  
        temperature=0.0,
    )

    if not js or "claims" not in js:
        logger.warning("[Synth] LLM returned invalid JSON")
        return []

    out: List[Claim] = []
    seen_text: Set[str] = set()
    valid_ids = {s.id for s in quality_spans}

    for idx, item in enumerate(js.get("claims", [])[:max_claims], 1):
        try:
            text = str(item.get("text", "")).strip()
            ids_raw = item.get("evidence_ids", []) or []
            confidence = str(item.get("confidence", "medium")).lower()
            ids = [eid for eid in ids_raw if eid in valid_ids]
            min_length = 25 if is_short_technical else 30
            max_length = 450
            min_citations = 2
            
            if len(text) < min_length:
                logger.warning("[Synth] Claim %d: Too short (%d chars)", idx, len(text))
                continue
            
            if len(text) > max_length:
                logger.warning("[Synth] Claim %d: Too long (%d chars)", idx, len(text))
                continue
            
            if len(ids) < min_citations:
                logger.warning("[Synth] Claim %d: Insufficient citations (%d < %d)", idx, len(ids), min_citations)
                continue
            irrelevant_phrases = [
                "evidence from.*papers indicates quantifiable relationships involving",
                "analysis of.*sources shows measurable patterns related to",
                "research suggests patterns",
                "studies demonstrate relationships",
                "fake.*passport",  
                "2025.*2024",  
            ]
            is_irrelevant = any(re.search(phrase, text.lower()) for phrase in irrelevant_phrases)
            if is_irrelevant:
                logger.warning("[Synth] Claim %d: Irrelevant/generic content detected", idx)
                continue
            
            if is_short_technical:
                
                technical_indicators = [
                    r'\b(lack|without|absence|no|not|limited|insufficient)\b', 
                    r'\b(has|have|includes|contains|provides|enables)\b',  
                    r'\b(method|approach|technique|algorithm|system|framework)\b',  
                    r'\b(formal|theoretical|empirical|practical)\b',  
                    r'\b(proof|guarantee|property|characteristic|feature)\b',  
                ]
                has_technical_content = any(re.search(pattern, text, re.IGNORECASE) for pattern in technical_indicators)
                
                if not has_technical_content:
                    logger.warning("[Synth] Claim %d: Lacks technical content", idx)
                    continue
                    
            else:
    
                vague_patterns = [
                    r'\b(suggests?|may|might|could|possibly|potentially)\b',
                    r'\b(patterns?|considerations?|aspects?)\b',
                    r'\b(further research|additional study)\b'
                ]
                vague_count = sum(1 for p in vague_patterns if re.search(p, text, re.IGNORECASE))
                if vague_count >= 2:
                    logger.warning("[Synth] Claim %d: Too vague (matched %d vague patterns)", idx, vague_count)
                    continue
                
                has_number = bool(re.search(r'\d+', text))
                has_specific_term = bool(re.search(
                    r'\b(percent|ratio|rate|share|declined?|increased?|from|to|approximately|around)\b', 
                    text, 
                    re.IGNORECASE
                ))
                
                if not (has_number or has_specific_term):
                    logger.warning("[Synth] Claim %d: Lacks quantitative specificity", idx)
                    continue
            key = text.lower()[:100]
            if key in seen_text:
                logger.warning("[Synth] Claim %d: Duplicate detected", idx)
                continue
            seen_text.add(key)
            
            claim = Claim(
                id=str(uuid.uuid4()),
                text=text,
                evidence_ids=ids[:4]
            )
            out.append(claim)
            logger.info("[Synth] ✓ Claim %d accepted: %s (citations: %d)", 
                       idx, text[:80], len(ids))
            
        except Exception as e:
            logger.warning("[Synth] Error validating claim %d: %s", idx, e)

    if not out:
        logger.warning("[Synth] No claims passed validation")
    
    return out


def _smart_heuristic_claims(
    evidence_spans: List[EvidenceSpan], 
    question: str, 
    max_claims: int
) -> List[Claim]:
    if len(evidence_spans) < 2:
        return []
    
    
    keywords = _extract_keywords(question)
    logger.info(f"[SmartHeur] Keywords: {list(keywords)[:5]}")
    
    if not keywords:
        return []
    scored_spans = []
    for span in evidence_spans[:12]:
        score = sum(1 for kw in keywords if kw in span.text.lower())
        if score > 0:
            scored_spans.append((span, score))
    
    scored_spans.sort(key=lambda x: x[1], reverse=True)
    
    if not scored_spans:
        logger.warning("[SmartHeur] No relevant spans found")
        return []
    
    
    top_spans = [s[0] for s in scored_spans[:6]]
    
    
    claims: List[Claim] = []
    
    if len(top_spans) >= 3:
        relevant_sentences = []
        for span in top_spans[:3]:
            sentences = re.split(r'[.!?]+', span.text)
            for sent in sentences:
                if any(kw in sent.lower() for kw in keywords) and len(sent.strip()) > 40:
                    relevant_sentences.append(sent.strip())
        
        if relevant_sentences:
            claim_text = relevant_sentences[0][:350]
            if len(claim_text) >= 50:
                claims.append(Claim(
                    id=str(uuid.uuid4()),
                    text=claim_text,
                    evidence_ids=[s.id for s in top_spans[:3]]
                ))
    
    return claims[:max_claims]


def _extract_keywords(question: str) -> Set[str]:
    """Extract meaningful keywords from question."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    keywords = {w for w in words if w not in stop_words}
    words_list = question.lower().split()
    bigrams = {f"{words_list[i]} {words_list[i+1]}" 
               for i in range(len(words_list)-1)}
    
    return keywords.union(bigrams)


def synthesize_claims_heuristic(evidence_spans: List[EvidenceSpan], max_claims: int = 2) -> List[Claim]:
    
    if len(evidence_spans) < 2:
        return []

    quality_spans = [s for s in evidence_spans if len(s.text.strip()) >= 80]
    if len(quality_spans) < 2:
        return []

    claims: List[Claim] = []
    

    by_paper: Dict[str, List[EvidenceSpan]] = {}
    for span in quality_spans[:8]:
        by_paper.setdefault(span.paper_id, []).append(span)
    
    paper_groups = list(by_paper.values())
    if len(paper_groups) >= 2:
        for i, group in enumerate(paper_groups[:max_claims], 1):
            if len(group) >= 2:
                claim = _create_claim_from_group(group, i)
                if claim:
                    claims.append(claim)
    else:
        mid = len(quality_spans) // 2
        g1 = quality_spans[:mid]
        g2 = quality_spans[mid:]
        if len(g1) >= 2:
            claims.append(_create_claim_from_group(g1, 1))
        if len(g2) >= 2:
            claims.append(_create_claim_from_group(g2, 2))

    return claims[:max_claims]


def _create_claim_from_group(evidence_group: List[EvidenceSpan], claim_num: int) -> Claim | None:
    """Create claim from evidence group."""
    if len(evidence_group) < 2:
        return None
    
    themes = _extract_themes(evidence_group)
    papers = {s.paper_id for s in evidence_group}
    
    numbers = []
    for s in evidence_group:
        numbers.extend(re.findall(r'\d+(?:\.\d+)?%?', s.text))
    
    topic = ", ".join(list(themes.keys())[:2]) if themes else "the available data"
    
    if numbers:
        num_str = numbers[0]
        templates = [
            f"Evidence from {len(papers)} papers indicates quantifiable relationships involving {topic}, with values around {num_str}.",
            f"Analysis of {len(papers)} sources shows measurable patterns related to {topic}, including figures near {num_str}.",
        ]
    else:
        templates = [
            f"Evidence from {len(papers)} independent sources converges on findings related to {topic}.",
            f"Cross-study analysis ({len(papers)} papers) reveals consistent patterns involving {topic}.",
        ]
    
    text = templates[claim_num % len(templates)]
    evidence_ids = [s.id for s in evidence_group[:4]]
    
    return Claim(id=str(uuid.uuid4()), text=text, evidence_ids=evidence_ids)


def _extract_themes(evidence_spans: List[EvidenceSpan]) -> Dict[str, int]:
    """Extract key themes from evidence."""
    combined = " ".join([s.text for s in evidence_spans])
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined.lower())
    filtered = [w for w in words if w not in stop_words]
    return dict(Counter(filtered).most_common(5))