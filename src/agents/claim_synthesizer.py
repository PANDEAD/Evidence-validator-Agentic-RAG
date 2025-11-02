# src/agents/claim_synthesizer.py
"""
Phase G: Claim Synthesizer v0 (Bundle B)
Conservative heuristic claim synthesis without LLM.
"""
from typing import List, Dict, Set
from collections import Counter
import uuid
import re
from ..core.schemas import EvidenceSpan, Claim

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
