# src/services/validation.py

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import torch
import traceback
import gc

from src.core.schemas import EvidenceSpan, Claim, Verdict, Label, TAU_SUPPORT, TAU_CONTRADICT

_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
_tokenizer = None
_model = None
_device = "cpu"

ALPHA_CAL = 1.08
TRIM_FRAC = 0.0
MIN_EVIDENCE = 2
MARGIN = 0.10

def _adaptive_thresholds(num_spans: int, claim_complexity: float = 1.0) -> Tuple[float, float]:
    """
    FIXED: Adaptive thresholds based on evidence count AND claim complexity.
    
    claim_complexity: 1.0 = normal, <1.0 = simple claim, >1.0 = complex claim
    """
    base_support = 0.60
    base_contradict = 0.40
    
    if num_spans >= 4:
        support_thresh = base_support
        contradict_thresh = base_contradict
    elif num_spans == 3:
        support_thresh = 0.50
        contradict_thresh = 0.35
    else:  
        support_thresh = 0.45
        contradict_thresh = 0.30
    
    support_thresh *= claim_complexity
    contradict_thresh *= claim_complexity
    
    return support_thresh, contradict_thresh


def _load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f" Loading NLI model: {_MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            _model.to(_device)
            _model.eval()
            print(f" NLI model loaded")
        except Exception as e:
            print(f"ERROR loading NLI model: {e}")
            raise


def _get_label_indices() -> Tuple[int, int, int]:
  
    try:
        id2label: Dict[int, str] = getattr(_model.config, "id2label", {})
        norm = {i: s.upper() for i, s in id2label.items()}
        
        def find(tag):
            for i, s in norm.items():
                if tag in s:
                    return i
            raise RuntimeError(f"{tag} not found in {norm}")
        
        idx_entail = find("ENTAIL")
        idx_contra = find("CONTRADICT")
        
        try:
            idx_neutral = find("NEUTRAL")
        except RuntimeError:
            idx_neutral = (set(norm.keys()) - {idx_entail, idx_contra}).pop()
        
        return idx_entail, idx_neutral, idx_contra
    except Exception as e:
        print(f" ERROR in _get_label_indices: {e}")
        raise


def _calibrate_probs(p: np.ndarray, alpha: float = ALPHA_CAL) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0)
    p = p ** alpha
    p = p / p.sum(axis=1, keepdims=True)
    return p


def _robust_mean(x: np.ndarray, trim: float = TRIM_FRAC) -> float:
    if x.size == 0:
        return 0.0
    
    if x.size <= 3 or trim == 0.0:
        return float(x.mean())
    
    k = int(len(x) * trim)
    if k == 0:
        return float(x.mean())
    
    xs = np.sort(x)
    xs = xs[k: len(xs) - k]
    return float(xs.mean()) if xs.size > 0 else float(x.mean())


def _estimate_claim_complexity(claim_text: str) -> float:
 
    length_factor = min(len(claim_text) / 150.0, 1.3)
    
  
    import re
    has_numbers = len(re.findall(r'\d+', claim_text)) > 0
    specificity_factor = 0.9 if has_numbers else 1.1
    hedge_words = ['may', 'might', 'could', 'suggest', 'indicate', 'possibly']
    hedge_count = sum(1 for w in hedge_words if w in claim_text.lower())
    hedge_factor = 1.0 + (hedge_count * 0.05)
    
    complexity = length_factor * specificity_factor * hedge_factor
    return np.clip(complexity, 0.8, 1.3)


def _nli_probs(premises: List[str], hypotheses: List[str], batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    try:
        _load()
        idx_entail, idx_neutral, idx_contra = _get_label_indices()

        probs_all: List[np.ndarray] = []
        logits_all: List[np.ndarray] = []
        
        with torch.no_grad():
            for i in range(0, len(hypotheses), batch_size):
                p = premises[i:i+batch_size]
                h = hypotheses[i:i+batch_size]
                
                enc = _tokenizer(p, h, padding=True, truncation=True, max_length=512, return_tensors="pt")
                enc = {k: v.to(_device) for k, v in enc.items()}
                
                logits = _model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs = probs[:, [idx_entail, idx_neutral, idx_contra]]
                probs_all.append(probs)
                logits_all.append(logits.cpu().numpy())
                
                del enc, logits
                gc.collect()

        if not probs_all:
            return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.float32)

        return np.vstack(probs_all), np.vstack(logits_all)
        
    except Exception as e:
        print(f"ERROR in _nli_probs: {e}")
        raise


def validate_claims_against_spans(
    claims: List[Claim],
    spans: List[EvidenceSpan],
    tau_support: float = TAU_SUPPORT,
    tau_contradict: float = TAU_CONTRADICT,
) -> List[Verdict]:
    
    try:
        print(f" validate_claims_against_spans")
        print(f"   Claims: {len(claims)}, Spans: {len(spans)}")
        print(f"   Base thresholds: support={tau_support}, contradict={tau_contradict}")
        
        span_map: Dict[str, EvidenceSpan] = {s.id: s for s in spans}
        verdicts: List[Verdict] = []

        for idx, cl in enumerate(claims, 1):
            print(f"\n    Claim {idx}/{len(claims)}: {cl.text[:80]}...")
            cited = [
                span_map[sid] for sid in cl.evidence_ids 
                if sid in span_map and isinstance(span_map[sid].text, str)
            ]
            
            print(f"      Found {len(cited)} cited spans")
            
            if len(cited) == 0:
                verdicts.append(Verdict(
                    claim_id=cl.id, label=Label.UNCERTAIN,
                    support_score=0.0, contradiction_score=0.0,
                    rationale="No valid evidence spans found."
                ))
                continue

            complexity = _estimate_claim_complexity(cl.text)
            print(f"      Claim complexity: {complexity:.2f}")
            
            premises = [s.text for s in cited]
            hypotheses = [cl.text] * len(cited)
            
            print(f"      Running NLI inference...")
            probs_raw, _logits = _nli_probs(premises, hypotheses)
            
            if probs_raw.size == 0:
                verdicts.append(Verdict(
                    claim_id=cl.id, label=Label.UNCERTAIN,
                    support_score=0.0, contradiction_score=0.0,
                    rationale="NLI inference failed."
                ))
                continue

            # Calibrate probabilities
            probs = _calibrate_probs(probs_raw, ALPHA_CAL)
            entail_scores = probs[:, 0]
            contra_scores = probs[:, 2]
            entail = float(np.max(entail_scores))
            contra = float(np.median(contra_scores))
            
            print(f"      Entailment scores: {entail_scores}")
            print(f"      Contradiction scores: {contra_scores}")
            print(f"      Aggregated - entail: {entail:.3f} (MAX), contra: {contra:.3f} (MEDIAN)")
            
            adapt_tau_sup, adapt_tau_con = _adaptive_thresholds(len(cited), complexity)
            print(f"      Adaptive thresholds - support: {adapt_tau_sup:.2f}, contradict: {adapt_tau_con:.2f}")
            
            margin_diff = abs(entail - contra)
            print(f"      Margin: {margin_diff:.3f} (threshold: {MARGIN})")
            
            confidence = min(margin_diff / 0.3, 1.0) * min(len(cited) / 4.0, 1.0)
            
            if len(cited) < MIN_EVIDENCE:
                label = Label.UNCERTAIN
                rationale = f"Insufficient evidence ({len(cited)}<{MIN_EVIDENCE})."
            
        
            elif entail >= 0.85:
                label = Label.SUPPORTED
                rationale = f"Very strong support from at least one span (entail={entail:.2f}), outweighs median contradiction ({contra:.2f})."
      
            elif entail >= adapt_tau_sup and contra < adapt_tau_con and margin_diff >= MARGIN:
                label = Label.SUPPORTED
                rationale = f"Strong support (entail={entail:.2f}, contra={contra:.2f}, confidence={confidence:.2f})."
            
            elif contra >= adapt_tau_con and entail < adapt_tau_sup and margin_diff >= MARGIN:
                label = Label.CONTESTED
                rationale = f"Clear contradiction (contra={contra:.2f}, entail={entail:.2f}, confidence={confidence:.2f})."
            
           
            elif entail >= adapt_tau_sup * 0.8 and contra < adapt_tau_con:
                label = Label.SUPPORTED
                rationale = f"Moderate support (entail={entail:.2f}, low contradiction={contra:.2f}, confidence={confidence:.2f})."
            
         
            elif entail >= adapt_tau_sup and margin_diff < MARGIN:
                label = Label.SUPPORTED
                rationale = f"Support with narrow margin (entail={entail:.2f}, contra={contra:.2f}, confidence={confidence:.2f})."
            
       
            elif contra >= adapt_tau_con and entail < 0.50:
                label = Label.CONTESTED
                rationale = f"Moderate contradiction (contra={contra:.2f}, weak support, confidence={confidence:.2f})."
            
         
            elif contra >= adapt_tau_con and margin_diff < MARGIN:
                label = Label.CONTESTED
                rationale = f"Contradiction with narrow margin (contra={contra:.2f}, entail={entail:.2f}, confidence={confidence:.2f})."
            
           
            elif entail >= 0.40 and contra < 0.30:
                label = Label.SUPPORTED
                rationale = f"Weak support (entail={entail:.2f}, very low contradiction, confidence={confidence:.2f})."
            
           
            elif contra >= 0.30 and entail < 0.40:
                label = Label.CONTESTED
                rationale = f"Weak contradiction (contra={contra:.2f}, low support, confidence={confidence:.2f})."
            
            
            else:
                label = Label.UNCERTAIN
                rationale = f"Mixed or inconclusive evidence (entail={entail:.2f}, contra={contra:.2f})."

            
            top_ent_i = int(np.argmax(entail_scores))
            top_con_i = int(np.argmax(contra_scores))
            
            ent_snip = premises[top_ent_i][:120].replace("\n", " ")
            con_snip = premises[top_con_i][:120].replace("\n", " ")
            
            rationale += f' | Top-support: "{ent_snip}..." | Top-contradict: "{con_snip}..."'

            print(f"   Verdict: {label.value} (confidence={confidence:.2f})")

            verdicts.append(Verdict(
                claim_id=cl.id, label=label,
                support_score=entail,
                contradiction_score=contra,
                rationale=rationale
            ))
            
            gc.collect()
        
        print(f"\n    All {len(verdicts)} verdicts computed")
        return verdicts
        
    except Exception as e:
        print(f"\n FATAL ERROR in validate_claims_against_spans: {e}")
        traceback.print_exc()
        raise