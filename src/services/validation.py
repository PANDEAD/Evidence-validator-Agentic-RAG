# src/services/validation.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import torch
import traceback
import sys
import gc

from src.core.schemas import EvidenceSpan, Claim, Verdict, Label, TAU_SUPPORT, TAU_CONTRADICT

# Small, public, RAM-friendly NLI model
_MODEL_NAME = "typeform/distilbert-base-uncased-mnli"

_tokenizer = None
_model = None
_device = "cpu"  # keep CPU for macOS portability

# --- FIXED: Smarter calibration & aggregation for sparse evidence
ALPHA_CAL = 1.10      # Gentler sharpening (was 1.15)
TRIM_FRAC = 0.0       # NO TRIMMING with sparse evidence! (was 0.15)
MIN_EVIDENCE = 2      # need at least 2 spans
MARGIN = 0.08         # Reduced margin (was 0.12)

# NEW: Adaptive thresholds based on evidence count
def _adaptive_thresholds(num_spans: int) -> Tuple[float, float]:
    """
    Lower thresholds when we have fewer spans.
    With 2 spans, we can't expect 0.65 after aggregation!
    """
    if num_spans >= 4:
        return 0.60, 0.40  # Original-ish thresholds
    elif num_spans == 3:
        return 0.50, 0.35  # Slightly lower
    else:  # 2 spans
        return 0.40, 0.30  # Much more forgiving

def _load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f" Loading NLI model: {_MODEL_NAME} (device={_device})")
            _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            _model.to(_device)
            _model.eval()
            print(f"✅ NLI model loaded successfully")
        except Exception as e:
            print(f"❌ ERROR loading NLI model: {type(e).__name__}: {e}")
            traceback.print_exc()
            raise

def _get_label_indices() -> Tuple[int, int, int]:
    """Get indices for entailment, neutral, contradiction labels."""
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
        
        # Handle neutral/unknown
        try:
            idx_neutral = find("NEUTRAL")
        except RuntimeError:
            idx_neutral = (set(norm.keys()) - {idx_entail, idx_contra}).pop()
        
        return idx_entail, idx_neutral, idx_contra
    except Exception as e:
        print(f"❌ ERROR in _get_label_indices: {e}")
        traceback.print_exc()
        raise

def _calibrate_probs(p: np.ndarray, alpha: float = ALPHA_CAL) -> np.ndarray:
    """
    Apply power-law calibration to probabilities and renormalize.
    alpha > 1: sharpens probabilities (more confident)
    alpha < 1: flattens probabilities (less confident)
    """
    p = np.clip(p, 1e-6, 1.0)
    p = p ** alpha
    p = p / p.sum(axis=1, keepdims=True)
    return p

def _robust_mean(x: np.ndarray, trim: float = TRIM_FRAC) -> float:
    """
    Compute mean with optional trimming.
    With sparse data (n<=3), NO trimming is applied.
    """
    if x.size == 0:
        return 0.0
    
    # FIXED: Don't trim when we have few samples!
    if x.size <= 3 or trim == 0.0:
        return float(x.mean())
    
    k = int(len(x) * trim)
    if k == 0:
        return float(x.mean())
    
    xs = np.sort(x)
    xs = xs[k: len(xs) - k]
    if xs.size == 0:
        return float(x.mean())
    return float(xs.mean())

def _nli_probs(premises: List[str], hypotheses: List[str], batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probs, logits) with probs in [ENTAIL, NEUTRAL, CONTRA] order.
    BATCH_SIZE=1 for macOS stability!
    """
    try:
        _load()
        idx_entail, idx_neutral, idx_contra = _get_label_indices()

        probs_all: List[np.ndarray] = []
        logits_all: List[np.ndarray] = []
        
        with torch.no_grad():
            for i in range(0, len(hypotheses), batch_size):
                p = premises[i:i+batch_size]
                h = hypotheses[i:i+batch_size]
                
                try:
                    enc = _tokenizer(p, h, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    enc = {k: v.to(_device) for k, v in enc.items()}
                    
                    logits = _model(**enc).logits  # [B,3]
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    
                    # reorder columns to [ENTAIL, NEUTRAL, CONTRA]
                    probs = probs[:, [idx_entail, idx_neutral, idx_contra]]
                    probs_all.append(probs)
                    logits_all.append(logits.cpu().numpy())
                    
                    # Clean up memory after each batch
                    del enc, logits
                    gc.collect()
                    
                except Exception as batch_error:
                    print(f"      ❌ Error in batch {i//batch_size + 1}: {batch_error}")
                    raise

        if not probs_all:
            return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.float32)

        probs = np.vstack(probs_all)
        logits = np.vstack(logits_all)
        return probs, logits
        
    except Exception as e:
        print(f"❌ ERROR in _nli_probs: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

def validate_claims_against_spans(
    claims: List[Claim],
    spans: List[EvidenceSpan],
    tau_support: float = TAU_SUPPORT,
    tau_contradict: float = TAU_CONTRADICT,
) -> List[Verdict]:
    """
    Robust claim validation with:
    - Adaptive thresholds based on evidence count
    - Smart aggregation (no trimming with sparse data)
    - Margin-based decision rules
    - Concise rationales with example snippets
    """
    try:
        print(f"🔬 validate_claims_against_spans called")
        print(f"   Claims: {len(claims)}, Spans: {len(spans)}")
        print(f"   Base thresholds: support={tau_support}, contradict={tau_contradict}")
        print(f"   Calibration: alpha={ALPHA_CAL}, trim={TRIM_FRAC}, margin={MARGIN}")
        
        span_map: Dict[str, EvidenceSpan] = {s.id: s for s in spans}
        verdicts: List[Verdict] = []

        for idx, cl in enumerate(claims, 1):
            print(f"\n   Claim {idx}/{len(claims)}: {cl.text[:80]}...")
            
            cited = [span_map[sid] for sid in cl.evidence_ids 
                    if sid in span_map and isinstance(span_map[sid].text, str)]
            
            print(f"      Found {len(cited)} cited spans")
            
            if len(cited) == 0:
                print(f"      ⚠️  No valid evidence spans")
                verdicts.append(Verdict(
                    claim_id=cl.id, label=Label.UNCERTAIN,
                    support_score=0.0, contradiction_score=0.0,
                    rationale="No valid evidence spans found for this claim."
                ))
                continue

            premises = [s.text for s in cited]
            hypotheses = [cl.text] * len(cited)

            print(f"      Running NLI inference...")
            probs_raw, _logits = _nli_probs(premises, hypotheses)
            
            if probs_raw.size == 0:
                print(f"      ⚠️  NLI returned no scores")
                verdicts.append(Verdict(
                    claim_id=cl.id, label=Label.UNCERTAIN,
                    support_score=0.0, contradiction_score=0.0,
                    rationale="NLI returned no scores."
                ))
                continue

            # Apply calibration
            probs = _calibrate_probs(probs_raw, ALPHA_CAL)
            entail_scores = probs[:, 0]
            contra_scores = probs[:, 2]

            # FIXED: Use robust mean (no trimming for sparse data)
            entail = _robust_mean(entail_scores, TRIM_FRAC)
            contra = _robust_mean(contra_scores, TRIM_FRAC)

            # FIXED: Get adaptive thresholds based on evidence count
            adapt_tau_sup, adapt_tau_con = _adaptive_thresholds(len(cited))
            
            print(f"      Raw scores - entail: {entail_scores}")
            print(f"      Raw scores - contra: {contra_scores}")
            print(f"      Aggregated - entail: {entail:.3f}, contra: {contra:.3f}")
            print(f"      Adaptive thresholds - support: {adapt_tau_sup}, contradict: {adapt_tau_con}")

            # pick exemplar spans for rationale (best entail & best contradict)
            top_ent_i = int(np.argmax(entail_scores))
            top_con_i = int(np.argmax(contra_scores))

            # FIXED: Use adaptive thresholds and smarter margin logic
            if len(cited) < MIN_EVIDENCE:
                label = Label.UNCERTAIN
                rationale = f"Insufficient evidence count ({len(cited)}<{MIN_EVIDENCE}) to assert a verdict."
            else:
                margin_diff = abs(entail - contra)
                print(f"      Margin: {margin_diff:.3f} (threshold: {MARGIN})")
                
                # FIXED: Check if we have a clear winner
                has_clear_winner = margin_diff >= MARGIN
                
                if (entail >= adapt_tau_sup) and (contra < adapt_tau_con) and has_clear_winner:
                    label = Label.SUPPORTED
                    rationale = f"Avg entailment {entail:.2f} exceeds threshold {adapt_tau_sup:.2f} with low contradiction."
                elif (contra >= adapt_tau_con) and (entail < adapt_tau_sup) and has_clear_winner:
                    label = Label.CONTESTED
                    rationale = f"Avg contradiction {contra:.2f} exceeds threshold {adapt_tau_con:.2f} with low entailment."
                elif entail >= adapt_tau_sup and not has_clear_winner:
                    # FIXED: Still mark as SUPPORTED if entail is high even without big margin
                    label = Label.SUPPORTED
                    rationale = f"High entailment {entail:.2f} despite narrow margin."
                elif contra >= adapt_tau_con and not has_clear_winner:
                    label = Label.CONTESTED
                    rationale = f"High contradiction {contra:.2f} despite narrow margin."
                else:
                    label = Label.UNCERTAIN
                    rationale = f"Mixed evidence (ent={entail:.2f}, con={contra:.2f})."

            # Add short, inspectable snippet hints
            ent_snip = premises[top_ent_i][:150].replace("\n", " ")
            con_snip = premises[top_con_i][:150].replace("\n", " ")
            rationale += f" Top-support: \"{ent_snip}...\" Top-contradict: \"{con_snip}...\""

            print(f"      Verdict: {label.value}")

            verdicts.append(Verdict(
                claim_id=cl.id, label=label,
                support_score=entail,
                contradiction_score=contra,
                rationale=rationale
            ))
            
            # Clean up memory after each claim
            gc.collect()
        
        print(f"\n   ✅ All {len(verdicts)} verdicts computed")
        return verdicts
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR in validate_claims_against_spans:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {e}")
        traceback.print_exc()
        raise