# src/services/llm.py
from __future__ import annotations
import os
import json
import time
import logging
from typing import Any, Dict, Optional, List

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

logger = logging.getLogger(__name__)

# --- MODEL SELECTION --------------------------------------------------------
# Using Claude 3.5 Haiku - the cheapest model that's still very capable
# Cost: $0.80 per million input tokens, $4.00 per million output tokens
# 
# If you want to override, set ANTHROPIC_MODEL in .env
# Example: ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
# ----------------------------------------------------------------------------

DEFAULT_MODEL = "claude-3-5-haiku-20241022"

def _anthropic_client():
    """Return Anthropic client if key present, else None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("[LLM] ANTHROPIC_API_KEY not found in environment")
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.exception("[LLM] Failed to import/init anthropic SDK: %s", e)
        return None

def _content_to_text(blocks: List[Any]) -> str:
    """Extract concatenated text from Anthropic content blocks."""
    parts: List[str] = []
    for b in blocks or []:
        txt = getattr(b, "text", None)
        if isinstance(txt, str):
            parts.append(txt)
        elif isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "".join(parts)

def _strip_fences(s: str) -> str:
    t = s.strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip()
        if t[:4].lower() == "json":
            t = t[4:].strip()
    return t

def llm_json(
    prompt: str,
    system: str = "",
    max_retries: int = 2,
    max_tokens: Optional[int] = None,
    temperature: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Call Claude 3.5 Haiku (cheapest model), expect STRICT JSON.
    Returns None if all attempts fail or JSON is invalid.
    """
    client = _anthropic_client()
    if client is None:
        logger.warning("[LLM] Client unavailable")
        return None

    # Use env override or default to cheap limits
    model_name = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    
    # Allow token cap override via env (to control cost)
    try:
        env_cap = int(os.environ.get("LLM_MAX_TOKENS", "0"))
    except ValueError:
        env_cap = 0
    if max_tokens is None:
        max_tokens = env_cap if env_cap > 0 else 800  # Increased for better claims

    logger.info("[LLM] Using model: %s (max_tokens=%d)", model_name, max_tokens)

    for attempt in range(max_retries + 1):
        try:
            msg = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a careful scientific assistant. Output STRICT JSON only with no code fences.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = _content_to_text(msg.content)
            cleaned = _strip_fences(text)
            
            # Log raw response for debugging
            logger.debug("[LLM] Raw response (first 200 chars): %s", cleaned[:200])
            
            try:
                js = json.loads(cleaned)
                logger.info("[LLM] Success (JSON len=%d)", len(cleaned))
                return js
            except json.JSONDecodeError as je:
                logger.warning("[LLM] JSON parse failed (attempt=%d): %s", attempt + 1, je)
                logger.warning("[LLM] Problematic JSON: %s", cleaned[:300])
        except Exception as e:
            logger.warning("[LLM] Call failed (attempt=%d): %s", attempt + 1, str(e))
        
        if attempt < max_retries:
            time.sleep(0.5)

    logger.warning("[LLM] All attempts failed → returning None")
    return None

if __name__ == "__main__":
    # Smoketest: should print a JSON dict if credentials work
    js = llm_json('Return {"ok": true} exactly.', system="Output STRICT JSON only.", max_retries=0)
    print("SMOKETEST:", js)