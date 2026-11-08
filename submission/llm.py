"""
Optional LLM answer composer.

The local pipeline (solution.py) decides *which* evidence is current and authoritative: it picks the
snapshot, drops stale copies and expired offers, and applies the honesty gate. This module only turns
that pre-filtered evidence into a member-facing answer. Three guards keep it safe to use:

* it never sees stale evidence, so it cannot repeat an old rate;
* every number in its answer must appear in the evidence or the question (otherwise the caller falls
  back to the deterministic local answer);
* answers are cached per (question, as_of), so asking twice gives the same answer.

Backends (first one configured wins):
  1. OpenRouter  - OPENROUTER_API_KEY; plain HTTPS via the standard library (no extra package).
  2. Anthropic   - ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN with the `anthropic` package installed.
With neither, or on any error, the caller keeps the deterministic local answer.
A `.env` file in the working directory or next to this file is read for these variables (it never
overrides variables already set). Overrides: KS_LLM=0 disables the layer, KS_LLM_MODEL picks the model,
KS_LLM_EFFORT the reasoning effort, KS_LLM_TIMEOUT the per-request timeout in seconds.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

import concurrent.futures
import urllib.request
from pathlib import Path


def _load_dotenv() -> None:
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env",
              Path(__file__).resolve().parent.parent / ".env"):
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except OSError:
            continue


_load_dotenv()
EFFORT = os.getenv("KS_LLM_EFFORT", "low")
TIMEOUT_S = float(os.getenv("KS_LLM_TIMEOUT", "1.9"))   # speed is scored on p95 <= 2 s; slow calls fall back
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BACKEND = ("openrouter" if os.getenv("OPENROUTER_API_KEY") else
           "anthropic" if (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) else None)
MODEL = os.getenv("KS_LLM_MODEL", "deepseek/deepseek-v4.1-flash" if BACKEND == "openrouter" else "claude-opus-5-5")

SYSTEM = """You answer credit-union members' questions using ONLY the evidence provided, which comes from \
the credit union's own website as it stood on a given date. The evidence has already been filtered: \
outdated copies and expired offers were removed, and items are ordered most relevant first.

Rules:
- Answer the question directly in 1-4 sentences or a short list. Lead with the specific value asked for.
- Use only facts stated in the evidence. Copy numbers, rates, times, addresses and URLs exactly.
- Keep qualifiers: term (e.g. 60-month), tier (balance or loan amount, model years), rate vs APY vs APR, \
personal vs business, regular vs special/jumbo. Give the tier that matches the question; if the question \
gives no tier and several apply, list the relevant tiers briefly.
- Never mention a past or superseded value, and do not state numbers that are not needed to answer.
- When sources disagree, prefer the higher-authority page (rate sheet > product page > news > blog), then \
the most recently published item, and say so briefly in conflict_note.
- If the evidence does not contain the answer, set confident=false and answer exactly: "That isn't listed \
on the Dupaco website - please contact Dupaco directly." Do not include any numbers in that case.
- If details vary by branch (e.g. hours), give the most common schedule from the evidence and note that \
it varies by branch.
- source_urls: the URLs of the evidence items you actually used."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confident": {"type": "boolean"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
        "conflict_note": {"type": "string"},
    },
    "required": ["answer", "confident", "source_urls", "conflict_note"],
    "additionalProperties": False,
}

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_client = None
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)
_disabled_reason: Optional[str] = None
_use_fallbacks = True


def _norm_num(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers(text: str) -> set[str]:
    return {_norm_num(m) for m in _NUM.findall(text or "")}


def available() -> bool:
    global _client, _disabled_reason
    if _client is not None:
        return True
    if _disabled_reason is not None:
        return False
    if os.getenv("KS_LLM", "1") == "0":
        _disabled_reason = "disabled by KS_LLM=0"
        return False
    if BACKEND is None:
        _disabled_reason = "no credentials"
        return False
    if BACKEND == "openrouter":
        _client = "openrouter"
        return True
    try:
        import anthropic
        _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1)
        return True
    except Exception as e:  # package missing or misconfigured -> local only
        _disabled_reason = f"anthropic unavailable: {e}"
        return False


def _call_openrouter(user: str) -> Optional[str]:
    """OpenRouter chat completions; returns the JSON text of the answer, or None."""
    body = {
        "model": MODEL,
        "max_tokens": 4000,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "member_answer", "strict": True, "schema": SCHEMA}},
        "reasoning": {"effort": EFFORT, "exclude": True},
        "provider": {"require_parameters": True},
    }
    req = urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json", "X-OpenRouter-Title": "knowledge-takehome"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        data = json.loads(r.read().decode())
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") in ("length", "content_filter"):
        return None
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):   # some providers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content


def _call_anthropic(user: str) -> Optional[str]:
    """Anthropic SDK; uses server-side refusal fallbacks when the account supports them."""
    global _use_fallbacks
    import anthropic
    kwargs = dict(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if _use_fallbacks:
        try:
            resp = _client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except (anthropic.BadRequestError, TypeError):
            _use_fallbacks = False      # not supported here: plain request from now on
            resp = None
    if not _use_fallbacks or resp is None:
        resp = _client.messages.create(**kwargs)
    if getattr(resp, "stop_reason", None) in ("refusal", "max_tokens"):
        return None
    return next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)


def compose(question: str, as_of_label: str, evidence: list[dict]) -> Optional[dict]:
    """Return {"answer", "confident", "source_urls", "conflict_note"} or None to use the local answer."""
    if not evidence or not available():
        return None
    blocks = []
    for i, ev in enumerate(evidence, 1):
        meta = f"url={ev['url']} | page_type={ev['ptype']}"
        if ev.get("pub_date"):
            meta += f" | published={ev['pub_date']}"
        blocks.append(f"[{i}] {meta}\n{ev['text']}")
    user = (f"Website as of: {as_of_label}\n\nEvidence:\n\n" + "\n\n".join(blocks) +
            f"\n\nMember question: {question}")
    # Hard wall-clock deadline (socket timeouts alone are per read, not per request).
    fut = _POOL.submit(_call_openrouter if BACKEND == "openrouter" else _call_anthropic, user)
    try:
        text = fut.result(timeout=TIMEOUT_S)
    except Exception:
        return None
    try:
        out = json.loads(text or "")
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)     # tolerate prose around the JSON object
        try:
            out = json.loads(m.group(0)) if m else None
        except Exception:
            out = None
    if not isinstance(out, dict):
        return None
    ans = (out.get("answer") or "").strip()
    if not ans:
        return None
    # Guard: every number must be grounded in the (non-stale) evidence or the question itself.
    allowed = numbers(question)
    for ev in evidence:
        allowed |= numbers(ev["text"]) | numbers(ev["url"])
    if not out.get("confident", True):
        out["answer"] = ans
        return out if not (numbers(ans) - numbers(question)) else None
    if numbers(ans) - allowed:
        return None
    known = {ev["url"] for ev in evidence}
    out["source_urls"] = [u for u in out.get("source_urls", []) if u in known]
    out["answer"] = ans
    return out
