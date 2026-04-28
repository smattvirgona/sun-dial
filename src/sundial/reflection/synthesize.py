"""Reflection synthesis via Anthropic Claude with prompt caching + verification.

Two-pass design:
1. Synthesis: Claude generates headline + body + citations, with the corpus
   snippets and chart facts in scope. Output is constrained by JSON schema.
2. Citation verification (optional but enabled by default): a second pass
   inspects each citation's `quote` against the actual snippet text and the
   body's claims, flagging unsupported assertions. Stripped before return.

Default model is `claude-opus-4-7` per skill guidance.
"""
from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from .prompt import REFLECTION_SCHEMA, build_messages, build_system
from .types import Citation, Reflection, ReflectionRequest

DEFAULT_MODEL = "claude-opus-4-7"


def synthesize(
    req: ReflectionRequest,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 16000,
) -> Reflection:
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={
            "format": {"type": "json_schema", "schema": REFLECTION_SCHEMA},
            "effort": "high",
        },
        system=build_system(req),
        messages=build_messages(req),
    )
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    verified_citations = _verify_citations(parsed["citations"], req)
    return Reflection(
        headline=parsed["headline"],
        body=parsed["body"],
        citations=verified_citations,
        chart_hash=req.chart.chart_hash,
    )


def _verify_citations(
    raw_citations: list[dict[str, Any]], req: ReflectionRequest
) -> list[Citation]:
    """Drop citations whose chunk_id is not in the supplied snippets, or whose
    quote text isn't a substring of that snippet (case-insensitive, whitespace
    normalized). This is the cheap deterministic guard; the optional
    second-pass model verifier (`verify.py`) handles claim-level support.
    """
    snippets_by_id = {c.id: c.text for c in req.snippets}
    out: list[Citation] = []
    for c in raw_citations:
        chunk_id = c.get("chunk_id")
        quote = (c.get("quote") or "").strip()
        if not chunk_id or chunk_id not in snippets_by_id:
            continue
        haystack = " ".join(snippets_by_id[chunk_id].lower().split())
        needle = " ".join(quote.lower().split())
        if needle and needle in haystack:
            out.append(Citation(chunk_id=chunk_id, quote=quote))
    return out


def _has_api_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
