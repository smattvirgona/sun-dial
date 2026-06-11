"""Prompt assembly for the reflection synthesis call.

Caching strategy (prefix-match):
- `system`: voice guide — stable, cached.
- First user content block: chart facts — stable per-user across reflections, cached.
- Remaining user blocks (snippets + question) — vary per request, not cached.

This keeps two breakpoints; the chart prefix becomes a hot cache for any returning user.
"""
from __future__ import annotations

import json
from typing import Any

from .types import ReflectionRequest

VOICE_GUIDE = """You write in two registers, layered:

- HEADLINE (one sentence, ≤14 words): pithy, declarative, Co-Star-style.
  No filler. Address the reader as "you". No emoji.
- BODY (~150 words): lyrical, slow, second-person, CHANI-style. Each
  interpretive claim must trace to a CITATION from the supplied snippets.
  Do not invent astrological correspondences. If the snippets don't cover
  the question, say so plainly in the body.

Return JSON only, with this shape:
{
  "headline": "...",
  "body": "...",
  "citations": [{"chunk_id": "...", "quote": "..."}]
}"""


def _chart_facts(req: ReflectionRequest) -> str:
    return json.dumps(
        {
            "system": req.chart.system,
            "options": req.chart.options,
            "data": req.chart.data,
        },
        sort_keys=True,
        default=str,
        indent=2,
    )


def _snippets_text(req: ReflectionRequest) -> str:
    parts = []
    for c in req.snippets:
        parts.append(
            f"[{c.id}] (system={c.system}, factors={c.factors})\n{c.text.strip()}"
        )
    return "\n\n".join(parts)


def build_system(req: ReflectionRequest) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": VOICE_GUIDE,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def build_messages(req: ReflectionRequest) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"=== CHART FACTS ===\n{_chart_facts(req)}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"=== CORPUS SNIPPETS ===\n{_snippets_text(req)}",
                },
                {
                    "type": "text",
                    "text": f"=== QUESTION ===\n{req.question}",
                },
            ],
        }
    ]


REFLECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "body": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["chunk_id", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["headline", "body", "citations"],
    "additionalProperties": False,
}
