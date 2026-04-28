"""Prompt assembly. Cached prefix (voice + corpus) + variable suffix (question).

The Anthropic call is added in the next slice — this module only builds the
message structure so the test suite can assert prompt shape without an API key.
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

Structure of your JSON response (and only this):
{
  "headline": "...",
  "body": "...",
  "citations": [{"chunk_id": "...", "quote": "..."}]
}"""


def _chart_facts(req: ReflectionRequest) -> str:
    """Compact, deterministic textual rendering of the chart for the prompt."""
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


def _snippets(req: ReflectionRequest) -> str:
    parts = []
    for c in req.snippets:
        parts.append(
            f"[{c.id}] (system={c.system}, factors={c.factors})\n{c.text.strip()}"
        )
    return "\n\n".join(parts)


def build_messages(req: ReflectionRequest) -> list[dict[str, Any]]:
    """Returns Anthropic-style messages with `cache_control` markers.

    System block + voice guide + chart facts + snippets are all marked for
    caching. The user's question is the only variable suffix.
    """
    cached_prefix = (
        f"{VOICE_GUIDE}\n\n"
        f"=== CHART FACTS ===\n{_chart_facts(req)}\n\n"
        f"=== CORPUS SNIPPETS ===\n{_snippets(req)}"
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"=== QUESTION ===\n{req.question}",
                },
            ],
        }
    ]
