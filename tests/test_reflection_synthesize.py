"""Tests for the synthesis path with a mocked Anthropic client.

We assert: (1) prompt structure is cache-friendly (cache_control on the system
voice guide and on the chart-facts user block), (2) the call uses the correct
model + JSON schema, and (3) citation verification drops claims whose chunk_id
is unknown or whose quote isn't a substring of the cited snippet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from sundial.charts import WesternEngine
from sundial.corpus import CorpusChunk
from sundial.reflection import synthesize
from sundial.reflection.synthesize import DEFAULT_MODEL
from sundial.reflection.types import ReflectionRequest
from sundial.shared import BirthData


def _request() -> ReflectionRequest:
    birth = BirthData(
        when_utc=datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc),
        lat=40.6782, lon=-73.9442,
    )
    chart = WesternEngine().compute(birth)
    snippets = [
        CorpusChunk(
            id="c1",
            source_id="ashmand_tetrabiblos_1822",
            system="western",
            factors=["saturn"],
            topics=["duty"],
            text="Saturn makes his subjects sedate, austere, laborious, and exact.",
            license_ring=1,
        ),
        CorpusChunk(
            id="c2",
            source_id="ashmand_tetrabiblos_1822",
            system="western",
            factors=["jupiter"],
            topics=["fortune"],
            text="Jupiter makes his subjects magnanimous and just.",
            license_ring=1,
        ),
    ]
    return ReflectionRequest(
        question="Should I leave my job?",
        chart=chart,
        systems_in_play=["western"],
        snippets=snippets,
    )


class _FakeClient:
    def __init__(self, response_json: dict[str, Any]) -> None:
        self.captured: dict[str, Any] = {}
        self._response_json = response_json
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        text_block = SimpleNamespace(
            type="text", text=json.dumps(self._response_json)
        )
        return SimpleNamespace(content=[text_block])


def test_synthesize_uses_cache_friendly_prompt_and_correct_model() -> None:
    fake = _FakeClient({
        "headline": "Stay grounded; the structure is the work.",
        "body": "You are being asked to hold a long line. Saturn makes his "
                "subjects laborious, and exact — that quality is yours now.",
        "citations": [{"chunk_id": "c1", "quote": "laborious, and exact"}],
    })
    result = synthesize(_request(), client=fake)

    assert fake.captured["model"] == DEFAULT_MODEL
    assert fake.captured["thinking"] == {"type": "adaptive"}
    assert fake.captured["output_config"]["format"]["type"] == "json_schema"

    system = fake.captured["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    user_blocks = fake.captured["messages"][0]["content"]
    assert user_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "CHART FACTS" in user_blocks[0]["text"]
    assert "CORPUS SNIPPETS" in user_blocks[1]["text"]
    assert "CURRENT MOMENT" in user_blocks[2]["text"]
    assert "Should I leave my job?" in user_blocks[3]["text"]
    assert "cache_control" not in user_blocks[3]

    assert result.headline.startswith("Stay")
    assert result.chart_hash
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"


def test_synthesize_drops_unsupported_citations() -> None:
    fake = _FakeClient({
        "headline": "h",
        "body": "b",
        "citations": [
            {"chunk_id": "c1", "quote": "laborious, and exact"},
            {"chunk_id": "c1", "quote": "phrase that is not in the snippet"},
            {"chunk_id": "nonexistent", "quote": "anything"},
        ],
    })
    result = synthesize(_request(), client=fake)
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "c1"
