"""Prompt-shape contract test. The Anthropic call is added in a later slice.
This locks the cached-prefix structure so the eventual API call is cache-friendly.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sundial.charts import WesternEngine
from sundial.corpus import CorpusChunk
from sundial.reflection import build_messages
from sundial.reflection.types import ReflectionRequest
from sundial.shared import BirthData


def test_messages_have_cached_prefix_and_variable_question() -> None:
    birth = BirthData(
        when_utc=datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc),
        lat=40.6782,
        lon=-73.9442,
    )
    chart = WesternEngine().compute(birth)
    snippets = [
        CorpusChunk(
            id="c1",
            source_id="ashmand_tetrabiblos_1822",
            system="western",
            factors=["saturn", "10th_house"],
            topics=["career"],
            text="Saturn placed in the tenth house tends to ...",
            license_ring=1,
        )
    ]
    req = ReflectionRequest(
        question="Should I switch jobs?",
        chart=chart,
        systems_in_play=["western"],
        snippets=snippets,
    )
    messages = build_messages(req)
    assert len(messages) == 1
    contents = messages[0]["content"]
    assert len(contents) == 2
    assert contents[0].get("cache_control") == {"type": "ephemeral"}
    assert "CHART FACTS" in contents[0]["text"]
    assert "CORPUS SNIPPETS" in contents[0]["text"]
    assert "Should I switch jobs?" in contents[1]["text"]
    assert "cache_control" not in contents[1]
