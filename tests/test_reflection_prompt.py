"""Prompt-shape contract test: locks the cache-friendly structure of the
synthesis prompt so any change has to go through this test."""
from __future__ import annotations

from datetime import datetime, timezone

from sundial.charts import WesternEngine
from sundial.corpus import CorpusChunk
from sundial.reflection import build_messages, build_system
from sundial.reflection.types import ReflectionRequest
from sundial.shared import BirthData


def test_system_voice_guide_is_cached() -> None:
    birth = BirthData(
        when_utc=datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc),
        lat=40.6782, lon=-73.9442,
    )
    chart = WesternEngine().compute(birth)
    req = ReflectionRequest(
        question="q",
        chart=chart,
        systems_in_play=["western"],
        snippets=[],
    )
    system = build_system(req)
    assert len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "HEADLINE" in system[0]["text"]


def test_user_messages_have_chart_cached_question_volatile() -> None:
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
    assert len(contents) == 3

    # Chart facts: cached, stable per user.
    assert contents[0]["cache_control"] == {"type": "ephemeral"}
    assert "CHART FACTS" in contents[0]["text"]

    # Snippets: vary per question, no cache marker.
    assert "CORPUS SNIPPETS" in contents[1]["text"]
    assert "cache_control" not in contents[1]

    # Question: variable suffix, no cache marker.
    assert "Should I switch jobs?" in contents[2]["text"]
    assert "cache_control" not in contents[2]
