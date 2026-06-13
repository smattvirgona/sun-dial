from __future__ import annotations

from sundial.journal import JournalStore


def _store() -> JournalStore:
    return JournalStore(":memory:")


def test_save_and_retrieve_roundtrip() -> None:
    s = _store()
    rid = s.save(
        system="western",
        chart_hash="abc123",
        question="should I switch jobs?",
        headline="Stay grounded; the structure is the work.",
        body="You are being asked to hold a long line.",
        citations=[{"chunk_id": "c1", "quote": "laborious, and exact"}],
        transits=[{"transiting": "saturn", "natal": "moon", "aspect": "square", "orb": 0.2}],
    )
    assert rid.startswith("refl_")
    rows = s.recent()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == rid
    assert row["question"] == "should I switch jobs?"
    assert row["citations"][0]["chunk_id"] == "c1"
    assert row["transits"][0]["aspect"] == "square"
    assert isinstance(row["created_at"], float)


def test_recent_filters_by_chart_hash() -> None:
    s = _store()
    s.save(system="western", chart_hash="A", question="q1",
           headline="h1", body="b", citations=[], transits=[])
    s.save(system="western", chart_hash="B", question="q2",
           headline="h2", body="b", citations=[], transits=[])
    s.save(system="vedic", chart_hash="A", question="q3",
           headline="h3", body="b", citations=[], transits=[])
    a = s.recent(chart_hash="A")
    assert {r["question"] for r in a} == {"q1", "q3"}
    b = s.recent(chart_hash="B")
    assert [r["question"] for r in b] == ["q2"]


def test_recent_orders_newest_first() -> None:
    s = _store()
    ids = [
        s.save(system="western", chart_hash="A", question=f"q{i}",
               headline=None, body=None, citations=[], transits=[])
        for i in range(5)
    ]
    rows = s.recent()
    assert [r["id"] for r in rows] == list(reversed(ids))


def test_recent_respects_limit() -> None:
    s = _store()
    for i in range(30):
        s.save(system="western", chart_hash="A", question=f"q{i}",
               headline=None, body=None, citations=[], transits=[])
    assert len(s.recent(limit=5)) == 5
    assert len(s.recent(limit=50)) == 30


def test_dry_run_persists_question_with_null_headline() -> None:
    """The /reflect dry-run path saves headline=None — we should still
    surface the question so the history list has something to render."""
    s = _store()
    s.save(system="western", chart_hash="A", question="hm",
           headline=None, body=None, citations=[], transits=[])
    row = s.recent()[0]
    assert row["question"] == "hm"
    assert row["headline"] is None
    assert row["citations"] == []
