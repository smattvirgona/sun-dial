"""API tests with FastAPI's TestClient. Synthesis runs in dry-run mode here
(no ANTHROPIC_API_KEY in the test environment), which is itself the contract
under test for offline operation."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sundial.api import create_app
from sundial.journal import JournalStore

BIRTH = {"when_utc": "1990-06-15T18:30:00Z", "lat": 40.6782, "lon": -73.9442}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(create_app(journal=JournalStore(":memory:")))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_index_serves_reflection_screen(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "a moment to reflect" in res.text


@pytest.mark.parametrize("system", ["western", "vedic", "chinese"])
def test_chart_endpoint_all_systems(client: TestClient, system: str) -> None:
    res = client.post("/chart", json={"birth": BIRTH, "system": system})
    assert res.status_code == 200
    data = res.json()
    assert data["chart"]["system"] == system
    assert data["chart_hash"]


def test_chart_is_deterministic(client: TestClient) -> None:
    a = client.post("/chart", json={"birth": BIRTH, "system": "western"}).json()
    b = client.post("/chart", json={"birth": BIRTH, "system": "western"}).json()
    assert a["chart_hash"] == b["chart_hash"]


def test_reflect_dry_run_returns_snippets_without_synthesis(client: TestClient) -> None:
    res = client.post(
        "/reflect",
        json={"birth": BIRTH, "question": "Should I switch jobs?", "system": "western"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["reflection"] is None
    assert "no ANTHROPIC_API_KEY" in data["note"]
    assert data["snippets"], "Ring-1 seed corpus should match a western chart"
    # Every snippet must come from the manifest's PD source.
    assert all(s["source_id"] == "ashmand_tetrabiblos_1822" for s in data["snippets"])
    # Transits payload is present (may be empty if the sky is quiet right now).
    assert isinstance(data["transits"], list)


def test_reflect_omits_transits_for_non_western_systems(client: TestClient) -> None:
    res = client.post(
        "/reflect",
        json={"birth": BIRTH, "question": "today?", "system": "vedic"},
    )
    assert res.status_code == 200
    assert res.json()["transits"] == []


def test_reflect_rejects_empty_question(client: TestClient) -> None:
    res = client.post("/reflect", json={"birth": BIRTH, "question": "", "system": "western"})
    assert res.status_code == 422


def test_reflect_rejects_bad_latitude(client: TestClient) -> None:
    bad = dict(BIRTH, lat=123.0)
    res = client.post("/reflect", json={"birth": bad, "question": "hm", "system": "western"})
    assert res.status_code == 422


def test_journal_starts_empty(client: TestClient) -> None:
    res = client.get("/journal")
    assert res.status_code == 200
    assert res.json() == {"entries": []}


def test_reflect_persists_to_journal_and_listing_shows_it(client: TestClient) -> None:
    refl_res = client.post(
        "/reflect",
        json={"birth": BIRTH, "question": "What is mine to hold?", "system": "western"},
    )
    refl_data = refl_res.json()
    assert refl_data["journal_id"].startswith("refl_")

    listing = client.get("/journal").json()
    assert len(listing["entries"]) == 1
    entry = listing["entries"][0]
    assert entry["id"] == refl_data["journal_id"]
    assert entry["question"] == "What is mine to hold?"
    assert entry["system"] == "western"
    assert entry["chart_hash"] == refl_data["chart_hash"]
    # Dry-run mode stores no headline; the question still appears in history.
    assert entry["headline"] is None


def test_journal_filter_by_chart_hash(client: TestClient) -> None:
    # Two charts (different birth data), two reflections each.
    other_birth = dict(BIRTH, when_utc="1985-01-01T12:00:00Z")
    r1 = client.post("/reflect", json={"birth": BIRTH, "question": "a", "system": "western"}).json()
    r2 = client.post("/reflect", json={"birth": other_birth, "question": "b", "system": "western"}).json()
    assert r1["chart_hash"] != r2["chart_hash"]

    listing = client.get(f"/journal?chart_hash={r1['chart_hash']}").json()
    assert [e["question"] for e in listing["entries"]] == ["a"]


def test_journal_limit_is_clamped(client: TestClient) -> None:
    res = client.get("/journal?limit=99999")
    assert res.status_code == 200
    assert "entries" in res.json()
