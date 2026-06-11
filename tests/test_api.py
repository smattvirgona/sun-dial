"""API tests with FastAPI's TestClient. Synthesis runs in dry-run mode here
(no ANTHROPIC_API_KEY in the test environment), which is itself the contract
under test for offline operation."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sundial.api import create_app

BIRTH = {"when_utc": "1990-06-15T18:30:00Z", "lat": 40.6782, "lon": -73.9442}


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return TestClient(create_app())


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
