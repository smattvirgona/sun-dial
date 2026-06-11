"""Transit math sanity. The orb is tight, so any 'active' transit means
two longitudes really are within ~1.5° of one of five fixed angles."""
from __future__ import annotations

from datetime import datetime, timezone

from sundial.charts import (
    WesternEngine,
    active_transits,
    current_sky,
    transit_placement_tags,
)
from sundial.shared import BirthData

BIRTH = BirthData(
    when_utc=datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc),
    lat=40.6782,
    lon=-73.9442,
)


def test_current_sky_uses_provided_now() -> None:
    now = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
    a = current_sky(BIRTH, now=now)
    b = current_sky(BIRTH, now=now)
    assert a.chart_hash == b.chart_hash


def test_transits_are_within_orb() -> None:
    natal = WesternEngine().compute(BIRTH)
    sky = current_sky(BIRTH, now=datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc))
    transits = active_transits(natal, sky)
    for t in transits:
        assert t["orb"] <= 1.5
        assert t["aspect"] in {
            "conjunction", "sextile", "square", "trine", "opposition"
        }


def test_transit_self_pair_only_returns_or_opposes() -> None:
    # Birth's natal Sun is at ~84.4°; one full solar return per year. Five
    # years on, the transiting Sun crosses natal Sun on the user's birthday.
    natal = WesternEngine().compute(BIRTH)
    on_birthday = datetime(2024, 6, 15, 18, 30, tzinfo=timezone.utc)
    sky = current_sky(BIRTH, now=on_birthday)
    transits = active_transits(natal, sky)
    sun_self = [t for t in transits if t["transiting"] == "sun" and t["natal"] == "sun"]
    assert sun_self, "transiting Sun should conjoin natal Sun on the birthday"
    assert sun_self[0]["aspect"] in {"conjunction", "opposition"}


def test_placement_tags_include_planet_and_aspect_strings() -> None:
    transits = [
        {"transiting": "saturn", "natal": "moon", "aspect": "square", "orb": 0.2}
    ]
    assert set(transit_placement_tags(transits)) == {
        "saturn", "saturn_square_moon",
    }
