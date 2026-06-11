"""Transits — aspects between *current* sky positions and a natal chart.

The chart engines compute longitudes at any UT instant; a transit is just
the angular separation between a transiting body and a natal body. We use
the same five major aspects as the natal aspect grid, but with a tighter
orb (default 1.5°) — moment-of-now influences are sharp, not diffuse.

Why this matters for the app: "a moment to reflect" only reads as a moment
if the reflection is shaped by *this* moment. Otherwise the same question
at noon and midnight gets the same answer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sundial.shared import BirthData

from .engine import Chart
from .western import _MAJOR_ASPECTS, WesternEngine

_TRANSIT_ORB = 1.5


def current_sky(birth: BirthData, now: datetime | None = None) -> Chart:
    """The Western chart for `now`, computed at the user's birth location.

    Location only changes the ascendant/houses, which we do not use for
    transit aspects — but keeping the same lat/lon makes the chart_hash
    semantics consistent across natal and current calls.
    """
    moment = now or datetime.now(timezone.utc)
    moment_utc = moment.astimezone(timezone.utc) if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return WesternEngine().compute(
        BirthData(when_utc=moment_utc, lat=birth.lat, lon=birth.lon)
    )


def active_transits(
    natal: Chart, sky: Chart, *, orb: float = _TRANSIT_ORB
) -> list[dict[str, Any]]:
    """Aspects formed by transiting planets to natal planets, within `orb`.

    Self-pairs (transiting Sun to natal Sun) are included only when the
    aspect is conjunction or opposition — the meaningful self-returns.
    """
    out: list[dict[str, Any]] = []
    for t_name, t in sky.data["planets"].items():
        for n_name, n in natal.data["planets"].items():
            sep = abs((t["longitude"] - n["longitude"] + 180) % 360 - 180)
            for asp_name, asp_deg in _MAJOR_ASPECTS.items():
                if abs(sep - asp_deg) > orb:
                    continue
                if t_name == n_name and asp_name not in ("conjunction", "opposition"):
                    continue
                out.append({
                    "transiting": t_name,
                    "natal": n_name,
                    "aspect": asp_name,
                    "orb": round(abs(sep - asp_deg), 3),
                })
                break
    out.sort(key=lambda d: d["orb"])
    return out


def transit_placement_tags(transits: list[dict[str, Any]]) -> list[str]:
    """Tags retrieval can index on, e.g. `saturn` (the transiting body) or
    `saturn_conjunct_moon` (the dynamic aspect). The plain transiting-body
    tag is included because most corpus content keys off the planet alone."""
    tags: list[str] = []
    for t in transits:
        tags.append(t["transiting"])
        tags.append(f"{t['transiting']}_{t['aspect']}_{t['natal']}")
    return tags
