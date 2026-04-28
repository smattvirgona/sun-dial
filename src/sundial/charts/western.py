"""Modern tropical Western astrology chart engine."""
from __future__ import annotations

from typing import Any

import swisseph as swe

from sundial.shared import BirthData, Placement, WesternOptions

from ._swe_setup import SIGNS, degree_in_sign, ensure, sign_of
from .engine import Chart

_PLANETS: dict[str, int] = {
    "sun": swe.SUN,
    "moon": swe.MOON,
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
    "north_node": swe.MEAN_NODE,
}

_MAJOR_ASPECTS = {
    "conjunction": 0,
    "sextile": 60,
    "square": 90,
    "trine": 120,
    "opposition": 180,
}
_ORB = 6.0


def _julian_day(birth: BirthData) -> float:
    dt = birth.when_utc
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    return swe.julday(dt.year, dt.month, dt.day, hour, swe.GREG_CAL)


def _houses(jd: float, lat: float, lon: float, system: str) -> tuple[list[float], float, float]:
    code = b"W" if system == "whole_sign" else b"P"
    cusps, ascmc = swe.houses(jd, lat, lon, code)
    return list(cusps), ascmc[0], ascmc[1]  # cusps, asc, mc


def _house_of(longitude: float, ascendant: float, system: str) -> int:
    if system == "whole_sign":
        asc_sign_start = (int(ascendant) // 30) * 30
        diff = (longitude - asc_sign_start) % 360
        return int(diff // 30) + 1
    # Placidus: cusps are precomputed by caller; this branch is unused here.
    raise NotImplementedError("placidus house lookup uses cusps directly")


def _aspects(planets: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    names = list(planets)
    out: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la, lb = planets[a]["longitude"], planets[b]["longitude"]
            sep = abs((la - lb + 180) % 360 - 180)
            for asp_name, asp_deg in _MAJOR_ASPECTS.items():
                if abs(sep - asp_deg) <= _ORB:
                    out.append(
                        {
                            "a": a,
                            "b": b,
                            "aspect": asp_name,
                            "orb": round(abs(sep - asp_deg), 3),
                        }
                    )
                    break
    return out


class WesternEngine:
    system = "western"

    def compute(self, birth: BirthData, options: WesternOptions | None = None) -> Chart:
        ensure()
        opts = options or WesternOptions()
        jd = _julian_day(birth)

        planets: dict[str, dict[str, float]] = {}
        for name, code in _PLANETS.items():
            xx, _ = swe.calc_ut(jd, code, swe.FLG_SWIEPH | swe.FLG_SPEED)
            longitude = xx[0] % 360
            planets[name] = {
                "longitude": round(longitude, 6),
                "speed": round(xx[3], 6),
                "sign": sign_of(longitude),
                "deg_in_sign": round(degree_in_sign(longitude), 4),
            }

        cusps, asc, mc = _houses(jd, birth.lat, birth.lon, opts.houses)
        for name, p in planets.items():
            if opts.houses == "whole_sign":
                p["house"] = _house_of(p["longitude"], asc, "whole_sign")
            else:
                # Placidus: longitude falls in house i if cusp[i] <= lon < cusp[i+1] (mod 360)
                lon = p["longitude"]
                for i in range(12):
                    start = cusps[i]
                    end = cusps[(i + 1) % 12]
                    span = (end - start) % 360
                    delta = (lon - start) % 360
                    if delta < span:
                        p["house"] = i + 1
                        break

        return Chart(
            system="western",
            options=opts.model_dump(),
            data={
                "julian_day_ut": jd,
                "planets": planets,
                "houses": [round(c, 6) for c in cusps],
                "ascendant": round(asc, 6),
                "midheaven": round(mc, 6),
                "asc_sign": sign_of(asc),
                "mc_sign": sign_of(mc),
                "aspects": _aspects(planets),
            },
        )

    def salient_placements(self, chart: Chart, topic: str | None = None) -> list[Placement]:
        out: list[Placement] = []
        for name, p in chart.data["planets"].items():
            out.append(Placement(system="western", factor=name, modifier=p["sign"]))
            out.append(
                Placement(system="western", factor=name, modifier=f"house_{p['house']}")
            )
        out.append(
            Placement(system="western", factor="ascendant", modifier=chart.data["asc_sign"])
        )
        for asp in chart.data["aspects"]:
            out.append(
                Placement(
                    system="western",
                    factor=f"{asp['a']}_{asp['aspect']}_{asp['b']}",
                )
            )
        return out
