"""Vedic (Jyotish) chart engine — sidereal zodiac, Whole Sign Bhava."""
from __future__ import annotations

from typing import Any

import swisseph as swe

from sundial.shared import BirthData, Placement, VedicOptions

from ._swe_setup import SIGNS, degree_in_sign, ensure, sign_of
from .engine import Chart
from .western import _PLANETS, _julian_day

_AYANAMSA_CODES = {
    "lahiri": swe.SIDM_LAHIRI,
    "raman": swe.SIDM_RAMAN,
    "kp": swe.SIDM_KRISHNAMURTI,
}

_RASHIS = SIGNS  # sign names align; interpretation differs

_NAKSHATRAS = (
    "ashwini", "bharani", "krittika", "rohini", "mrigashira", "ardra",
    "punarvasu", "pushya", "ashlesha", "magha", "purva_phalguni", "uttara_phalguni",
    "hasta", "chitra", "swati", "vishakha", "anuradha", "jyeshtha",
    "mula", "purva_ashadha", "uttara_ashadha", "shravana", "dhanishta",
    "shatabhisha", "purva_bhadrapada", "uttara_bhadrapada", "revati",
)


def _nakshatra_of(longitude: float) -> tuple[str, int]:
    """Returns (nakshatra_name, pada 1-4)."""
    n = (longitude % 360) / (360 / 27)
    nak_idx = int(n) % 27
    pada = int((n - int(n)) * 4) + 1
    return _NAKSHATRAS[nak_idx], pada


class VedicEngine:
    system = "vedic"

    def compute(self, birth: BirthData, options: VedicOptions | None = None) -> Chart:
        ensure()
        opts = options or VedicOptions()
        swe.set_sid_mode(_AYANAMSA_CODES[opts.ayanamsa], 0, 0)
        jd = _julian_day(birth)

        ayanamsa_deg = swe.get_ayanamsa_ut(jd)

        planets: dict[str, dict[str, Any]] = {}
        for name, code in _PLANETS.items():
            xx, _ = swe.calc_ut(jd, code, swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED)
            longitude = xx[0] % 360
            nak, pada = _nakshatra_of(longitude)
            planets[name] = {
                "longitude": round(longitude, 6),
                "speed": round(xx[3], 6),
                "rashi": sign_of(longitude),
                "deg_in_rashi": round(degree_in_sign(longitude), 4),
                "nakshatra": nak,
                "pada": pada,
            }

        # South Node (Ketu) opposes North Node (Rahu)
        rahu = planets["north_node"]["longitude"]
        ketu_lon = (rahu + 180) % 360
        nak, pada = _nakshatra_of(ketu_lon)
        planets["south_node"] = {
            "longitude": round(ketu_lon, 6),
            "speed": -planets["north_node"]["speed"],
            "rashi": sign_of(ketu_lon),
            "deg_in_rashi": round(degree_in_sign(ketu_lon), 4),
            "nakshatra": nak,
            "pada": pada,
        }
        planets["rahu"] = planets.pop("north_node")
        planets["ketu"] = planets.pop("south_node")

        # Ascendant in sidereal: tropical asc minus ayanamsa
        cusps, ascmc = swe.houses(jd, birth.lat, birth.lon, b"W")
        sidereal_asc = (ascmc[0] - ayanamsa_deg) % 360
        sidereal_mc = (ascmc[1] - ayanamsa_deg) % 360

        # Whole-sign Bhavas: 1st bhava = sign of ascendant
        asc_sign_idx = int(sidereal_asc) // 30
        for name, p in planets.items():
            planet_sign_idx = int(p["longitude"]) // 30
            p["bhava"] = ((planet_sign_idx - asc_sign_idx) % 12) + 1

        return Chart(
            system="vedic",
            options=opts.model_dump(),
            data={
                "julian_day_ut": jd,
                "ayanamsa_deg": round(ayanamsa_deg, 6),
                "planets": planets,
                "ascendant": round(sidereal_asc, 6),
                "midheaven": round(sidereal_mc, 6),
                "lagna_rashi": sign_of(sidereal_asc),
                "lagna_nakshatra": _nakshatra_of(sidereal_asc)[0],
            },
        )

    def salient_placements(self, chart: Chart, topic: str | None = None) -> list[Placement]:
        out: list[Placement] = []
        for name, p in chart.data["planets"].items():
            out.append(Placement(system="vedic", factor=name, modifier=p["rashi"]))
            out.append(
                Placement(system="vedic", factor=name, modifier=f"bhava_{p['bhava']}")
            )
            out.append(
                Placement(system="vedic", factor=name, modifier=f"nak_{p['nakshatra']}")
            )
        out.append(
            Placement(system="vedic", factor="lagna", modifier=chart.data["lagna_rashi"])
        )
        return out
