"""Chinese BaZi (Four Pillars of Destiny) engine.

Pillar boundaries:
- **Year**: Lichun (sun longitude 315°), NOT lunar new year.
- **Month**: the 12 "joint" solar terms (every 30° of tropical sun longitude
  starting at 315°). BaZi months are *not* Chinese lunar months.
- **Day**: 60-day sexagenary cycle; reference epoch 1984-02-02 = Jia-Zi (index 0).
  Day boundary at local mean solar midnight (lon/15 hours from UTC).
  Late-zi convention: 23:00–23:59 belongs to the next day's pillar.
- **Hour**: 12 two-hour blocks of local mean solar time, late-zi.

Stems and branches are stored as 0-indexed integers plus pinyin labels.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import swisseph as swe

from sundial.shared import BirthData, ChineseOptions, Placement

from ._swe_setup import ensure
from .engine import Chart

STEMS = ("jia", "yi", "bing", "ding", "wu", "ji", "geng", "xin", "ren", "gui")
BRANCHES = (
    "zi", "chou", "yin", "mao", "chen", "si",
    "wu_h", "wei", "shen", "you", "xu", "hai",
)
STEM_ELEMENTS = (
    "wood", "wood", "fire", "fire", "earth",
    "earth", "metal", "metal", "water", "water",
)
STEM_YIN_YANG = ("yang", "yin") * 5
BRANCH_ANIMALS = (
    "rat", "ox", "tiger", "rabbit", "dragon", "snake",
    "horse", "goat", "monkey", "rooster", "dog", "pig",
)
BRANCH_ELEMENTS = (
    "water", "earth", "wood", "wood", "earth", "fire",
    "fire", "earth", "metal", "metal", "earth", "water",
)

# Reference: 1984-02-02 UTC noon → JDN 2445733 → day index 0 (Jia-Zi)
_DAY_REF_JDN = 2445733
# Five Tiger Pivot: month stem at Yin given year stem
_MONTH_STEM_AT_YIN = {0: 2, 5: 2, 1: 4, 6: 4, 2: 6, 7: 6, 3: 8, 8: 8, 4: 0, 9: 0}
# Five Mouse Pivot: hour stem at Zi given day stem
_HOUR_STEM_AT_ZI = {0: 0, 5: 0, 1: 2, 6: 2, 2: 4, 7: 4, 3: 6, 8: 6, 4: 8, 9: 8}


def _sun_longitude(jd_ut: float) -> float:
    xx, _ = swe.calc_ut(jd_ut, swe.SUN, swe.FLG_SWIEPH)
    return xx[0] % 360


def _find_solar_longitude_crossing(target: float, year: int) -> float:
    """Find UT JD in `year` when sun ecliptic longitude crosses `target`.

    Sun longitude grows ~0.985°/day, so we scan day-by-day for a sign change
    in (longitude - target) wrapped into (-180, 180], then bisect a 1-day
    window. Works for any target in [0, 360).
    """
    def diff(jd: float) -> float:
        return (_sun_longitude(jd) - target + 180) % 360 - 180

    start = swe.julday(year, 1, 1, 0, swe.GREG_CAL)
    a = start
    fa = diff(a)
    for day in range(1, 380):
        b = start + day
        fb = diff(b)
        # Look for sign change with a small magnitude (sun moves ~1°/day, so
        # a true crossing has |fa|, |fb| both small; jumps of ~360° around
        # the wraparound have |fb-fa| close to 360 — reject those).
        if fa * fb < 0 and abs(fb - fa) < 5:
            for _ in range(60):
                mid = (a + b) / 2
                fm = diff(mid)
                if abs(fm) < 1e-7:
                    return mid
                if fa * fm < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            return (a + b) / 2
        a, fa = b, fb
    raise RuntimeError(f"could not find solar longitude {target} in {year}")


def _month_branch_index(sun_long: float) -> int:
    """Joint terms at 315°, 345°, 15°, ..., bucket → branch index.

    315°→Yin(2), 345°→Mao(3), 15°→Chen(4), ..., 285°→Chou(1).
    """
    bucketed = int(((sun_long - 315) % 360) // 30)
    return (bucketed + 2) % 12


def _local_solar_dt(birth: BirthData):
    return birth.when_utc + timedelta(hours=birth.lon / 15.0)


def _bazi_year(birth: BirthData) -> int:
    """Solar year of the year pillar (rolls back if before Lichun)."""
    local_dt = _local_solar_dt(birth)
    solar_year = local_dt.year
    jd = swe.julday(
        birth.when_utc.year, birth.when_utc.month, birth.when_utc.day,
        birth.when_utc.hour + birth.when_utc.minute / 60,
        swe.GREG_CAL,
    )
    lichun_jd = _find_solar_longitude_crossing(315.0, solar_year)
    if jd < lichun_jd:
        return solar_year - 1
    return solar_year


def _day_index(birth: BirthData) -> int:
    local_dt = _local_solar_dt(birth)
    # Late-zi: 23:00-23:59 rolls forward.
    if local_dt.hour >= 23:
        local_dt = local_dt + timedelta(hours=1)
    jdn = swe.julday(local_dt.year, local_dt.month, local_dt.day, 12, swe.GREG_CAL)
    return int(round(jdn - _DAY_REF_JDN)) % 60


def _hour_branch_index(birth: BirthData) -> int:
    local_dt = _local_solar_dt(birth)
    h = local_dt.hour
    # 23-1 → zi (0), 1-3 → chou (1), ...
    return ((h + 1) // 2) % 12


def _pillar(stem_idx: int, branch_idx: int) -> dict[str, Any]:
    return {
        "stem": STEMS[stem_idx],
        "stem_index": stem_idx,
        "stem_element": STEM_ELEMENTS[stem_idx],
        "stem_yin_yang": STEM_YIN_YANG[stem_idx],
        "branch": BRANCHES[branch_idx],
        "branch_index": branch_idx,
        "branch_animal": BRANCH_ANIMALS[branch_idx],
        "branch_element": BRANCH_ELEMENTS[branch_idx],
    }


class ChineseEngine:
    system = "chinese"

    def compute(self, birth: BirthData, options: ChineseOptions | None = None) -> Chart:
        ensure()
        opts = options or ChineseOptions()

        year = _bazi_year(birth)
        # Year pillar: 1984 → Jia-Zi; offset = (year - 1984) mod cycle
        year_stem = (year - 1984) % 10
        year_branch = (year - 1984) % 12

        jd_birth = swe.julday(
            birth.when_utc.year, birth.when_utc.month, birth.when_utc.day,
            birth.when_utc.hour + birth.when_utc.minute / 60 + birth.when_utc.second / 3600,
            swe.GREG_CAL,
        )
        sun_long = _sun_longitude(jd_birth)
        month_branch = _month_branch_index(sun_long)
        # Five Tiger Pivot: month stem at Yin (branch 2), then walk forward.
        base_month_stem = _MONTH_STEM_AT_YIN[year_stem]
        month_stem = (base_month_stem + (month_branch - 2)) % 10

        day_idx = _day_index(birth)
        day_stem = day_idx % 10
        day_branch = day_idx % 12

        hour_branch = _hour_branch_index(birth)
        base_hour_stem = _HOUR_STEM_AT_ZI[day_stem]
        hour_stem = (base_hour_stem + hour_branch) % 10

        pillars = {
            "year": _pillar(year_stem, year_branch),
            "month": _pillar(month_stem, month_branch),
            "day": _pillar(day_stem, day_branch),
            "hour": _pillar(hour_stem, hour_branch),
        }

        return Chart(
            system="chinese",
            options=opts.model_dump(),
            data={
                "pillars": pillars,
                "day_master": pillars["day"]["stem"],
                "day_master_element": pillars["day"]["stem_element"],
                "sun_longitude_at_birth": round(sun_long, 6),
                "bazi_solar_year": year,
            },
        )

    def salient_placements(self, chart: Chart, topic: str | None = None) -> list[Placement]:
        out: list[Placement] = []
        for pillar_name, p in chart.data["pillars"].items():
            out.append(
                Placement(
                    system="chinese",
                    factor=pillar_name,
                    modifier=f"{p['stem']}_{p['branch']}",
                )
            )
        out.append(
            Placement(
                system="chinese",
                factor="day_master",
                modifier=chart.data["day_master_element"],
            )
        )
        return out
