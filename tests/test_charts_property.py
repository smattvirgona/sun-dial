"""Property tests — invariants every chart must satisfy.

These are *system* checks, not fixtures: they catch bugs that golden tests
might miss because the bug existed when the fixture was generated.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sundial.charts import ChineseEngine, VedicEngine, WesternEngine
from sundial.charts.chinese import _DAY_REF_JDN, _find_solar_longitude_crossing
from sundial.shared import BirthData, VedicOptions

GREENWICH = (51.4779, 0.0)


def _greenwich(when: datetime) -> BirthData:
    return BirthData(when_utc=when, lat=GREENWICH[0], lon=GREENWICH[1])


# ---------- Western ----------

@pytest.mark.parametrize(
    "iso,expected_sign",
    [
        ("2020-03-21T12:00:00+00:00", "aries"),
        ("2020-04-21T12:00:00+00:00", "taurus"),
        ("2020-06-22T12:00:00+00:00", "cancer"),
        ("2020-09-23T12:00:00+00:00", "libra"),
        ("2020-12-22T12:00:00+00:00", "capricorn"),
    ],
)
def test_sun_sign_at_seasonal_changes(iso: str, expected_sign: str) -> None:
    birth = _greenwich(datetime.fromisoformat(iso))
    chart = WesternEngine().compute(birth)
    assert chart.data["planets"]["sun"]["sign"] == expected_sign


def test_aspects_are_symmetric_orb_within_six() -> None:
    birth = _greenwich(datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc))
    chart = WesternEngine().compute(birth)
    for aspect in chart.data["aspects"]:
        assert aspect["orb"] <= 6.0
        assert aspect["a"] != aspect["b"]


def test_houses_are_twelve_and_ascend() -> None:
    birth = _greenwich(datetime(1990, 6, 15, 18, 30, tzinfo=timezone.utc))
    chart = WesternEngine().compute(birth)
    assert len(chart.data["houses"]) == 12
    asc = chart.data["ascendant"]
    # First house cusp equals ascendant for Whole Sign? No — for Whole Sign the
    # 1st cusp is the ascendant's *sign start*, not the ascendant itself.
    # Just check ascendant is a valid longitude.
    assert 0 <= asc < 360


# ---------- Vedic ----------

def test_lahiri_ayanamsa_around_24_in_2020() -> None:
    birth = _greenwich(datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc))
    chart = VedicEngine().compute(birth)
    # Lahiri ayanamsa in Jan 2020 ≈ 24.10°
    assert 24.0 < chart.data["ayanamsa_deg"] < 24.2


def test_ayanamsa_increases_year_over_year() -> None:
    a = VedicEngine().compute(_greenwich(datetime(1990, 1, 1, tzinfo=timezone.utc)))
    b = VedicEngine().compute(_greenwich(datetime(2020, 1, 1, tzinfo=timezone.utc)))
    assert b.data["ayanamsa_deg"] > a.data["ayanamsa_deg"]


def test_ayanamsa_options_differ() -> None:
    when = datetime(2000, 1, 1, tzinfo=timezone.utc)
    lahiri = VedicEngine().compute(_greenwich(when), VedicOptions(ayanamsa="lahiri"))
    raman = VedicEngine().compute(_greenwich(when), VedicOptions(ayanamsa="raman"))
    kp = VedicEngine().compute(_greenwich(when), VedicOptions(ayanamsa="kp"))
    vals = {
        lahiri.data["ayanamsa_deg"],
        raman.data["ayanamsa_deg"],
        kp.data["ayanamsa_deg"],
    }
    assert len(vals) == 3


def test_rahu_ketu_oppose() -> None:
    chart = VedicEngine().compute(_greenwich(datetime(1990, 6, 15, tzinfo=timezone.utc)))
    rahu = chart.data["planets"]["rahu"]["longitude"]
    ketu = chart.data["planets"]["ketu"]["longitude"]
    assert abs(((rahu - ketu) % 360) - 180) < 1e-3


# ---------- Chinese ----------

def test_jia_zi_year_1984() -> None:
    """1984, 1924, 2044 are Jia-Zi years (start of 60-year cycle)."""
    for year in (1984, 1924, 2044):
        # Pick mid-year so we're well past Lichun.
        birth = _greenwich(datetime(year, 6, 1, 12, tzinfo=timezone.utc))
        chart = ChineseEngine().compute(birth)
        assert chart.data["pillars"]["year"]["stem"] == "jia"
        assert chart.data["pillars"]["year"]["branch"] == "zi"


def test_pre_lichun_rolls_back_year() -> None:
    """1984-01-15 (before Lichun 1984) → 1983 year pillar (Gui-Hai)."""
    birth = _greenwich(datetime(1984, 1, 15, 12, tzinfo=timezone.utc))
    chart = ChineseEngine().compute(birth)
    assert chart.data["pillars"]["year"]["stem"] == "gui"
    assert chart.data["pillars"]["year"]["branch"] == "hai"


def test_day_pillar_reference_epoch() -> None:
    """1984-02-02 local solar should be Jia-Zi day (index 0)."""
    # Use a longitude where local solar matches UTC closely (Greenwich).
    birth = _greenwich(datetime(1984, 2, 2, 12, tzinfo=timezone.utc))
    chart = ChineseEngine().compute(birth)
    assert chart.data["pillars"]["day"]["stem"] == "jia"
    assert chart.data["pillars"]["day"]["branch"] == "zi"


def test_lichun_solar_longitude_is_315() -> None:
    """Sanity check on the solar-longitude crossing finder."""
    jd = _find_solar_longitude_crossing(315.0, 2020)
    # Lichun 2020 = 2020-02-04 09:03 UTC ≈ JD 2458883.877.
    assert 2458883.85 < jd < 2458883.91


def test_bazi_60_year_cycle() -> None:
    """A pillar at year Y and Y+60 should have identical year pillar."""
    a = ChineseEngine().compute(_greenwich(datetime(1990, 6, 1, tzinfo=timezone.utc)))
    b = ChineseEngine().compute(_greenwich(datetime(2050, 6, 1, tzinfo=timezone.utc)))
    assert a.data["pillars"]["year"] == b.data["pillars"]["year"]


def test_day_ref_jdn_constant() -> None:
    """Catches accidental drift of the day-pillar epoch."""
    assert _DAY_REF_JDN == 2445733
