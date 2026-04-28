"""Cross-verify Western longitudes against flatlib.

flatlib is a separate Swiss-Ephemeris-using library with its own implementation
path. Agreement to <0.01° catches whole classes of bugs that golden-file tests
cannot (e.g. wrong unit, frame mix-up, off-by-one body code).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from flatlib import const
from flatlib.chart import Chart as FLChart
from flatlib.datetime import Datetime as FLDatetime
from flatlib.geopos import GeoPos as FLGeoPos

from sundial.charts import WesternEngine
from sundial.shared import BirthData

_BODIES = [
    ("sun", const.SUN),
    ("moon", const.MOON),
    ("mercury", const.MERCURY),
    ("venus", const.VENUS),
    ("mars", const.MARS),
    ("jupiter", const.JUPITER),
    ("saturn", const.SATURN),
]


def _flatlib_chart(birth: BirthData) -> FLChart:
    dt = birth.when_utc
    fl_date = f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d}"
    fl_time = f"{dt.hour:02d}:{dt.minute:02d}"
    fl_dt = FLDatetime(fl_date, fl_time, "+00:00")
    fl_pos = FLGeoPos(birth.lat, birth.lon)
    return FLChart(fl_dt, fl_pos, IDs=[b for _, b in _BODIES])


@pytest.mark.parametrize(
    "iso,lat,lon",
    [
        ("1879-03-14T10:50:00+00:00", 48.4011, 9.9876),
        ("2000-01-01T00:00:00+00:00", 51.4779, 0.0),
        ("1990-06-15T18:30:00+00:00", 40.6782, -73.9442),
    ],
)
def test_western_longitudes_match_flatlib(iso: str, lat: float, lon: float) -> None:
    birth = BirthData(when_utc=datetime.fromisoformat(iso), lat=lat, lon=lon)
    sundial_chart = WesternEngine().compute(birth)
    fl_chart = _flatlib_chart(birth)

    for our_name, fl_id in _BODIES:
        ours = sundial_chart.data["planets"][our_name]["longitude"]
        theirs = fl_chart.get(fl_id).lon
        delta = abs((ours - theirs + 180) % 360 - 180)
        assert delta < 0.01, f"{our_name}: sundial={ours} flatlib={theirs} Δ={delta}"
