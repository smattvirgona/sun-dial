from datetime import datetime, timezone

import pytest

from sundial.shared import BirthData


@pytest.fixture
def einstein() -> BirthData:
    # Albert Einstein — Ulm, Germany. 1879-03-14 11:30 LMT.
    # Ulm lon ≈ 9.99°E → LMT offset ≈ +0:40, so UTC ≈ 10:50.
    return BirthData(
        when_utc=datetime(1879, 3, 14, 10, 50, tzinfo=timezone.utc),
        lat=48.4011,
        lon=9.9876,
        place_name="Ulm, Germany",
    )


@pytest.fixture
def y2k_greenwich() -> BirthData:
    # 2000-01-01 00:00 UTC at Greenwich — clean reference.
    return BirthData(
        when_utc=datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc),
        lat=51.4779,
        lon=0.0,
        place_name="Greenwich, UK",
    )


@pytest.fixture
def lichun_1984() -> BirthData:
    # 1984-02-05 00:00 UTC, Beijing. Just after Lichun 1984. Should be Jia-Zi year.
    return BirthData(
        when_utc=datetime(1984, 2, 5, 0, 0, tzinfo=timezone.utc),
        lat=39.9042,
        lon=116.4074,
        place_name="Beijing, China",
    )
