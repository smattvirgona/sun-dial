"""Regenerate golden chart fixtures.

Run deliberately after a verified change:

    python scripts/regen_fixtures.py
"""
from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone

from sundial.charts import ChineseEngine, VedicEngine, WesternEngine
from sundial.shared import BirthData

BIRTHS = {
    "einstein": BirthData(
        when_utc=datetime(1879, 3, 14, 10, 50, tzinfo=timezone.utc),
        lat=48.4011, lon=9.9876, place_name="Ulm",
    ),
    "y2k_greenwich": BirthData(
        when_utc=datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc),
        lat=51.4779, lon=0.0, place_name="Greenwich",
    ),
    "lichun_1984": BirthData(
        when_utc=datetime(1984, 2, 5, 0, 0, tzinfo=timezone.utc),
        lat=39.9042, lon=116.4074, place_name="Beijing",
    ),
}


def main() -> None:
    out_dir = pathlib.Path("content/fixtures/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, birth in BIRTHS.items():
        payload = {
            "birth": birth.model_dump(mode="json"),
            "western": WesternEngine().compute(birth).model_dump(),
            "vedic": VedicEngine().compute(birth).model_dump(),
            "chinese": ChineseEngine().compute(birth).model_dump(),
        }
        (out_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
        print(f"wrote {name}.json")


if __name__ == "__main__":
    main()
