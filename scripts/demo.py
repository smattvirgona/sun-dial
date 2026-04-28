"""CLI demo — print all three charts for a given birth.

Usage:
    python scripts/demo.py 1990-06-15T18:30Z 40.6782 -73.9442
"""
from __future__ import annotations

import json
import sys
from datetime import datetime

from sundial.charts import ChineseEngine, VedicEngine, WesternEngine
from sundial.shared import BirthData


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    iso, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    birth = BirthData(when_utc=when, lat=lat, lon=lon)
    out = {
        "birth": birth.model_dump(mode="json"),
        "western": WesternEngine().compute(birth).model_dump(),
        "vedic": VedicEngine().compute(birth).model_dump(),
        "chinese": ChineseEngine().compute(birth).model_dump(),
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
