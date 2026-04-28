"""Chart engine protocol shared by Western, Vedic, and Chinese engines."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from pydantic import BaseModel

from sundial.shared import BirthData, Placement, System


class Chart(BaseModel):
    system: System
    options: dict[str, Any]
    data: dict[str, Any]

    @property
    def chart_hash(self) -> str:
        payload = json.dumps(
            {"system": self.system, "options": self.options, "data": self.data},
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class ChartEngine(Protocol):
    system: System

    def compute(self, birth: BirthData, options: Any | None = None) -> Chart: ...

    def salient_placements(
        self, chart: Chart, topic: str | None = None
    ) -> list[Placement]: ...
