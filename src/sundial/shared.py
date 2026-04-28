"""Shared types for chart engines.

`BirthData` is in UTC throughout. Callers convert local civil time → UTC at the
edge using the user's stated timezone; the engines never see local time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

System = Literal["western", "vedic", "chinese"]
HouseSystem = Literal["whole_sign", "placidus"]
Ayanamsa = Literal["lahiri", "raman", "kp"]


class BirthData(BaseModel):
    when_utc: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    place_name: str | None = None

    @field_validator("when_utc")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("when_utc must be timezone-aware (UTC)")
        return v.astimezone(timezone.utc)


class WesternOptions(BaseModel):
    houses: HouseSystem = "whole_sign"


class VedicOptions(BaseModel):
    ayanamsa: Ayanamsa = "lahiri"


class ChineseOptions(BaseModel):
    use_true_solar_time: bool = True


class Placement(BaseModel):
    """A factor to query the corpus with — e.g. ('western','sun','aries')."""

    system: System
    factor: str
    modifier: str | None = None

    def tag(self) -> str:
        if self.modifier:
            return f"{self.system}:{self.factor}:{self.modifier}"
        return f"{self.system}:{self.factor}"
