"""Corpus types. License ring is enforced at ingest, not at query time."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sundial.shared import System

# Ring 1: public domain. Ring 2: permissive / CC. Ring 3: copyrighted (skip
# unless explicitly licensed). Demo runs on Rings 1 + 2 only.
LicenseRing = Literal[1, 2, 3]


class CorpusSource(BaseModel):
    id: str
    title: str
    author: str | None = None
    year: int | None = None
    system: System
    license_ring: LicenseRing
    source_url: str | None = None
    verified_on: str | None = None  # ISO date string


class CorpusChunk(BaseModel):
    id: str
    source_id: str
    system: System
    factors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    text: str
    license_ring: LicenseRing


class RetrievalQuery(BaseModel):
    placement_tags: list[str]
    topic: str | None = None
    systems: list[System]
    rings: list[LicenseRing] = Field(default_factory=lambda: [1, 2])
    k: int = 8
