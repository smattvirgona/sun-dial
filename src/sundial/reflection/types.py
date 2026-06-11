from __future__ import annotations

from pydantic import BaseModel

from sundial.charts.engine import Chart
from sundial.corpus import CorpusChunk
from sundial.shared import System


class ReflectionRequest(BaseModel):
    question: str
    chart: Chart
    systems_in_play: list[System]
    snippets: list[CorpusChunk]
    transits: list[dict] = []  # active transits at the moment of the request


class Citation(BaseModel):
    chunk_id: str
    quote: str


class Reflection(BaseModel):
    headline: str  # pithy, Co-Star surface
    body: str  # lyrical, CHANI body
    citations: list[Citation]
    chart_hash: str
