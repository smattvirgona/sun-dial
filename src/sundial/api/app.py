"""FastAPI service — chart computation and the moment-to-reflect endpoint.

POST /chart    → compute a chart for one system.
POST /reflect  → chart + corpus retrieval + Claude synthesis.

When no ANTHROPIC_API_KEY is configured the /reflect endpoint degrades to a
"dry run": it returns the chart and the retrieved passages without a
synthesized reflection, so the rest of the stack stays demoable offline.
"""
from __future__ import annotations

import os
import pathlib
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sundial.charts import (
    ChineseEngine,
    VedicEngine,
    WesternEngine,
    active_transits,
    current_sky,
    transit_placement_tags,
)
from sundial.corpus import RetrievalQuery
from sundial.corpus.retrieve import InMemoryRetriever
from sundial.journal import JournalStore
from sundial.reflection import synthesize
from sundial.reflection.types import ReflectionRequest
from sundial.shared import BirthData, System

_ENGINES = {
    "western": WesternEngine(),
    "vedic": VedicEngine(),
    "chinese": ChineseEngine(),
}

_SOURCES_DIR = pathlib.Path(__file__).resolve().parents[3] / "content" / "sources"
_STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"


class BirthInput(BaseModel):
    when_utc: datetime
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    place_name: str | None = None

    def to_birth_data(self) -> BirthData:
        when = self.when_utc
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return BirthData(
            when_utc=when, lat=self.lat, lon=self.lon, place_name=self.place_name
        )


class ChartRequest(BaseModel):
    birth: BirthInput
    system: System = "western"


class ReflectRequest(BaseModel):
    birth: BirthInput
    question: str = Field(min_length=1, max_length=500)
    system: System = "western"


def create_app(
    retriever: InMemoryRetriever | None = None,
    journal: JournalStore | None = None,
) -> FastAPI:
    app = FastAPI(title="sun-dial", version="0.1.0")
    app.state.retriever = retriever or InMemoryRetriever.from_directory(_SOURCES_DIR)
    app.state.journal = journal or JournalStore()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chart")
    def chart(req: ChartRequest) -> dict[str, Any]:
        engine = _ENGINES[req.system]
        result = engine.compute(req.birth.to_birth_data())
        return {"chart": result.model_dump(), "chart_hash": result.chart_hash}

    @app.post("/reflect")
    def reflect(req: ReflectRequest) -> dict[str, Any]:
        engine = _ENGINES[req.system]
        birth = req.birth.to_birth_data()
        chart_obj = engine.compute(birth)
        placements = engine.salient_placements(chart_obj)

        # Current sky → active transits → extra retrieval tags. Western only
        # for transit math in v1; the natal chart can still be Vedic/Chinese
        # and the transit tags only fire on the western retrieval row.
        sky = current_sky(birth)
        transits = (
            active_transits(chart_obj, sky) if req.system == "western" else []
        )

        placement_tags = [p.factor for p in placements] + transit_placement_tags(transits)
        query = RetrievalQuery(
            placement_tags=placement_tags,
            systems=[req.system],
            rings=[1, 2],
            k=8,
        )
        snippets = app.state.retriever.query(query)

        payload: dict[str, Any] = {
            "chart_hash": chart_obj.chart_hash,
            "system": req.system,
            "transits": transits,
            "snippets": [
                {"id": c.id, "source_id": c.source_id, "text": c.text}
                for c in snippets
            ],
        }

        if not os.environ.get("ANTHROPIC_API_KEY"):
            payload["reflection"] = None
            payload["note"] = "synthesis unavailable: no ANTHROPIC_API_KEY configured"
            # Still record the moment so the journal shows that the user
            # asked, even if no reflection was synthesized.
            payload["journal_id"] = app.state.journal.save(
                system=req.system,
                chart_hash=chart_obj.chart_hash,
                question=req.question,
                headline=None,
                body=None,
                citations=[],
                transits=transits,
            )
            return payload

        if not snippets:
            raise HTTPException(
                status_code=422,
                detail="no corpus passages matched this chart for the selected system",
            )

        reflection = synthesize(
            ReflectionRequest(
                question=req.question,
                chart=chart_obj,
                systems_in_play=[req.system],
                snippets=snippets,
                transits=transits,
            )
        )
        payload["reflection"] = reflection.model_dump()
        payload["journal_id"] = app.state.journal.save(
            system=req.system,
            chart_hash=chart_obj.chart_hash,
            question=req.question,
            headline=reflection.headline,
            body=reflection.body,
            citations=[c.model_dump() for c in reflection.citations],
            transits=transits,
        )
        return payload

    @app.get("/journal")
    def journal(chart_hash: str | None = None, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        entries = app.state.journal.recent(chart_hash=chart_hash, limit=limit)
        return {"entries": entries}

    return app


app = create_app()
