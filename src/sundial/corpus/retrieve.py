"""Retrieval interface. v1 implementation is an in-memory store seeded from
`content/sources/*.json`; production swaps in pgvector behind the same API.
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable

from .types import CorpusChunk, RetrievalQuery


class InMemoryRetriever:
    """Tag-filtered linear scan. Replace with pgvector + embeddings in v1.5."""

    def __init__(self, chunks: Iterable[CorpusChunk]) -> None:
        self._chunks = list(chunks)

    @classmethod
    def from_directory(cls, path: pathlib.Path) -> "InMemoryRetriever":
        chunks: list[CorpusChunk] = []
        for f in path.glob("*.json"):
            for raw in json.loads(f.read_text()):
                chunks.append(CorpusChunk(**raw))
        return cls(chunks)

    def query(self, q: RetrievalQuery) -> list[CorpusChunk]:
        wanted = set(q.placement_tags)
        results: list[tuple[int, CorpusChunk]] = []
        for c in self._chunks:
            if c.system not in q.systems:
                continue
            if c.license_ring not in q.rings:
                continue
            score = len(wanted.intersection(c.factors))
            if q.topic and q.topic in c.topics:
                score += 1
            if score == 0:
                continue
            results.append((score, c))
        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results[: q.k]]
