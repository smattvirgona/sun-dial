from __future__ import annotations

import pathlib

from sundial.corpus import RetrievalQuery
from sundial.corpus.retrieve import InMemoryRetriever


def test_retrieves_by_factor_tag() -> None:
    retriever = InMemoryRetriever.from_directory(pathlib.Path("content/sources"))
    q = RetrievalQuery(
        placement_tags=["saturn"],
        topic=None,
        systems=["western"],
        rings=[1, 2],
        k=5,
    )
    results = retriever.query(q)
    assert results, "expected at least one Saturn-tagged chunk"
    assert all("saturn" in c.factors for c in results)


def test_filters_out_ring_three_by_default() -> None:
    retriever = InMemoryRetriever.from_directory(pathlib.Path("content/sources"))
    # All seed chunks are ring 1; with rings=[3] we should get nothing back.
    q = RetrievalQuery(
        placement_tags=["saturn", "jupiter", "moon"],
        systems=["western"],
        rings=[3],
        k=10,
    )
    assert retriever.query(q) == []


def test_topic_boost() -> None:
    retriever = InMemoryRetriever.from_directory(pathlib.Path("content/sources"))
    # Both 'jupiter' and 'sun' chunks would match by tag; topic 'fortune'
    # should rank Jupiter higher.
    q = RetrievalQuery(
        placement_tags=["jupiter", "sun"],
        topic="fortune",
        systems=["western"],
        rings=[1, 2],
        k=2,
    )
    results = retriever.query(q)
    assert results
    assert "fortune" in results[0].topics
