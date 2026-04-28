"""Corpus pipeline — ingest, segment, embed, retrieve.

v1 implementation is intentionally thin; the public types are stable so the
ingestion + retrieval slices can be filled in without touching `reflection`.
"""
from .types import CorpusChunk, CorpusSource, LicenseRing, RetrievalQuery

__all__ = ["CorpusChunk", "CorpusSource", "LicenseRing", "RetrievalQuery"]
