"""One-time Swiss Ephemeris setup.

Uses the built-in Moshier ephemeris (sub-arcsecond accuracy without external
data files). For sub-millisecond JPL DE431 precision, drop the .se1 files into
`./ephe/` and they will be picked up automatically.
"""
from __future__ import annotations

import os
import threading

import swisseph as swe

_EPHE_DIR = os.environ.get("SUNDIAL_EPHE_DIR", "ephe")
_lock = threading.Lock()


def ensure() -> None:
    """Pin the ephemeris path. Called at the top of every compute().

    Re-set on every call (not once): the path is process-global state in
    libswe, and other libraries sharing the process (e.g. flatlib in the
    cross-verification tests) overwrite it with their own bundled files,
    which would silently change our results mid-run.
    """
    with _lock:
        swe.set_ephe_path(_EPHE_DIR)


SIGNS = (
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
)


def sign_of(longitude: float) -> str:
    return SIGNS[int(longitude % 360) // 30]


def degree_in_sign(longitude: float) -> float:
    return longitude % 30
