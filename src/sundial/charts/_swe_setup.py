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
_initialised = False


def ensure() -> None:
    global _initialised
    with _lock:
        if _initialised:
            return
        swe.set_ephe_path(_EPHE_DIR)
        _initialised = True


SIGNS = (
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
)


def sign_of(longitude: float) -> str:
    return SIGNS[int(longitude % 360) // 30]


def degree_in_sign(longitude: float) -> float:
    return longitude % 30
