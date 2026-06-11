"""Golden-file regression tests for all three chart engines.

Fixtures live in `content/fixtures/charts/`. Regenerate them deliberately with
`scripts/regen_fixtures.py` after a verified change.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from sundial.charts import ChineseEngine, VedicEngine, WesternEngine
from sundial.shared import BirthData

FIXTURES = pathlib.Path("content/fixtures/charts")


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _approx_equal(a, b, tol=1e-3) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_approx_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_approx_equal(x, y, tol) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("name", ["einstein", "y2k_greenwich", "lichun_1984"])
def test_western_golden(name: str) -> None:
    fx = _load(name)
    birth = BirthData(**fx["birth"])
    chart = WesternEngine().compute(birth)
    assert _approx_equal(chart.model_dump()["data"], fx["western"]["data"])


@pytest.mark.parametrize("name", ["einstein", "y2k_greenwich", "lichun_1984"])
def test_vedic_golden(name: str) -> None:
    fx = _load(name)
    birth = BirthData(**fx["birth"])
    chart = VedicEngine().compute(birth)
    assert _approx_equal(chart.model_dump()["data"], fx["vedic"]["data"])


@pytest.mark.parametrize("name", ["einstein", "y2k_greenwich", "lichun_1984"])
def test_chinese_golden(name: str) -> None:
    fx = _load(name)
    birth = BirthData(**fx["birth"])
    chart = ChineseEngine().compute(birth)
    # _approx_equal compares non-floats exactly, so pillar stems/branches stay
    # an exact contract; only the sun-longitude float gets tolerance.
    assert _approx_equal(chart.model_dump()["data"], fx["chinese"]["data"])
