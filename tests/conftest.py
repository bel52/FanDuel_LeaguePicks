"""Suite-wide: the tests are offline by design (README). The nflverse whole-pool
grader runs automatically inside build and capture, so stub its loader to behave
like a network outage — the grader must degrade to one printed line, which is
itself worth asserting. Tests that exercise the grader inject their own frames."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def _nflverse_offline(monkeypatch):
    def _offline(kind, season):
        raise ConnectionError("offline test suite")
    monkeypatch.setattr("dfs.actuals._NflverseLoader._load", staticmethod(_offline))
