Offline test suite — no API keys or network required (`conftest.py` makes nflverse
look offline; tests that exercise the nflverse grader inject their own frames).
Run: `python3 -m pytest tests/ -q`. One swap-accept test is time-dependent and fails
when run long after its fixture date; it is a known, unrelated failure.
