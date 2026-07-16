"""Enforce 1-1 pandas API correspondence.

Fails if pandas grows public API we haven't implemented or explicitly
declared out of scope in scripts/api_coverage.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pandas as pd  # noqa: E402
from api_coverage import NOT_PLANNED, public  # noqa: E402

import jaxframe as jf  # noqa: E402


def test_dataframe_coverage_100pct():
    target = public(pd.DataFrame) - set(NOT_PLANNED["DataFrame"])
    missing = sorted(target - public(jf.DataFrame))
    assert not missing, f"DataFrame missing pandas API members: {missing}"


def test_series_coverage_100pct():
    target = public(pd.Series) - set(NOT_PLANNED["Series"])
    missing = sorted(target - public(jf.Series))
    assert not missing, f"Series missing pandas API members: {missing}"


def test_not_planned_entries_are_real_pandas_members():
    """Out-of-scope declarations must reference actual pandas API (catch typos)."""
    for cls_name, cls in [("DataFrame", pd.DataFrame), ("Series", pd.Series)]:
        stale = sorted(set(NOT_PLANNED[cls_name]) - public(cls))
        assert not stale, f"NOT_PLANNED[{cls_name}] has stale entries: {stale}"
