#!/usr/bin/env python3
"""Report pandas API coverage for jaxframe.

Usage: uv run python scripts/api_coverage.py [--missing]

Compares the public API surface of pandas DataFrame/Series against
jaxframe's, excluding members we have explicitly declared out of scope
(NOT_PLANNED). The test suite asserts 100% coverage of everything else,
so "1-1 correspondence" is enforced by construction, not by recall.
"""

import sys

import pandas as pd

import jaxframe as jf

# Members deliberately out of scope, with reasons.
NOT_PLANNED = {
    "DataFrame": {
        # pandas-internal / deprecated / exotic storage
        "sparse": "sparse extension arrays are out of scope for JAX backing",
        "style": "Styler (Jinja HTML styling) is out of scope",
        "to_iceberg": "requires pyiceberg; niche I/O",
        "from_arrow": "arrow interchange out of scope (use from_pandas)",
        # Timezone / period semantics require datetime index machinery
        "to_period": "PeriodIndex out of scope",
        "to_timestamp": "PeriodIndex out of scope",
        "tz_convert": "tz-aware DatetimeIndex out of scope",
        "tz_localize": "tz-aware DatetimeIndex out of scope",
        "asfreq": "DatetimeIndex frequency conversion out of scope",
        "at_time": "DatetimeIndex selection out of scope",
        "between_time": "DatetimeIndex selection out of scope",
        # MultiIndex-only
        "droplevel": "MultiIndex out of scope",
        "reorder_levels": "MultiIndex out of scope",
        "swaplevel": "MultiIndex out of scope",
    },
    "Series": {
        "sparse": "sparse extension arrays are out of scope",
        "struct": "arrow struct accessor out of scope",
        "list": "arrow list accessor out of scope",
        "cat": "categorical accessor out of scope",
        "from_arrow": "arrow interchange out of scope",
        "to_period": "PeriodIndex out of scope",
        "to_timestamp": "PeriodIndex out of scope",
        "tz_convert": "tz-aware DatetimeIndex out of scope",
        "tz_localize": "tz-aware DatetimeIndex out of scope",
        "asfreq": "DatetimeIndex frequency conversion out of scope",
        "at_time": "DatetimeIndex selection out of scope",
        "between_time": "DatetimeIndex selection out of scope",
        "droplevel": "MultiIndex out of scope",
        "reorder_levels": "MultiIndex out of scope",
        "swaplevel": "MultiIndex out of scope",
    },
}


def public(cls):
    return {m for m in dir(cls) if not m.startswith("_")}


def coverage(verbose_missing=False):
    ok = True
    for name, pcls, jcls in [
        ("DataFrame", pd.DataFrame, jf.DataFrame),
        ("Series", pd.Series, jf.Series),
    ]:
        target = public(pcls) - set(NOT_PLANNED[name])
        have = public(jcls)
        missing = sorted(target - have)
        pct = 100.0 * (len(target) - len(missing)) / len(target)
        print(
            f"{name}: {len(target) - len(missing)}/{len(target)} "
            f"({pct:.1f}%) — {len(missing)} missing, "
            f"{len(NOT_PLANNED[name])} declared out of scope"
        )
        if missing:
            ok = False
            if verbose_missing:
                for m in missing:
                    print(f"  MISSING {name}.{m}")
    return ok


if __name__ == "__main__":
    ok = coverage(verbose_missing="--missing" in sys.argv)
    sys.exit(0 if ok else 1)
