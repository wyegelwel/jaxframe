"""Pandas equivalence tests for the API-parity expansion (Series methods,
DataFrame named ops, missing-data ops, selection, combining, reshaping).

Style mirrors test_pandas_mirror.py: one lambda run against both libraries.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import jaxframe as jf

# ============================
# Helpers
# ============================

S_DATA = [3.0, 1.0, np.nan, 2.0, 2.0]
S_CLEAN = [3.0, 1.0, 4.0, 2.0, 2.0]
DF_DATA = {"a": [3.0, 1.0, np.nan, 2.0, 2.0], "b": [10.0, 20.0, 30.0, np.nan, 50.0]}
DF_CLEAN = {"a": [3.0, 1.0, 4.0, 2.0, 2.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}


def _values(x):
    if hasattr(x, "values"):
        return np.asarray(x.values, dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def run_series_equiv(data, op, rtol=1e-4):
    ps = pd.Series(data, name="x")
    js = jf.Series(data, name="x")
    p, j = op(ps), op(js)
    pv, jv = _values(p), _values(j)
    assert pv.shape == jv.shape, f"shape {pv.shape} != {jv.shape}"
    assert_allclose(jv, pv, rtol=rtol, equal_nan=True)


def run_df_equiv(data, op, rtol=1e-4):
    pdf = pd.DataFrame(data)
    jdf = jf.DataFrame(data)
    p, j = op(pdf), op(jdf)
    pv, jv = _values(p), _values(j)
    assert pv.shape == jv.shape, f"shape {pv.shape} != {jv.shape}"
    assert_allclose(jv, pv, rtol=rtol, equal_nan=True)


# ============================
# Series: reductions
# ============================

SERIES_REDUCTIONS = [
    ("min", lambda s: s.min()),
    ("max", lambda s: s.max()),
    ("prod", lambda s: s.prod()),
    ("sum", lambda s: s.sum()),
    ("mean", lambda s: s.mean()),
    ("std", lambda s: s.std()),
    ("var", lambda s: s.var()),
    ("sem", lambda s: s.sem()),
    ("median", lambda s: s.median()),
    ("quantile", lambda s: s.quantile(0.3)),
    ("count", lambda s: s.count()),
    ("nunique", lambda s: s.nunique()),
    ("skew", lambda s: s.skew()),
    ("kurt", lambda s: s.kurt()),
    ("argmax", lambda s: s.argmax()),
    ("argmin", lambda s: s.argmin()),
    ("idxmax", lambda s: s.idxmax()),
    ("idxmin", lambda s: s.idxmin()),
    ("autocorr", lambda s: s.autocorr()),
]


@pytest.mark.parametrize("name,op", SERIES_REDUCTIONS, ids=[t[0] for t in SERIES_REDUCTIONS])
def test_series_reduction(name, op):
    run_series_equiv(S_DATA, op)


# ============================
# Series: elementwise / cumulative / fills
# ============================

SERIES_ELEMENTWISE = [
    ("cumsum", lambda s: s.cumsum()),
    ("cumprod", lambda s: s.cumprod()),
    ("cummax", lambda s: s.cummax()),
    ("cummin", lambda s: s.cummin()),
    ("ffill", lambda s: s.ffill()),
    ("bfill", lambda s: s.bfill()),
    ("interpolate", lambda s: s.interpolate()),
    ("fillna", lambda s: s.fillna(0.0)),
    ("clip", lambda s: s.clip(1.5, 2.5)),
    ("round", lambda s: s.round()),
    ("isna", lambda s: s.isna()),
    ("notna", lambda s: s.notna()),
    ("isin", lambda s: s.isin([1.0, 2.0])),
    ("where", lambda s: s.where(s > 1.5, 0.0)),
    ("mask", lambda s: s.mask(s > 1.5, 0.0)),
    ("add", lambda s: s.add(2)),
    ("rsub", lambda s: s.rsub(2)),
    ("mul", lambda s: s.mul(3)),
    ("rdiv", lambda s: s.rdiv(1.0)),
    ("pow", lambda s: s.pow(2)),
    ("eq", lambda s: s.eq(2.0)),
    ("ne", lambda s: s.ne(2.0)),
    ("le", lambda s: s.le(2.0)),
    ("gt", lambda s: s.gt(2.0)),
    ("abs", lambda s: (-s).abs()),
    ("diff", lambda s: s.diff()),
    ("pct_change", lambda s: s.pct_change()),
    ("rank_average", lambda s: s.rank()),
    ("rank_min", lambda s: s.rank(method="min")),
    ("rank_max", lambda s: s.rank(method="max")),
    ("rank_dense", lambda s: s.rank(method="dense")),
    ("rank_first", lambda s: s.rank(method="first")),
    ("rank_desc", lambda s: s.rank(ascending=False)),
]


@pytest.mark.parametrize("name,op", SERIES_ELEMENTWISE, ids=[t[0] for t in SERIES_ELEMENTWISE])
def test_series_elementwise(name, op):
    run_series_equiv(S_DATA, op)


# ============================
# Series: selection / sorting / uniqueness
# ============================

SERIES_SELECTION = [
    ("head", lambda s: s.head(2)),
    ("tail", lambda s: s.tail(2)),
    ("tail_negative", lambda s: s.tail(-2)),
    ("take", lambda s: s.take([0, 2])),
    ("sort_values", lambda s: s.sort_values()),
    ("sort_values_desc", lambda s: s.sort_values(ascending=False)),
    ("sort_index", lambda s: s.sort_index(ascending=False)),
    ("nlargest", lambda s: s.nlargest(2)),
    ("nsmallest", lambda s: s.nsmallest(2)),
    ("drop_duplicates", lambda s: s.drop_duplicates()),
    ("duplicated", lambda s: s.duplicated()),
    ("duplicated_last", lambda s: s.duplicated(keep="last")),
    ("duplicated_false", lambda s: s.duplicated(keep=False)),
    ("unique", lambda s: s.unique()),
    ("dropna", lambda s: s.dropna()),
    ("reindex", lambda s: s.reindex([4, 0, 9])),
    ("mode", lambda s: s.mode()),
    ("describe", lambda s: s.describe()),
    ("repeat", lambda s: s.repeat(2)),
    ("factorize_codes", lambda s: s.factorize()[0]),
]


@pytest.mark.parametrize("name,op", SERIES_SELECTION, ids=[t[0] for t in SERIES_SELECTION])
def test_series_selection(name, op):
    run_series_equiv(S_DATA, op)


# ============================
# Series: windows / grouping / combining
# ============================

SERIES_WINDOWS = [
    ("rolling_mean", lambda s: s.rolling(2).mean()),
    ("rolling_sum", lambda s: s.rolling(2).sum()),
    ("expanding_sum", lambda s: s.expanding().sum()),
    ("expanding_mean", lambda s: s.expanding().mean()),
    ("ewm_mean", lambda s: s.ewm(alpha=0.5).mean()),
    ("apply", lambda s: s.apply(lambda x: x * 2)),
    ("agg_mean", lambda s: s.agg("mean")),
    ("transform", lambda s: s.transform(lambda x: x - x.mean())),
]


@pytest.mark.parametrize("name,op", SERIES_WINDOWS, ids=[t[0] for t in SERIES_WINDOWS])
def test_series_windows(name, op):
    run_series_equiv(S_DATA, op)


def test_series_ewm_with_nan():
    run_series_equiv(S_DATA, lambda s: s.ewm(alpha=0.5).mean())


def test_series_groupby_sum_skips_nan():
    keys = [0, 1, 0, 1, 0]
    ps = pd.Series(S_DATA).groupby(pd.Series(keys)).sum()
    js = jf.Series(S_DATA).groupby(jf.Series(keys)).sum()
    assert_allclose(_values(js), _values(ps), rtol=1e-5)


def test_series_combine_first():
    other_p = pd.Series([9.0] * 5)
    other_j = jf.Series([9.0] * 5)
    ps = pd.Series(S_DATA).combine_first(other_p)
    js = jf.Series(S_DATA).combine_first(other_j)
    assert_allclose(_values(js), _values(ps))


def test_series_corr_cov_dot():
    ps = pd.Series(S_DATA)
    js = jf.Series(S_DATA)
    assert_allclose(float(js.corr(js * 2)), float(ps.corr(ps * 2)), rtol=1e-4)
    assert_allclose(float(js.cov(js * 2)), float(ps.cov(ps * 2)), rtol=1e-4)
    p2, j2 = pd.Series(S_CLEAN), jf.Series(S_CLEAN)
    assert_allclose(float(j2.dot(j2)), float(p2.dot(p2)), rtol=1e-5)


def test_series_equals_and_item():
    assert jf.Series([1.0, 2.0]).equals(jf.Series([1.0, 2.0]))
    assert not jf.Series([1.0, 2.0]).equals(jf.Series([1.0, 3.0]))
    assert jf.Series([7.0]).item() == 7.0


def test_series_indexers():
    ps = pd.Series(S_CLEAN, name="x")
    js = jf.Series(S_CLEAN, name="x")
    assert float(js.iloc[1]) == float(ps.iloc[1])
    assert float(js.loc[3]) == float(ps.loc[3])
    assert float(js.at[2]) == float(ps.at[2])
    assert float(js.iat[4]) == float(ps.iat[4])
    assert_allclose(_values(js.iloc[1:3]), _values(ps.iloc[1:3]))
    assert_allclose(_values(js[js > 1.5]), _values(ps[ps > 1.5]))


def test_series_properties():
    ps = pd.Series(S_DATA, name="x")
    js = jf.Series(S_DATA, name="x")
    assert js.shape == ps.shape
    assert js.size == ps.size
    assert js.ndim == ps.ndim
    assert js.empty == ps.empty
    assert js.hasnans == ps.hasnans
    assert js.is_unique == ps.is_unique
    assert js.is_monotonic_increasing == ps.is_monotonic_increasing
    assert js.to_dict() == {k: pytest.approx(v, nan_ok=True) for k, v in ps.to_dict().items()}
    assert js.tolist() == pytest.approx(ps.tolist(), nan_ok=True)


def test_series_case_when():
    ps = pd.Series(S_CLEAN)
    js = jf.Series(S_CLEAN)
    p = ps.case_when([(ps > 2.5, 100.0), (ps > 1.5, 50.0)])
    j = js.case_when([(js > 2.5, 100.0), (js > 1.5, 50.0)])
    assert_allclose(_values(j), _values(p))


# ============================
# DataFrame: named ops / comparisons
# ============================

DF_NAMED_OPS = [
    ("add", lambda d: d.add(2)),
    ("radd", lambda d: d.radd(2)),
    ("sub", lambda d: d.sub(2)),
    ("rsub", lambda d: d.rsub(2)),
    ("mul", lambda d: d.mul(3)),
    ("div", lambda d: d.div(2)),
    ("rdiv", lambda d: d.rdiv(1)),
    ("floordiv", lambda d: d.floordiv(2)),
    ("mod", lambda d: d.mod(3)),
    ("pow", lambda d: d.pow(2)),
    ("mul_df", lambda d: d.mul(d)),
    ("eq", lambda d: d.eq(2.0)),
    ("ne", lambda d: d.ne(2.0)),
    ("lt", lambda d: d.lt(2.0)),
    ("le", lambda d: d.le(2.0)),
    ("gt", lambda d: d.gt(2.0)),
    ("ge", lambda d: d.ge(2.0)),
]


@pytest.mark.parametrize("name,op", DF_NAMED_OPS, ids=[t[0] for t in DF_NAMED_OPS])
def test_df_named_op(name, op):
    run_df_equiv(DF_CLEAN, op)


# ============================
# DataFrame: cumulative / missing data
# ============================

DF_NAN_OPS = [
    ("cummax", lambda d: d.cummax()),
    ("cummin", lambda d: d.cummin()),
    ("ffill", lambda d: d.ffill()),
    ("bfill", lambda d: d.bfill()),
    ("dropna", lambda d: d.dropna()),
    ("dropna_all", lambda d: d.dropna(how="all")),
    ("take", lambda d: d.take([0, 2])),
    ("truncate", lambda d: d.truncate(1, 3)),
    ("filter_like", lambda d: d.filter(like="a")),
    ("reindex_rows", lambda d: d.reindex([4, 0, 9])),
    ("reindex_cols", lambda d: d.reindex(columns=["b", "z"])),
    ("agg_mean", lambda d: d.agg("mean")),
    ("transform", lambda d: d.transform(lambda x: x * 2)),
    ("map", lambda d: d.map(lambda x: x + 1)),
    ("replace", lambda d: d.replace(2.0, 99.0)),
    ("corrwith_self2x", lambda d: d.corrwith(d * 2)),
    ("query", lambda d: d.query("a > 1.5")),
    ("eval_assign", lambda d: d.eval("c = a + b")),
    ("xs", lambda d: d.xs(2)),
    ("squeeze", lambda d: d[["a"]].squeeze()),
    ("value_counts", lambda d: d[["a"]].value_counts()),
    ("stack", lambda d: d.stack()),
    ("unstack", lambda d: d.unstack()),
    ("add_prefix", lambda d: d.add_prefix("p_")),
    ("add_suffix", lambda d: d.add_suffix("_s")),
]


@pytest.mark.parametrize("name,op", DF_NAN_OPS, ids=[t[0] for t in DF_NAN_OPS])
def test_df_nan_op(name, op):
    run_df_equiv(DF_DATA, op)


# ============================
# DataFrame: indexing / mutation / combining
# ============================


def test_df_loc_at_iat():
    pdf, jdf = pd.DataFrame(DF_CLEAN), jf.DataFrame(DF_CLEAN)
    assert_allclose(_values(jdf.loc[2]), _values(pdf.loc[2]))
    assert float(jdf.loc[2, "b"]) == float(pdf.loc[2, "b"])
    assert_allclose(_values(jdf.loc[jdf["a"] > 1.5]), _values(pdf.loc[pdf["a"] > 1.5]))
    assert float(jdf.at[1, "a"]) == float(pdf.at[1, "a"])
    assert float(jdf.iat[1, 0]) == float(pdf.iat[1, 0])


def test_df_setitem_insert_pop():
    pdf, jdf = pd.DataFrame(DF_CLEAN), jf.DataFrame(DF_CLEAN)
    pdf["c"] = pdf["a"] * 2
    jdf["c"] = jdf["a"] * 2
    pdf.insert(1, "z", pdf["b"])
    jdf.insert(1, "z", jdf["b"])
    assert list(pdf.columns) == list(jdf.columns)
    pp, jp = pdf.pop("z"), jdf.pop("z")
    assert_allclose(_values(jp), _values(pp))
    assert_allclose(_values(jdf), _values(pdf))


def test_df_update():
    pdf, jdf = pd.DataFrame(DF_DATA), jf.DataFrame(DF_DATA)
    patch_p = pd.DataFrame({"a": [100.0, np.nan, 100.0, np.nan, np.nan]})
    patch_j = jf.DataFrame({"a": [100.0, np.nan, 100.0, np.nan, np.nan]})
    pdf.update(patch_p)
    jdf.update(patch_j)
    assert_allclose(_values(jdf), _values(pdf), rtol=1e-5)


def test_df_join():
    lp = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
    rp = pd.DataFrame({"w": [10.0, 30.0]}, index=[0, 2])
    lj = jf.DataFrame({"v": [1.0, 2.0, 3.0]})
    rj = jf.DataFrame({"w": [10.0, 30.0]}, index=[0, 2])
    assert_allclose(_values(lj.join(rj)), _values(lp.join(rp)), rtol=1e-5)


def test_df_merge_left_nan_fill():
    a_p = pd.DataFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    b_p = pd.DataFrame({"k": [2, 3, 4], "w": [20.0, 30.0, 40.0]})
    a_j = jf.DataFrame({"k": [1, 2, 3], "v": [1.0, 2.0, 3.0]})
    b_j = jf.DataFrame({"k": [2, 3, 4], "w": [20.0, 30.0, 40.0]})
    for how in ("left", "right", "outer", "inner"):
        mp = a_p.merge(b_p, on="k", how=how)
        mj = a_j.merge(b_j, on="k", how=how)
        assert_allclose(_values(mj), _values(mp), rtol=1e-5, err_msg=how)


def test_df_combine_first_and_equals():
    pdf, jdf = pd.DataFrame(DF_DATA), jf.DataFrame(DF_DATA)
    fill_p = pd.DataFrame({"a": [9.0] * 5, "b": [9.0] * 5})
    fill_j = jf.DataFrame({"a": [9.0] * 5, "b": [9.0] * 5})
    assert_allclose(
        _values(jdf.combine_first(fill_j)), _values(pdf.combine_first(fill_p)), rtol=1e-5
    )
    assert jdf.equals(jdf.copy())
    assert not jdf.equals(fill_j)


def test_df_iterrows_itertuples_items():
    pdf, jdf = pd.DataFrame(DF_CLEAN), jf.DataFrame(DF_CLEAN)
    p_rows = [(i, r.tolist()) for i, r in pdf.iterrows()]
    j_rows = [(i, np.asarray(r.values, dtype=np.float64).tolist()) for i, r in jdf.iterrows()]
    assert p_rows == j_rows
    for tp, tj in zip(pdf.itertuples(), jdf.itertuples()):
        assert tp.Index == tj.Index
        assert tp.a == pytest.approx(float(tj.a))
    for (cp, sp), (cj, sj) in zip(pdf.items(), jdf.items()):
        assert cp == cj
        assert_allclose(_values(sj), _values(sp))


def test_df_from_dict_from_records_to_dict():
    assert_allclose(
        _values(jf.DataFrame.from_dict({"x": [1, 2]})),
        _values(pd.DataFrame.from_dict({"x": [1, 2]})),
    )
    assert_allclose(
        _values(jf.DataFrame.from_records([{"x": 1}, {"x": 2}])),
        _values(pd.DataFrame.from_records([{"x": 1}, {"x": 2}])),
    )
    pdf, jdf = pd.DataFrame(DF_CLEAN), jf.DataFrame(DF_CLEAN)
    assert pdf.to_dict("list") == jdf.to_dict("list")
    assert pdf.to_dict("records") == jdf.to_dict("records")


def test_df_pivot():
    raw = {"k": ["x", "y", "x"], "i": [1, 1, 2], "v": [1.0, 2.0, 3.0]}
    p = pd.DataFrame(raw).pivot(columns="k", index="i", values="v")
    j = jf.DataFrame(raw).pivot(columns="k", index="i", values="v")
    assert_allclose(_values(j), _values(p), rtol=1e-5)


def test_df_explode():
    p = pd.DataFrame({"l": [[1, 2], [3]], "v": [10.0, 20.0]}).explode("l")
    j = jf.DataFrame({"l": np.asarray([[1, 2], [3]], dtype=object), "v": [10.0, 20.0]}).explode("l")
    assert_allclose(_values(j["v"]), _values(p["v"]))
    assert [int(x) for x in j["l"].values] == [int(x) for x in p["l"].values]


def test_df_sample_shape_and_determinism():
    jdf = jf.DataFrame(DF_CLEAN)
    s1 = jdf.sample(3, random_state=0)
    s2 = jdf.sample(3, random_state=0)
    assert s1.shape == (3, 2)
    assert_allclose(_values(s1), _values(s2))
