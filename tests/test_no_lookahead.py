# -*- coding: utf-8 -*-
"""防前视："截断不变性"——只用截至 t 的面板重算特征，必须与全样本算出的逐位相同；
并故意构造一个偷看未来的特征，确认这个测试能抓到它。"""
import numpy as np
import pandas as pd
import pytest

from aqr import config as C, features as FE, models as M, panel

PROBES = [pd.Timestamp(x) for x in ("2026-02-03 05:00", "2026-05-08 22:00", "2026-07-16 09:00", "2026-09-20 23:00")]


def _features_at(sym, p, t, extra=None):
    d = FE.hourly(sym, p)
    cols = FE.ALL + (["leak"] if extra else [])
    if extra:
        d["leak"] = extra(p).reindex(d.index)
    return d.loc[t, cols].astype(float)


def _max_diff(sym, t, extra=None):
    p = panel.minute_panel(sym)
    a = _features_at(sym, p, t, extra).to_numpy()
    b = _features_at(sym, p.loc[:t], t, extra).to_numpy()
    if (np.isnan(a) != np.isnan(b)).any():  # 一边有值一边缺：截断改变了特征
        return np.inf
    return np.nanmax(np.abs(a - b))


@pytest.mark.needs_data
@pytest.mark.parametrize("sym", C.SYMBOLS)
@pytest.mark.parametrize("t", PROBES)
def test_truncation_invariance(sym, t):
    assert _max_diff(sym, t) < 1e-9


@pytest.mark.needs_data
def test_leak_is_detected():
    leak = lambda p: p["basis"].shift(-60)  # 偷看 1 小时后的基差
    with pytest.raises(AssertionError):
        assert _max_diff("XAUUSDT", PROBES[0], leak) < 1e-9


@pytest.mark.needs_data
@pytest.mark.parametrize("sym", C.SYMBOLS)
def test_split_rules(sym):
    d = FE.samples(sym)
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
    assert (tr.index + C.H * FE.HOUR <= C.TEST_START).all() and (te.index >= C.TEST_START).all()
    assert d.index.min() >= C.SAMPLE_START and (d.index.minute == 0).all()
    assert d["basis"].notna().all() and d["basis_fut"].notna().all()
    for fit, val in M.folds(d):  # 交叉验证的每一折：训练目标窗口不越过验证月起点
        assert (fit.index + C.H * FE.HOUR <= val.index.min()).all()


@pytest.mark.needs_data
def test_baseline_mean_window_excludes_t_minus_24h():
    p = panel.minute_panel("PAXGUSDT")
    t = pd.Timestamp("2026-04-01 12:00")
    d = FE.hourly("PAXGUSDT", p)
    w = p.loc[(p.index > t - pd.Timedelta(hours=24)) & (p.index <= t), "basis"]
    assert len(w) == 1440
    assert np.isclose(d.loc[t, "mean_24h"], w.mean())
