# -*- coding: utf-8 -*-
"""资金费公式、时变参数表、已知关键期。"""
import numpy as np
import pandas as pd
import pytest

from aqr import funding, funding_params as FP


def test_formula_dead_zone_and_caps():
    assert np.isclose(funding.formula(3e-4, 1e-4, 4), 0.5e-4)       # 死区内 F = I/2（4h）
    assert np.isclose(funding.formula(9e-4, 1e-4, 4), 2e-4)         # 死区外 F8 = P − 0.05% = 4bp → 4h 2bp
    assert np.isclose(funding.formula(2e-3, 0.0, 8, 5e-4, -5e-4), 5e-4)


def test_params_boundary_semantics():
    # 利息项 01-23 08:00 切换：08:00 结算期仍用旧 I，之后的期用新 I
    assert FP.params_for("XAUUSDT", pd.Timestamp("2026-01-23 08:00")).I == 1e-4
    assert FP.params_for("XAUUSDT", pd.Timestamp("2026-01-23 12:00")).I == 0.0
    assert FP.params_for("XAUUSDT", pd.Timestamp("2026-01-30 16:00")).cap == 5e-4
    assert FP.params_for("XAUUSDT", pd.Timestamp("2026-01-30 20:00")).cap == 5e-3
    assert FP.params_for("PAXGUSDT", pd.Timestamp("2026-09-30 20:00")).I == 1e-4


@pytest.fixture(scope="module")
def rep():
    return {s: funding.reproduce(s) for s in ("XAUUSDT", "PAXGUSDT")}


@pytest.mark.needs_data
def test_known_periods(rep):
    x = rep["XAUUSDT"]
    assert abs(x.loc[pd.Timestamp("2026-01-23 08:00"), "err_bp"]) < 1e-9      # 边界语义
    assert x.loc[pd.Timestamp("2026-01-30 16:00"), "capped"]                    # 8h 期被 ±0.05% 截断
    assert np.isclose(x.loc[pd.Timestamp("2026-02-28 12:00"), "official"], 5e-3)
    assert x.loc[pd.Timestamp("2026-02-28 12:00"), "calc_uncapped"] > 5e-3      # 0.5% 上限生效


@pytest.mark.needs_data
def test_reproduction_quality(rep):
    for s, (max_big, max_err) in {"XAUUSDT": (4, 0.8), "PAXGUSDT": (0, 0.1)}.items():
        d = rep[s]
        assert len(d) == 1636
        assert (d["err_bp"].abs() > 0.1).sum() <= max_big and d["err_bp"].abs().max() < max_err
    # 包络外只有一期，且是 03-05 指数异常那一期
    out = rep["XAUUSDT"][~rep["XAUUSDT"]["in_envelope"]]
    assert list(out.index) == [pd.Timestamp("2026-03-05 08:00")]


@pytest.mark.needs_data
def test_today_params_backfill_is_wrong():
    naive = funding.naive_today_params("XAUUSDT")
    assert (naive["err_bp"].abs() > 0.1).sum() > 50
