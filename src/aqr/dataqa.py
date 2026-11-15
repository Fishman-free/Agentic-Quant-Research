# -*- coding: utf-8 -*-
"""隐藏数据问题：检测 + 证据 + 处理 + 对结论数字的影响（加分项）。

每个函数返回一个 dict，report_numbers 汇总成表并写入 docs/DATA_ISSUES.md。
"""
import numpy as np
import pandas as pd

from . import clock, config as C, features as FE, funding, io_raw as R, panel
from .models import baseline_fit, baseline_predict, rmse

MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)


# ------------------------------------------------------------------ 1. HistData 时钟
def _xau_panel_with_spot(spot_utc: pd.Series) -> pd.DataFrame:
    p = panel.minute_panel("XAUUSDT").copy()
    p["spot"] = spot_utc.reindex(p.index)
    p["basis"] = 1e4 * np.log(p["perp"] / p["spot"])
    p["index_leg"] = 1e4 * np.log(p["index"] / p["spot"])
    return p


def _baseline_numbers(p: pd.DataFrame) -> dict:
    d = FE.hourly("XAUUSDT", p)
    d = d[d["valid"] & (d["split"] != "gap")]
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
    ab = baseline_fit(tr)
    return {"n_train": len(tr), "n_test": len(te), "beta": ab[1],
            "rmse_zero": rmse(te["y"]), "rmse_base": rmse(te["y"] - baseline_predict(te, ab)),
            "basis_sd": float(p["basis"].std())}


def clock_alternatives() -> pd.DataFrame:
    """三种换算规则下的核心数字：本文规则 / 文档式'EST 无夏令时'(恒 UTC−5) / 美国夏令时日期(03-08 切换)。"""
    hd = R.merge_histdata(R.load_histdata_raw())
    local = hd.index
    rules = {
        "本文：03-29 周末由 UTC−5 切到 UTC−4": clock.local_to_utc_end(local),
        "恒为 UTC−5（EST 无夏令时）": local + pd.Timedelta(hours=5) + MIN,
        "按美国夏令时（03-08 切换）": local + pd.to_timedelta(
            np.where(local < pd.Timestamp("2026-03-08 03:00"), 5, 4), unit="h") + MIN,
    }
    rows = {}
    for name, idx in rules.items():
        s = pd.Series(hd["close"].to_numpy(), index=idx)
        s = s[~s.index.duplicated()]
        rows[name] = _baseline_numbers(_xau_panel_with_spot(s))
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------ 2. HistData 重复行
def duplicates() -> dict:
    raw = R.load_histdata_raw()
    dup = raw[raw.duplicated("local", keep=False)]
    nonmono = (raw.groupby("file_month")["local"].diff() < pd.Timedelta(0)).sum()
    g = dup.groupby("local")["close"]
    spread_bp = 1e4 * np.log(g.max() / g.min())
    # 对样本的影响：按题面规则 vs "收盘取第一行" 的另一种合并
    hd_rule = R.merge_histdata(raw)
    alt = hd_rule.copy()
    alt["close"] = raw.sort_values(["file_month", "row"]).groupby("local")["close"].first()
    out = {"first_local": dup["local"].min(), "n_minutes": int(dup["local"].nunique()),
           "n_rows": len(dup), "nonmono_rows": int(nonmono),
           "close_spread_bp_mean": float(spread_bp.mean()), "close_spread_bp_max": float(spread_bp.max())}
    s_alt = pd.Series(alt["close"].to_numpy(), index=clock.local_to_utc_end(alt.index))
    out["alt_first_close"] = _baseline_numbers(_xau_panel_with_spot(s_alt))
    out["rule"] = _baseline_numbers(panel.minute_panel("XAUUSDT"))
    hs = panel.histdata_utc()
    d = FE.samples("XAUUSDT")
    touched = hs["n_rows"].reindex(d.index).gt(1) | hs["n_rows"].reindex(d.index + C.H * HOUR).gt(1).to_numpy()
    out["samples_touched"] = int(touched.sum())
    return out


# ------------------------------------------------------------------ 3. 月度文件缺最后一分钟 → 落在整点样本上
def month_last_minute() -> dict:
    raw = R.load_histdata_raw()
    lost = []
    for m in range(1, 10):
        x = raw[raw["file_month"] == m]["local"].max()
        expect = x + MIN
        utc_end = clock.local_to_utc_end(pd.DatetimeIndex([expect]))[0]
        lost.append((m, expect, utc_end, utc_end.minute == 0))
    return {"rows": lost, "n_on_hour": sum(r[3] for r in lost)}


# ------------------------------------------------------------------ 4. Binance 月度归档缺整天
def missing_day() -> dict:
    res = {}
    for sym in C.SYMBOLS:
        for kind in ("index", "premium"):
            k = panel._bn(kind, sym, False)
            res[(sym, kind)] = int((k["source"] == "daily").sum())
    f = funding.reproduce("PAXGUSDT")
    affected = f[(f.index > pd.Timestamp("2026-06-29")) & (f.index <= pd.Timestamp("2026-06-30 00:00"))]
    res["funding_periods_rescued_per_symbol"] = len(affected)
    return res


# ------------------------------------------------------------------ 5. 溢价指数缺样本 / 参考指数异常
def premium_gaps() -> dict:
    out = {}
    for sym in C.SYMBOLS:
        pre = panel._bn("premium", sym, False)
        out[sym] = {"count_lt12": int((pre["count"] < 12).sum()), "count0": int((pre["count"] == 0).sum())}
    pre = panel._bn("premium", "XAUUSDT", False)
    z = pre[pre["count"] == 0]
    out["count0_blocks"] = sorted({(t.floor("D").date().isoformat(), t.strftime("%H:%M")) for t in z.index})[:3]
    a = funding.reproduce("XAUUSDT", "ohlc4")
    b = funding.reproduce("XAUUSDT", "ohlc4_cnt")
    out["max_err_naive"], out["max_err_cnt"] = a["err_bp"].abs().max(), b["err_bp"].abs().max()
    out["rmse_naive"], out["rmse_cnt"] = float(np.sqrt((a["err_bp"] ** 2).mean())), float(np.sqrt((b["err_bp"] ** 2).mean()))
    return out


def index_anomaly_0305() -> dict:
    p = panel.minute_panel("XAUUSDT")
    w = p.loc["2026-03-05 04:31":"2026-03-05 05:54"]
    idx = panel._bn("index", "XAUUSDT", False)["close"]
    two_dec = (np.abs(idx * 100 - np.round(idx * 100)) < 1e-6)
    m = funding.reproduce("XAUUSDT").loc[pd.Timestamp("2026-03-05 08:00")]
    return {"index_vs_spot_max_bp": float(w["index_leg"].max()),
            "perp_vs_spot_med_bp": float(w["basis"].median()),
            "prem_min_bp": float(1e4 * w["prem_close"].min()),
            "two_dec_minutes": int(two_dec.loc["2026-03-05 04:30":"2026-03-05 06:00"].sum()),
            "official_bp": m["official"] * 1e4, "calc_bp": m["calc"] * 1e4,
            "env_lo": m["F_lo"] * 1e4, "env_hi": m["F_hi"] * 1e4}


def usdt_switch() -> dict:
    """07-16 换算：分钟级定位指数腿跳变时刻，并量化跳变幅度。"""
    p = panel.minute_panel("XAUUSDT")
    w = p.loc["2026-07-16 07:30":"2026-07-16 10:00"]
    pre = 1e4 * w["prem_close"]
    jump_t = pre.diff().abs().idxmax()
    before = p.loc["2026-07-01":"2026-07-15", "index_leg"].mean()
    after = p.loc["2026-07-17":"2026-09-30", "index_leg"].mean()
    prem_frozen = panel._bn("premium", "XAUUSDT", False).loc["2026-07-16 08:00":"2026-07-16 09:00", "count"].eq(0).sum()
    return {"jump_minute": jump_t, "prem_jump_bp": float(pre.diff().loc[jump_t]),
            "index_leg_before": float(before), "index_leg_after": float(after), "prem_frozen_min": int(prem_frozen)}


# ------------------------------------------------------------------ 6. 资金费归档与边界
def funding_archive() -> dict:
    out = {}
    for sym in C.SYMBOLS:
        f = panel.funding(sym)
        full = pd.date_range("2026-01-01", "2026-09-30 20:00", freq="4h")
        out[sym] = {"missing": [str(x) for x in full.difference(f.index)],
                    "calc_ms_max": int((f["calc_time"] % 3_600_000).max()),
                    "test_0624": funding.missing_settlement_test(sym)}
    m = funding.reproduce("XAUUSDT")
    naive = funding.reproduce("XAUUSDT", params=lambda s, t: _naive_boundary(s, t))
    out["boundary_err_naive"] = float(naive.loc[pd.Timestamp("2026-01-23 08:00"), "err_bp"])
    out["boundary_err_rule"] = float(m.loc[pd.Timestamp("2026-01-23 08:00"), "err_bp"])
    return out


def _naive_boundary(sym, t):
    """照字面"自 08:00 起 I=0"、把 08:00 结算期也算成新参数。"""
    from . import funding_params as FP
    seg = FP.params_for(sym, t)
    if sym == "XAUUSDT" and t == pd.Timestamp("2026-01-23 08:00"):
        return FP.Seg(sym, t, 0.0, seg.cap, seg.floor, "", "")
    return seg


# ------------------------------------------------------------------ 7. 1 月陈旧成交 / HistData 偏低
def stale_jan() -> dict:
    p = panel.minute_panel("XAUUSDT")
    z = (p["perp_vol"] == 0).groupby(p.index.month).mean()
    d = FE.samples("XAUUSDT")
    tr = d[d["split"] == "train"]
    ab_all = baseline_fit(tr)
    ab_nojan = baseline_fit(tr[tr.index >= pd.Timestamp("2026-02-01")])
    return {"zero_vol_share_jan": float(z.loc[1]), "zero_vol_share_feb": float(z.loc[2]),
            "beta_all": ab_all[1], "beta_ex_jan": ab_nojan[1]}


def histdata_bias() -> dict:
    p = panel.minute_panel("XAUUSDT")
    x = p["index_leg"]
    return {"pre_0716_median": float(x[x.index < C.XAU_INDEX_USDT_CONV].median()),
            "by_month": x.groupby(x.index.month).median().round(2).to_dict()}


def close_minute_effect() -> dict:
    """日休前最后一分钟 / 开盘第一分钟的 HistData 报价是否系统性偏离（以 XAU 基差平均值衡量）。"""
    from . import closure
    p = panel.minute_panel("XAUUSDT")
    g = closure.episodes()
    g = g[g["kind"] == "日休"]
    c, o = pd.DatetimeIndex(g["close"]), pd.DatetimeIndex(g["reopen"])
    b = lambda idx: float(np.nanmean(p["basis"].reindex(idx).to_numpy()))
    d = FE.samples("XAUUSDT")
    hit = d.index.isin(c) | (d.index + C.H * HOUR).isin(c)
    return {"basis_close_m1": b(c - MIN), "basis_close": b(c), "basis_open": b(o), "basis_open_p5": b(o + 5 * MIN),
            "n_breaks": len(g), "samples_on_close_minute": int(hit.sum())}


def run_all() -> dict:
    return {"clock": clock_alternatives(), "dups": duplicates(), "lastmin": month_last_minute(),
            "missing_day": missing_day(), "premium_gaps": premium_gaps(), "idx0305": index_anomaly_0305(),
            "usdt": usdt_switch(), "funding_archive": funding_archive(), "stale": stale_jan(),
            "hd_bias": histdata_bias(), "close_minute": close_minute_effect()}
