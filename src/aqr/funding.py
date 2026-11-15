# -*- coding: utf-8 -*-
"""资金费机制复现。

F = [P̄ + clamp(I − P̄, −0.05%, +0.05%)] / (8/N)，再截在 [floor, cap] 内。

P̄ 的构造（Binance：每 5 秒一个溢价指数样本，按样本序号 1,2,…,n 线性加权）：
  一分钟溢价 K 线只给出这一分钟 ≤12 个样本的 OHLC 与样本数 count，所以只能近似。
  主方法 ohlc4_cnt：分钟值取 (O+H+L+C)/4；分钟权重取这一分钟内样本序号之和
  （按 count 累计：前面缺样本的分钟不占序号，count=0 的分钟权重为 0）。
  其余方法作为敏感性对照。
误差包络：把每分钟的值分别换成 low/high，得到 P̄ 的下界/上界——官方值若落在 [F(P̄_lo), F(P̄_hi)]
之内，说明不匹配完全可以由"分钟内样本位置未知"解释（F 对 P̄ 单调不减）。
"""
import numpy as np
import pandas as pd

from . import config as C, funding_params as FP, panel

METHODS = ("ohlc4_cnt", "ohlc4", "oc2", "close_lin", "mean_close")
HOUR = pd.Timedelta(hours=1)


def _weights(cnt: np.ndarray, method: str) -> np.ndarray:
    n = len(cnt)
    j = np.arange(n, dtype=float)
    if method == "mean_close":
        return np.ones(n)
    if method == "close_lin":
        return j + 1
    if method in ("ohlc4", "oc2"):
        return 144 * j + 78  # 满 12 个样本时，第 j 分钟样本序号之和
    s = np.concatenate([[0.0], np.cumsum(cnt)[:-1]])
    return cnt * s + cnt * (cnt + 1) / 2


def _values(o, h, l, c, method):
    if method in ("close_lin", "mean_close"):
        return c
    if method == "oc2":
        return (o + c) / 2
    return (o + h + l + c) / 4


def pbar(w: pd.DataFrame, method: str = "ohlc4_cnt") -> dict:
    o, h, l, c = (w[x].to_numpy(float) for x in ("prem_open", "prem_high", "prem_low", "prem_close"))
    cnt = w["prem_count"].to_numpy(float)
    wt = _weights(cnt, method)
    sw = wt.sum()
    return {"P": (_values(o, h, l, c, method) * wt).sum() / sw,
            "P_lo": (l * wt).sum() / sw, "P_hi": (h * wt).sum() / sw,
            "n_min": len(w), "n_partial": int((cnt < 12).sum()), "n_zero": int((cnt == 0).sum())}


def formula(P, I, N, cap=None, floor=None, band=FP.BAND):
    F = (P + np.clip(I - P, -band, band)) / (8.0 / N)
    if cap is not None:
        F = np.minimum(F, cap)
    if floor is not None:
        F = np.maximum(F, floor)
    return F


def premium_minutes(sym: str) -> pd.DataFrame:
    p = panel.minute_panel(sym)
    pre = panel._bn("premium", sym, False)
    out = p[["prem_open", "prem_high", "prem_low", "prem_close"]].copy()
    out["prem_count"] = pre["count"].reindex(out.index)
    return out


def comparable_settlements(sym: str) -> pd.DataFrame:
    """题面口径：已公布、结算时刻严格在 (2026-01-01, 2026-10-01) 内、且平均窗口完整落在区间内。"""
    f = panel.funding(sym)
    f = f[(f.index > C.START) & (f.index < C.END)].copy()
    f["window_start"] = f.index - pd.to_timedelta(f["N"], unit="h")
    return f[f["window_start"] >= C.START]


def reproduce(sym: str, method: str = "ohlc4_cnt", params=FP.params_for) -> pd.DataFrame:
    pre = premium_minutes(sym)
    f = comparable_settlements(sym)
    rows = []
    for t, r in f.iterrows():
        w = pre.loc[(pre.index > r.window_start) & (pre.index <= t)]
        seg = params(sym, t)
        pb = pbar(w, method)
        F = formula(pb["P"], seg.I, r.N, seg.cap, seg.floor)
        F_lo = formula(pb["P_lo"], seg.I, r.N, seg.cap, seg.floor)
        F_hi = formula(pb["P_hi"], seg.I, r.N, seg.cap, seg.floor)
        F_free = formula(pb["P"], seg.I, r.N)
        rows.append({"t": t, "N": int(r.N), "I": seg.I, "cap": seg.cap, "official": r.rate,
                     "calc": F, "calc_uncapped": F_free, "F_lo": F_lo, "F_hi": F_hi, **pb})
    d = pd.DataFrame(rows).set_index("t")
    d["err_bp"] = (d["calc"] - d["official"]) * 1e4
    d["in_envelope"] = (d["official"] >= d["F_lo"] - 1e-10) & (d["official"] <= d["F_hi"] + 1e-10)
    gap = d["P"] - d["I"]
    d["state"] = np.where(gap.abs() <= FP.BAND, "dead", np.where(gap > 0, "above", "below"))
    d["capped"] = d["calc"] != d["calc_uncapped"]
    d["reason"] = classify(d)
    return d


def classify(d: pd.DataFrame) -> pd.Series:
    """误差 >0.1bp 的期归因（按优先级）。"""
    big = d["err_bp"].abs() > 0.1
    r = pd.Series("", index=d.index)
    r[big & d["in_envelope"]] = "分钟内采样位置未知（落在包络内）"
    r[big & ~d["in_envelope"] & (d["n_partial"] > 0)] = "溢价样本缺失+参考指数异常（包络外）"
    r[big & (r == "")] = "包络外"
    return r


def summary(d: pd.DataFrame) -> dict:
    e = d["err_bp"].abs()
    return {"n": len(d), "max_abs": e.max(), "n_gt_0.1": int((e > 0.1).sum()),
            "n_gt_0.01": int((e > 0.01).sum()), "rmse": float(np.sqrt((d["err_bp"] ** 2).mean())),
            "median_abs": e.median(), "share_in_env": d["in_envelope"].mean()}


# ---------------------------------------------------------------- 参数推断（证据）
def implied_interest(d: pd.DataFrame) -> pd.DataFrame:
    """P̄ 稳在两种候选 I 的死区内时，官方 F·8/N 就是 I 本身：直接读出各期利息项。"""
    safe = (d["P_hi"] < 3.5e-4) & (d["P_lo"] > -3.5e-4)  # 0 与 1bp 两种 I 下都在死区里
    x = d.loc[safe, ["official", "N"]].copy()
    x["I_implied_bp"] = x["official"] * 8 / x["N"] * 1e4
    return x


def naive_today_params(sym: str, method: str = "ohlc4_cnt") -> pd.DataFrame:
    """反例：用"今天"（样本末）的参数回填全部 9 个月，看会错多少。"""
    last = [s for s in FP.SEGMENTS if s.sym == sym][-1]
    return reproduce(sym, method, params=lambda s, t: last)


def interval_hypotheses(sym="XAUUSDT", t=pd.Timestamp("2026-01-30 16:00")) -> pd.DataFrame:
    """01-30 16:00：字段 N=8 但距上期只隔 4 小时。逐一检验窗口/N/上下限假设。"""
    pre = premium_minutes(sym)
    off = panel.funding(sym).loc[t, "rate"]
    rows = []
    for win in (4, 8):
        w = pre.loc[(pre.index > t - win * HOUR) & (pre.index <= t)]
        P = pbar(w)["P"]
        for N in (4, 8):
            for cap in (None, 5e-4):
                F = formula(P, 0.0, N, cap, -cap if cap else None)
                rows.append({"窗口(h)": win, "N": N, "上下限": "±0.05%" if cap else "无",
                             "F_bp": F * 1e4, "官方_bp": off * 1e4, "误差_bp": (F - off) * 1e4})
    return pd.DataFrame(rows)


def missing_settlement_test(sym: str, t_next=pd.Timestamp("2026-06-24 08:00")) -> dict:
    """06-24 04:00 缺失：若这一期真没结算，08:00 期应按 8h 窗口算；比较 4h/8h 两种窗口谁对得上。"""
    pre = premium_minutes(sym)
    off = panel.funding(sym).loc[t_next, "rate"]
    seg = FP.params_for(sym, t_next)
    out = {"official_bp": off * 1e4}
    for win in (4, 8):
        w = pre.loc[(pre.index > t_next - win * HOUR) & (pre.index <= t_next)]
        out[f"err_{win}h_bp"] = (formula(pbar(w)["P"], seg.I, win, seg.cap, seg.floor) - off) * 1e4
    t_miss = t_next - 4 * HOUR
    w = pre.loc[(pre.index > t_miss - 4 * HOUR) & (pre.index <= t_miss)]
    out["calc_missing_bp"] = formula(pbar(w)["P"], seg.I, 4, seg.cap, seg.floor) * 1e4
    return out


def run_all() -> dict:
    res = {}
    for sym in C.SYMBOLS:
        main = reproduce(sym)
        res[sym] = {"main": main, "summary": summary(main),
                    "methods": {m: summary(reproduce(sym, m)) for m in METHODS},
                    "naive": summary(naive_today_params(sym)),
                    "implied_I": implied_interest(main),
                    "missing_0624": missing_settlement_test(sym)}
    res["XAUUSDT"]["interval_0130"] = interval_hypotheses()
    return res
