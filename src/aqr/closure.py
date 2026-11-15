# -*- coding: utf-8 -*-
"""深挖角度：现货黄金休市时，XAU 永续的基差和资金费发生了什么？休市期间永续的走势能不能预告开盘金价？

自然实验：2026-05-08 21:00 起，休市期间 XAU 参考指数改由永续自身盘口生成（此前冻结在最后报价）。
  * 之前：指数冻结 → 溢价 = 永续相对"周五收盘价"的涨跌 → 资金费把永续往一个过时的价格拉；
  * 之后：指数跟着永续走 → 溢价≈0 → 资金费≈利息项 → 资金费对真实金价的偏离完全不拉。
休市事件由 HistData 实际缺口识别（这里是事后描述，不进入预测特征）。
"""
import numpy as np
import pandas as pd

from . import config as C, funding_params as FP, panel

MIN = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)


def episodes(min_gap=pd.Timedelta(minutes=30)) -> pd.DataFrame:
    hd = panel.histdata_utc()
    t = hd.index.to_series()
    gap = t.diff()
    m = gap >= min_gap
    ep = pd.DataFrame({"close": t.shift()[m].values, "reopen": t[m].values})
    ep["hours"] = (ep["reopen"] - ep["close"]) / HOUR
    ep["kind"] = np.where(ep["hours"] >= 24, "周末/假日", np.where(ep["hours"] >= 0.9, "日休", "数据缺口"))
    ep["post58"] = ep["close"] >= C.XAU_INDEX_CLOSED_RULE
    return ep[(ep["close"] > C.START) & (ep["reopen"] < C.END)].reset_index(drop=True)


def _lr(s: pd.Series, a, b):
    va, vb = s.reindex(pd.DatetimeIndex(a)).to_numpy(), s.reindex(pd.DatetimeIndex(b)).to_numpy()
    return 1e4 * (np.log(vb) - np.log(va))


def episode_table() -> pd.DataFrame:
    px, pp = panel.minute_panel("XAUUSDT"), panel.minute_panel("PAXGUSDT")
    ep = episodes()
    c, o = ep["close"], ep["reopen"]
    pre_open = o - MIN  # 开盘前最后一分钟（永续一直在交易）
    ep["gold_gap"] = _lr(px["spot"], c, o)
    ep["xau_perp_move"] = _lr(px["perp"], c, pre_open)
    ep["paxg_spot_move"] = _lr(pp["spot"], c, pre_open)
    ep["paxg_perp_move"] = _lr(pp["perp"], c, pre_open)
    ep["basis_close"] = px["basis"].reindex(c).to_numpy()
    ep["basis_reopen"] = px["basis"].reindex(o).to_numpy()
    ep["basis_reopen_30m"] = px["basis"].reindex(o + 30 * MIN).to_numpy()
    rows = []
    for a, b in zip(c, o):
        w = px.loc[(px.index > a) & (px.index < b)]
        frozen = (w["index"].diff().abs() < 1e-9).mean() if len(w) > 1 else np.nan
        rows.append((frozen, w["prem_leg"].abs().median(), w["prem_leg"].iloc[-1] if len(w) else np.nan))
    ep[["index_frozen_share", "prem_leg_absmed", "prem_leg_last"]] = rows
    # 休市期间结算的资金费
    fr = panel.funding("XAUUSDT")
    s = []
    for a, b in zip(c, o):
        f = fr[(fr.index > a) & (fr.index < b)]
        s.append((len(f), 1e4 * f["rate"].sum(), 1e4 * f["rate"].abs().mean() if len(f) else np.nan))
    ep[["n_settle", "funding_sum_bp", "funding_absmean_bp"]] = s
    return ep


def predictive(ep: pd.DataFrame, x: str, y: str = "gold_gap", B=2000, seed=C.SEED) -> dict:
    d = ep[[x, y]].dropna()
    if len(d) < 5:
        return {"n": len(d)}
    X, Y = d[x].to_numpy(), d[y].to_numpy()
    b, a = np.polyfit(X, Y, 1)
    r2 = np.corrcoef(X, Y)[0, 1] ** 2
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        if np.ptp(X[i]) > 0:
            bs.append(np.polyfit(X[i], Y[i], 1)[0])
    # 方向命中率与"开盘缺口有多少已被永续提前走完"
    hit = np.mean(np.sign(X) == np.sign(Y))
    resid_sd, gap_sd = np.std(Y - (a + b * X)), np.std(Y)
    return {"n": len(d), "slope": b, "intercept": a, "r2": r2, "ci": tuple(np.quantile(bs, [.025, .975])),
            "hit": hit, "gap_sd": gap_sd, "resid_sd": resid_sd}


def funding_in_closures() -> pd.DataFrame:
    """结算时刻落在现货休市/开市时，官方资金费的绝对值与溢价——按 05-08 前后分组。"""
    from . import calendar as K
    fr = panel.funding("XAUUSDT")
    fr = fr[(fr.index > C.START) & (fr.index < C.END)]
    px = panel.minute_panel("XAUUSDT")
    # 结算前 4h 窗口里现货有没有数据：完全无数据=休市期结算
    has = px["spot"].notna()
    share_open = np.array([has.loc[(has.index > t - 4 * HOUR) & (has.index <= t)].mean() for t in fr.index])
    d = pd.DataFrame({"F_bp": fr["rate"].to_numpy() * 1e4, "share_open": share_open,
                      "post58": fr.index >= C.XAU_INDEX_CLOSED_RULE}, index=fr.index)
    d["state"] = np.where(d["share_open"] == 0, "全休市", np.where(d["share_open"] < 1, "部分休市", "全开市"))
    g = d.groupby(["post58", "state"])["F_bp"].agg(n="size", absmean=lambda s: s.abs().mean(),
                                                   mean="mean", nonzero=lambda s: (s.abs() > 1e-9).mean())
    return g.reset_index()


def reopen_profile(ep: pd.DataFrame, kind="周末/假日", horizon=240) -> pd.DataFrame:
    """开盘后 XAU 基差 |basis − 收盘前基差| 的平均路径（分钟），按 05-08 前后。"""
    px = panel.minute_panel("XAUUSDT")
    offs = np.arange(0, horizon + 1)
    out = {}
    for post, g in ep[ep["kind"] == kind].groupby("post58"):
        M = np.vstack([px["basis"].reindex(o + offs * MIN).to_numpy() - bc
                       for o, bc in zip(g["reopen"], g["basis_close"])])
        out["05-08 后" if post else "05-08 前"] = pd.Series(np.nanmean(np.abs(M), axis=0), index=offs)
    return pd.DataFrame(out)


def closure_path(ep: pd.DataFrame, kind="周末/假日") -> dict:
    """一个代表性周末：指数、永续、溢价的分钟路径（画图用），05-08 前后各取一个波动最大的。"""
    px = panel.minute_panel("XAUUSDT")
    res = {}
    for post, g in ep[ep["kind"] == kind].groupby("post58"):
        r = g.loc[g["gold_gap"].abs().idxmax()]
        w = px.loc[r["close"] - 6 * HOUR: r["reopen"] + 6 * HOUR, ["perp", "index", "spot", "prem_leg"]]
        res["05-08 后" if post else "05-08 前"] = (r, w)
    return res


def cross_spread(ep: pd.DataFrame, horizon=240) -> dict:
    """05-08 后周末：XAU 永续相对 PAXG 代币的价差在休市期间变了多少，开盘后回吐多少（下一步方向的对照）。"""
    px, pp = panel.minute_panel("XAUUSDT"), panel.minute_panel("PAXGUSDT")
    sp = 1e4 * np.log(px["perp"] / pp["spot"])
    g = ep[(ep["kind"] == "周末/假日") & ep["post58"]]
    c, o = pd.DatetimeIndex(g["close"]), pd.DatetimeIndex(g["reopen"])
    wk = sp.reindex(o - MIN).to_numpy() - sp.reindex(c).to_numpy()
    after = sp.reindex(o + horizon * MIN).to_numpy() - sp.reindex(o - MIN).to_numpy()
    ok = ~np.isnan(wk) & ~np.isnan(after)
    return {"post_abs": float(np.mean(np.abs(wk[ok]))), "post_slope": float(np.polyfit(wk[ok], after[ok], 1)[0]),
            "n": int(ok.sum())}


def run() -> dict:
    ep = episode_table()
    out = {"episodes": ep, "funding": funding_in_closures(), "reopen": reopen_profile(ep),
           "paths": closure_path(ep), "spread": cross_spread(ep)}
    pred = {}
    for kind in ("周末/假日", "日休"):
        for post in (False, True):
            sub = ep[(ep["kind"] == kind) & (ep["post58"] == post)]
            for x in ("xau_perp_move", "paxg_spot_move", "paxg_perp_move"):
                pred[(kind, "05-08 后" if post else "05-08 前", x)] = predictive(sub, x)
        sub = ep[ep["kind"] == kind]
        for x in ("xau_perp_move", "paxg_spot_move", "paxg_perp_move"):
            pred[(kind, "全样本", x)] = predictive(sub, x)
    out["pred"] = pred
    # 05-08 前：休市溢价（= 永续相对冻结指数的涨跌）与开盘缺口同号吗？——资金费在"拉错方向"
    wk = ep[(ep["kind"] == "周末/假日")]
    out["wrong_way"] = {post: {"n": len(g), "sign_agree_prem_gap": float(np.mean(np.sign(g["prem_leg_last"]) == np.sign(g["gold_gap"]))),
                               "funding_sum_mean_abs": float(g["funding_sum_bp"].abs().mean()),
                               "frozen_share": float(g["index_frozen_share"].mean()),
                               "prem_absmed": float(g["prem_leg_absmed"].median())}
                        for post, g in wk.groupby("post58")}
    return out
