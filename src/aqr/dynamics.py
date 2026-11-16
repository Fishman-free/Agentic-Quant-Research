# -*- coding: utf-8 -*-
"""拉得多快、谁向谁靠、是不是资金费在拉。

1. 半衰期：E[Δ4h dev] ≈ β·dev ⇒ 每 4h 剩 (1+β)，半衰期 = 4·ln0.5 / ln(1+β)。
   注意这是"相对 24h 移动均值"的回归，不是回到 0。
2. 局部投影：h = 1..24 小时，y_h = 基差_{t+h} − 基差_t 对 dev_24h 回归，画衰减曲线。
3. 噪声偏差：基差_t 同时出现在 y 和 dev 里，分钟噪声会机械地造出负 β。
   用 t−15 分钟的偏离作工具变量（噪声在 15 分钟内不相关），给出"去噪"β。
4. 谁在动：Δ永续、Δ现货（都相对第三个独立的黄金价格）分别对 dev 回归，β = α_永续 − α_现货；
   α_永续/β 是"永续向现货靠"的份额（类似 VECM 调整系数 / Gonzalo-Granger 权重）。
   直接用 4h 收益做会被共同金价波动淹没（标准误 ~0.6），所以必须扣掉参照市场。
5. 资金费的作用：资金费只作用于溢价（永续−指数）。比较 P̄ 在死区内（F≡利息项，不随溢价变）
   与死区外（F 随溢价线性变）时溢价的回归速度，并在相同偏离幅度分档里比较。
   再做结算时刻 ±60 分钟的事件研究。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C, features as FE, funding_params as FP, panel

HOUR = pd.Timedelta(hours=1)
MIN = pd.Timedelta(minutes=1)


def half_life(beta: float, h: float = C.H) -> float:
    return float(h * np.log(0.5) / np.log1p(beta)) if -1 < beta < 0 else np.nan


def ols_hac(y, X, lag=24):
    X = sm.add_constant(np.asarray(X, float))
    r = sm.OLS(np.asarray(y, float), X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return r


def _hourly_with(sym, hs=range(1, 25)):
    p = panel.minute_panel(sym)
    d = FE.hourly(sym, p)
    t = d.index
    for h in hs:
        d[f"y{h}"] = p["basis"].reindex(t + h * HOUR).to_numpy() - d["basis"].to_numpy()
    for leg in ("perp", "spot", "index"):
        d[f"r_{leg}_4h_fwd"] = 1e4 * (np.log(p[leg].reindex(t + C.H * HOUR).to_numpy()) - np.log(p[leg].reindex(t).to_numpy()))
    d["prem_leg_fut"] = p["prem_leg"].reindex(t + C.H * HOUR).to_numpy()
    d["prem_leg_now"] = p["prem_leg"].reindex(t).to_numpy()
    lag = t - 15 * MIN
    m24 = p["basis"].rolling("24h", min_periods=1).mean()
    d["dev24_lag15"] = (p["basis"] - m24).reindex(lag).to_numpy()
    d["sample"] = d["valid"] & (d["split"] != "gap")
    return d


def local_projection(d, hs=range(1, 25), sub=None):
    """Jordà (2005) 局部投影法：测算基差偏离在未来 h 小时（h=1..24）的动态脉冲响应衰减路径。
    
    采用 Newey-West HAC 异方差与自相关稳健标准误（maxlags=max(h, 24)）。
    """
    rows = []
    for h in hs:
        x = d[d["sample"]] if sub is None else d[d["sample"] & sub]
        x = x[[f"y{h}", "dev_24h"]].dropna()
        r = ols_hac(x[f"y{h}"], x[["dev_24h"]], lag=max(h, 24))
        rows.append({"h": h, "beta": r.params[1], "se": r.bse[1], "n": len(x)})
    return pd.DataFrame(rows).set_index("h")


def iv_beta(d, mask):
    """工具变量（IV）估计：利用滞后 15 分钟的偏离 dev24_lag15 作为工具变量，排除因变量与自变量共享当前价格导致的买卖微观价差弹跳伪回归噪声。"""
    x = d[d["sample"] & mask][["y", "dev_24h", "dev24_lag15"]].dropna()
    # 2SLS（恰好识别）：β_IV = cov(z, y) / cov(z, x)
    z, xv, y = (x[c] - x[c].mean() for c in ("dev24_lag15", "dev_24h", "y"))
    return float((z * y).sum() / (z * xv).sum()), len(x)


def leg_shares(d, mask, dev="dev_24h"):
    """微观调整份额分解：
    通过引入第三方对冲资产剥离宏观系统性黄金 Beta，精准测算 4 小时基差修复中永续合约与现货价格各自承担的调整比例。
    """
    x = d[d["sample"] & mask].dropna(subset=["r_perp_4h_fwd", "r_spot_4h_fwd", dev])
    rp = ols_hac(x["r_perp_4h_fwd"], x[[dev]])
    rs = ols_hac(x["r_spot_4h_fwd"], x[[dev]])
    ap, asp = rp.params[1], rs.params[1]
    return {"alpha_perp": ap, "se_perp": rp.bse[1], "alpha_spot": asp, "se_spot": rs.bse[1],
            "beta": ap - asp, "share_perp": ap / (ap - asp), "n": len(x)}


def leg_shares_iv(d, mask):
    """永续、现货各自对 dev 的 IV 回归（工具 = t−15 分钟的偏离），排除"现货报价噪声"造成的机械回归。"""
    x = d[d["sample"] & mask].dropna(subset=["r_perp_4h_fwd", "r_spot_4h_fwd", "dev_24h", "dev24_lag15"])
    z, xv = x["dev24_lag15"] - x["dev24_lag15"].mean(), x["dev_24h"] - x["dev_24h"].mean()
    ap = float((z * (x["r_perp_4h_fwd"] - x["r_perp_4h_fwd"].mean())).sum() / (z * xv).sum())
    asp = float((z * (x["r_spot_4h_fwd"] - x["r_spot_4h_fwd"].mean())).sum() / (z * xv).sum())
    return {"alpha_perp": ap, "alpha_spot": asp, "beta": ap - asp, "share_perp": ap / (ap - asp)}


REFS = {"XAUUSDT": ("PAXGUSDT", "spot", "PAXG 代币"), "PAXGUSDT": ("XAUUSDT", "perp", "XAU 永续")}


def leg_shares_rel(sym: str, d: pd.DataFrame, mask, iv: bool = False) -> dict:
    """谁在动（相对第三个市场）：r = Δ4h ln(价格/参照)。参照是另一个独立交易的黄金价格，
    共同的金价波动被扣掉后，永续、现货各自的调整系数才估得准；且 α_永续 − α_现货 ≡ β（参照项抵消）。
    iv=True 时用 t−15 分钟的偏离作工具，排除"现货报价噪声"造成的机械回归。"""
    osym, col, _ = REFS[sym]
    p, ref = panel.minute_panel(sym), panel.minute_panel(osym)[col]
    x = d[d["sample"] & mask]
    t, t4 = x.index, x.index + C.H * HOUR
    lr = lambda s_: 1e4 * (np.log(s_.reindex(t4).to_numpy()) - np.log(s_.reindex(t).to_numpy()))
    a = pd.DataFrame({"rp": lr(p["perp"]) - lr(ref), "rs": lr(p["spot"]) - lr(ref),
                      "dev": x["dev_24h"].to_numpy(), "z": x["dev24_lag15"].to_numpy()}, index=t).dropna()
    if iv:
        z, xv = a["z"] - a["z"].mean(), a["dev"] - a["dev"].mean()
        ap = float((z * (a["rp"] - a["rp"].mean())).sum() / (z * xv).sum())
        asp = float((z * (a["rs"] - a["rs"].mean())).sum() / (z * xv).sum())
        return {"alpha_perp": ap, "alpha_spot": asp, "beta": ap - asp, "share_perp": ap / (ap - asp), "n": len(a)}
    rp, rs = ols_hac(a["rp"], a[["dev"]]), ols_hac(a["rs"], a[["dev"]])
    ap, asp = rp.params[1], rs.params[1]
    return {"alpha_perp": ap, "se_perp": rp.bse[1], "alpha_spot": asp, "se_spot": rs.bse[1],
            "beta": ap - asp, "share_perp": ap / (ap - asp), "n": len(a)}


def vecm(sym: str, step: int = 5, lags: int = 3, mask_fn=None) -> dict:
    """高频误差修正：Δln永续、Δln现货（bp/步）对上一时刻偏离 e = 基差 − 24h 均值 回归，控制 lags 阶滞后收益。
    4h 收益里共同的金价波动（标准差 ~40bp）会把永续、现货各自的调整系数淹没；在 1–5 分钟频率上，共同波动小得多。
    返回 α_永续（<0 表示永续向现货靠）、α_现货（>0 表示现货向永续靠）与永续承担的调整份额。"""
    p = panel.minute_panel(sym)
    g = p.iloc[step - 1::step]
    lp, ls = 1e4 * np.log(g["perp"]), 1e4 * np.log(g["spot"])
    e = (p["basis"] - p["basis"].rolling("24h", min_periods=1).mean()).reindex(g.index)
    X = pd.DataFrame({"e": e.shift(1)}, index=g.index)
    dp, ds = lp.diff(), ls.diff()
    for k in range(1, lags + 1):
        X[f"dp{k}"], X[f"ds{k}"] = dp.shift(k), ds.shift(k)
    ok = X.notna().all(axis=1) & dp.notna() & ds.notna()
    if mask_fn is not None:
        ok &= mask_fn(g.index)
    rp = ols_hac(dp[ok], X[ok], lag=6)
    rs = ols_hac(ds[ok], X[ok], lag=6)
    ap, asp = rp.params[1], rs.params[1]
    return {"alpha_perp": ap, "se_perp": rp.bse[1], "alpha_spot": asp, "se_spot": rs.bse[1],
            "share_perp": ap / (ap - asp), "per_step": ap - asp, "n": int(ok.sum()), "step_min": step}


def prem_index_legs(sym: str, d: pd.DataFrame, mask=None) -> dict:
    """基差 = 溢价（永续−指数）+ 指数差（指数−现货）：4h 回归里各占多少。
    资金费只作用于溢价；指数差的回归与资金费无关。两者都是价差，不含共同金价波动，估计很准。"""
    p = panel.minute_panel(sym)
    x = d[d["sample"]] if mask is None else d[d["sample"] & mask]
    t, t4 = x.index, x.index + C.H * HOUR
    a = pd.DataFrame({"dprem": p["prem_leg"].reindex(t4).to_numpy() - p["prem_leg"].reindex(t).to_numpy(),
                      "didx": p["index_leg"].reindex(t4).to_numpy() - p["index_leg"].reindex(t).to_numpy(),
                      "dev": x["dev_24h"].to_numpy()}, index=t).dropna()
    r1, r2 = ols_hac(a["dprem"], a[["dev"]]), ols_hac(a["didx"], a[["dev"]])
    return {"prem": r1.params[1], "se_prem": r1.bse[1], "idx": r2.params[1], "se_idx": r2.bse[1], "n": len(a)}


def premium_leg_by_state(d, sym):
    """溢价 4h 变化 vs 溢价偏离，按死区内/外、偏离幅度分档。"""
    x = d[d["sample"]].dropna(subset=["prem_leg_fut", "prem_leg_now", "prem_leg_dev"]).copy()
    x["dprem"] = x["prem_leg_fut"] - x["prem_leg_now"]
    x["absdev"] = x["prem_leg_dev"].abs()
    x["bin"] = pd.cut(x["absdev"], [0, 1, 2, 4, 8, np.inf], right=False)
    rows = []
    for (st, b), g in x.groupby([x["outside_dead"].map({0.0: "死区内", 1.0: "死区外"}), "bin"], observed=True):
        if len(g) < 30:
            continue
        r = ols_hac(g["dprem"], g[["prem_leg_dev"]])
        rows.append({"状态": st, "|偏离|档(bp)": str(b), "beta": r.params[1], "se": r.bse[1], "n": len(g)})
    tot = []
    for st, g in x.groupby(x["outside_dead"]):
        r = ols_hac(g["dprem"], g[["prem_leg_dev"]])
        tot.append({"状态": "死区外" if st else "死区内", "beta": r.params[1], "se": r.bse[1], "n": len(g),
                    "median_absdev": g["absdev"].median()})
    return pd.DataFrame(rows), pd.DataFrame(tot)


def settlement_event(sym, pre=60, post=60, shift_h=0):
    """结算时刻 ±60 分钟：溢价指数（bp，相对 −60 分钟）的平均路径，按本期官方资金费正/负/死区分组。
    shift_h≠0 时把事件点挪到非结算时刻做安慰剂（分组仍按本期资金费），用来排除"条件均值回归"。"""
    p = panel.minute_panel(sym)
    fr = panel.funding(sym)
    fr = fr[(fr.index > C.START + 2 * HOUR) & (fr.index < C.END - 2 * HOUR)]
    prem = 1e4 * p["prem_close"]
    I_t = FP.interest_at(sym, fr.index)
    F8 = fr["rate"].to_numpy() * 8 / fr["N"].to_numpy()
    grp = np.where(np.isclose(F8, I_t, atol=1e-9), "F=利息项", np.where(F8 > I_t, "F>利息项", "F<利息项"))
    offs = np.arange(-pre, post + 1)
    paths, counts = {}, {}
    for g in np.unique(grp):
        ts = fr.index[grp == g] - shift_h * HOUR
        M = np.vstack([prem.reindex(t + offs * MIN).to_numpy() for t in ts])
        M = M - M[:, [0]]
        paths[g] = pd.Series(np.nanmean(M, axis=0), index=offs)
        counts[g] = len(ts)
    out = pd.DataFrame(paths)
    out.attrs["n"] = counts
    return out


def premium_distribution(sym) -> dict:
    """各期 P̄（bp）相对死区的位置：死区内资金费恒为利息项，没有边际拉力。"""
    from . import funding
    m = funding.reproduce(sym)
    P = m["P"] * 1e4
    return {"P_median": float(P.median()), "P_q10": float(P.quantile(.1)), "P_q90": float(P.quantile(.9)),
            "share_dead": float((m["state"] == "dead").mean()), "share_above": float((m["state"] == "above").mean()),
            "share_below": float((m["state"] == "below").mean())}


def run(sym: str, model_res: dict = None) -> dict:
    d = _hourly_with(sym)
    tr = d["split"] == "train"
    beta_ols = ols_hac(d.loc[d["sample"] & tr, "y"], d.loc[d["sample"] & tr, ["dev_24h"]])
    b_iv, n_iv = iv_beta(d, tr)
    out = {
        "beta_train": beta_ols.params[1], "beta_se": beta_ols.bse[1],
        "hl_beta": half_life(beta_ols.params[1]),
        "beta_iv": b_iv, "hl_iv": half_life(b_iv),
        "lp": local_projection(d),
        "lp_train": local_projection(d, sub=tr),
        "legs_raw": {"全样本": leg_shares(d, d["sample"])},
        "legs": {"全样本": leg_shares_rel(sym, d, d["sample"]), "训练": leg_shares_rel(sym, d, tr),
                 "测试": leg_shares_rel(sym, d, d["split"] == "test")},
        "legs_iv": {"全样本": leg_shares_rel(sym, d, d["sample"], iv=True)},
        "premium_state": premium_leg_by_state(d, sym),
        "event": settlement_event(sym),
        "event_placebo": settlement_event(sym, shift_h=2),
        "prem_index_legs": prem_index_legs(sym, d),
        "pdist": premium_distribution(sym),
    }
    if sym == "XAUUSDT":
        pre58 = d.index < C.XAU_INDEX_CLOSED_RULE
        out["legs"]["05-08 前"] = leg_shares_rel(sym, d, pre58)
        out["legs"]["05-08 后"] = leg_shares_rel(sym, d, ~pre58)
        out["legs_iv"]["05-08 前"] = leg_shares_rel(sym, d, pre58, iv=True)
        out["legs_iv"]["05-08 后"] = leg_shares_rel(sym, d, ~pre58, iv=True)
        out["vecm5_pre58"] = vecm(sym, 5, mask_fn=lambda i: i < C.XAU_INDEX_CLOSED_RULE)
        out["vecm5_post58"] = vecm(sym, 5, mask_fn=lambda i: (i >= C.XAU_INDEX_CLOSED_RULE) & (i < C.XAU_INDEX_USDT_CONV))
        out["vecm5_post716"] = vecm(sym, 5, mask_fn=lambda i: i >= C.XAU_INDEX_USDT_CONV)
    if model_res is not None:
        out["model_response"] = model_response(model_res)
    return out


def model_response(res) -> dict:
    """模型隐含的回归比例：两类冲击下预测的 4h 变化 / 冲击幅度（Ridge 系数，原始单位）。
    "新偏离"：基差刚跳 1bp（各窗口偏离都 +1，噪声项 +1，溢价 +1）；
    "旧偏离"：偏离已持续 4h 以上（dev_24h、dev_72h、溢价偏离 +1，短窗口偏离与噪声项 ≈0）。"""
    c = res["coef_raw"]
    g = lambda k: float(c.get(k, 0.0))
    fresh = sum(g(k) for k in ("dev_1h", "dev_4h", "dev_24h", "dev_72h", "noise_15m", "prem_leg_dev"))
    old = g("dev_24h") + g("dev_72h") + g("prem_leg_dev")
    return {"fresh": fresh, "old": old, "hl_fresh": half_life(fresh), "hl_old": half_life(old)}
