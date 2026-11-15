# -*- coding: utf-8 -*-
"""整点样本、目标、基线与特征。

样本：t 取每个整点；t 与 t+4h 两个时刻，永续与对应现货都必须有该分钟 K 线（不填充、不插值）。
所有特征只用 t 及以前的数据：一律是向后看的滚动窗口 (t−w, t]，由 tests 里的"截断不变性"测试保证。

设计原则（在看测试集之前定下）：
  题面已提前告知 07-16 起 XAU 基差含 USDT/USD 偏离——训练期从未出现这种水平漂移，
  所以特征只用"相对滚动均值的偏离"类量，不用基差、指数腿的绝对水平。
"""
import numpy as np
import pandas as pd

from . import calendar as K, config as C, funding_params as FP, panel

HOUR = pd.Timedelta(hours=1)

# 特征分组：报告里逐组说明为什么放进去，并在验证集上做逐组消融
GROUPS = {
    "偏离": ["dev_1h", "dev_4h", "dev_24h", "dev_72h"],
    "噪声": ["noise_15m", "trade_vs_book"],
    "两腿": ["prem_leg_dev", "index_leg_dev"],
    "资金费": ["F_hat", "F_last_off", "tts", "outside_dead_x_dev"],
    "动量": ["r_perp_15m", "r_perp_1h", "r_spot_15m", "r_spot_1h"],
    "波动与流动性": ["vol_basis_24h", "vol_x_dev", "taker_imb_1h", "stale_1h"],
    "休市日历": ["h_to_close", "h_since_open", "closed_frac_window", "closed_x_dev"],
}
ALL = [f for g in GROUPS.values() for f in g]


def _roll_mean(s: pd.Series, w: str) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()  # 时间窗默认 closed='right'：(t−w, t]


def minute_features(p: pd.DataFrame, sym: str) -> pd.DataFrame:
    """在分钟网格上构造特征（之后按整点抽样）。p 可以是被截断的面板（用于防前视测试）。"""
    b = p["basis"]
    f = pd.DataFrame(index=p.index)
    f["basis"] = b
    for w in ("1h", "4h", "24h", "72h"):
        f[f"mean_{w}"] = _roll_mean(b, w)
        f[f"dev_{w}"] = b - f[f"mean_{w}"]
    f["noise_15m"] = b - _roll_mean(b, "15min")
    # 成交价相对盘口：溢价指数由冲击买/卖价算出，最后成交价落在买一或卖一会让 ln(永续/指数) 偏离它
    f["trade_vs_book"] = p["prem_leg"] - 1e4 * p["prem_close"]
    f["prem_leg_dev"] = p["prem_leg"] - _roll_mean(p["prem_leg"], "24h")
    f["index_leg_dev"] = p["index_leg"] - _roll_mean(p["index_leg"], "24h")
    # 资金费：最近 4h 溢价（OHLC4）均值代入当期参数的公式 → "下一期大概会收多少"
    prem = (p["prem_open"] + p["prem_high"] + p["prem_low"] + p["prem_close"]) / 4
    P4 = _roll_mean(prem, "4h")
    I_t = FP.interest_at(sym, p.index)
    f["F_hat"] = 1e4 * (P4 + np.clip(I_t - P4, -FP.BAND, FP.BAND)) / 2
    f["outside_dead"] = (np.abs(P4 - I_t) > FP.BAND).astype(float)
    lp, ls = np.log(p["perp"]), np.log(p["spot"])
    for w, k in (("15m", 15), ("1h", 60)):
        f[f"r_perp_{w}"] = 1e4 * (lp - lp.shift(k))
        f[f"r_spot_{w}"] = 1e4 * (ls - ls.shift(k))
    f["vol_basis_24h"] = b.rolling("24h", min_periods=30).std()
    q = p["perp_qvol"].rolling("1h").sum()
    f["taker_imb_1h"] = p["perp_taker_q"].rolling("1h").sum() / q.replace(0, np.nan) - 0.5
    f["stale_1h"] = (p["perp_vol"] == 0).astype(float).rolling("1h").mean()
    return f


def official_last(sym: str, times: pd.DatetimeIndex) -> pd.Series:
    """严格早于 t 的最近一期官方资金费（bp）：t 正好是结算时刻时，这一期要晚几毫秒才公布，不能用。"""
    fr = panel.funding(sym)
    s = pd.Series(fr["rate"].to_numpy() * 1e4, index=fr["calc_raw"])
    pos = np.searchsorted(s.index.to_numpy(), times.to_numpy(), side="left") - 1
    out = np.where(pos >= 0, s.to_numpy()[np.maximum(pos, 0)], np.nan)
    return pd.Series(out, index=times)


def hourly(sym: str, p: pd.DataFrame = None) -> pd.DataFrame:
    p = panel.minute_panel(sym) if p is None else p
    mf = minute_features(p, sym)
    times = pd.date_range(C.SAMPLE_START, C.END - C.H * HOUR, freq="h")
    times = times[times <= p.index.max()]
    d = mf.reindex(times).copy()
    fut = p["basis"].reindex(times + C.H * HOUR).to_numpy()
    d["basis_fut"] = fut
    d["y"] = fut - d["basis"].to_numpy()
    d["valid"] = d["basis"].notna() & d["basis_fut"].notna()
    # 资金费与日历类（均事先可知）
    d["tts"] = 4 - (times.hour % 4)  # 距下一次结算（小时）
    d["F_last_off"] = official_last(sym, times)
    d = d.join(K.schedule_features(times))
    # 状态交互项：偏离在不同状态下回归速度不同（报告里据此分状态给半衰期）
    d["outside_dead_x_dev"] = d["outside_dead"] * d["dev_24h"]
    d["closed_x_dev"] = d["closed_frac_window"] * d["dev_24h"]
    d["vol_x_dev"] = d["vol_basis_24h"] * d["dev_24h"] / 10
    d["split"] = np.where(times + C.H * HOUR <= C.TEST_START, "train",
                          np.where(times >= C.TEST_START, "test", "gap"))
    d["month"] = times.month
    return d


def samples(sym: str) -> pd.DataFrame:
    d = hourly(sym)
    d = d[d["valid"] & (d["split"] != "gap")].copy()
    d[ALL] = d[ALL].fillna(0.0)  # 缺失只出现在刚开盘等处：按"无信息"=0 处理，规则在训练期定下
    return d
