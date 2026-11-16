# -*- coding: utf-8 -*-
"""HistData 时间戳 → UTC。

规则（由 evidence() 的两条独立证据得出，不是照搬文档）：
  * 标签是这一分钟的 **起点**；
  * 2026-03-27（周五）收盘前，本地时间 = UTC−5；2026-03-29（周日）开盘起，本地时间 = UTC−4。
    切换不在美国夏令时的 03-08，而在欧洲夏令时的那个周末；切换落在周末休市内，具体时刻不可识别，
    取休市内任一时刻都等价，这里取本地 2026-03-28 12:00。
  * 因此 UTC 结束时刻 = 本地标签 + 偏移 + 1 分钟。
"""
import numpy as np
import pandas as pd

SWITCH_LOCAL = pd.Timestamp("2026-03-28 12:00")
OFF_BEFORE = pd.Timedelta(hours=5)
OFF_AFTER = pd.Timedelta(hours=4)
MINUTE = pd.Timedelta(minutes=1)


def local_to_utc_end(local: pd.DatetimeIndex) -> pd.DatetimeIndex:
    local = pd.DatetimeIndex(local)
    off = np.where(local < SWITCH_LOCAL, OFF_BEFORE.value, OFF_AFTER.value)
    return local + pd.to_timedelta(off) + MINUTE


def utc_end_to_local_label(t: pd.Timestamp) -> pd.Timestamp:
    """核对用：UTC 结束时刻对应的 HistData 本地标签。"""
    start = t - MINUTE
    for off in (OFF_BEFORE, OFF_AFTER):
        lab = start - off
        if (lab < SWITCH_LOCAL) == (off == OFF_BEFORE):
            return lab
    raise ValueError(t)


def _ret(s: pd.Series) -> pd.Series:
    """只在相邻分钟都存在时计算对数收益（不跨缺口）。"""
    r = np.log(s).diff()
    gap = s.index.to_series().diff() != MINUTE
    r[gap.values] = np.nan
    return r


def offset_scan(hd_close_local: pd.Series, perp_close_end: pd.Series,
                offsets=range(230, 311), min_obs=300, keep=(240, 300)) -> pd.DataFrame:
    """逐个本地交易日，在候选偏移（分钟）下算 HistData 与 XAU 永续一分钟收益的相关系数。
    只统计同时有 ≥min_obs 个可配对分钟的交易日（区间首尾的残缺交易日不计）。

    hd_close_local：以本地 **标签** 为索引；perp_close_end：以 UTC 结束时刻为索引。
    偏移 k 的含义：UTC 结束时刻 = 本地标签 + k 分钟 + 1 分钟。
    """
    rh = _ret(hd_close_local)
    rp = _ret(perp_close_end)
    day = (rh.index + pd.Timedelta(hours=6)).floor("D")  # 本地 18:00 开盘的时段记到下一个交易日
    rows = []
    for d, idx in rh.groupby(day).groups.items():
        x = rh.loc[idx].dropna()
        if len(x) < min_obs:
            continue
        cs = {}
        for k in offsets:
            y = rp.reindex(x.index + pd.Timedelta(minutes=k) + MINUTE).values
            m = ~np.isnan(y)
            if m.sum() >= min_obs:
                cs[k] = np.corrcoef(x.values[m], y[m])[0, 1]
        if not cs:
            continue
        s = pd.Series(cs).sort_values(ascending=False)
        rows.append({"day": d, "best": int(s.index[0]), "corr": s.iloc[0],
                     **{f"corr_{k}": cs.get(k, np.nan) for k in keep},
                     "corr_m1": cs.get(int(s.index[0]) - 1, np.nan),
                     "corr_p1": cs.get(int(s.index[0]) + 1, np.nan),
                     "second": s.iloc[1], "n": len(x)})
    return pd.DataFrame(rows).set_index("day")


def daily_break_schedule(hd_local_index: pd.DatetimeIndex) -> pd.DataFrame:
    """工作日内 ≥45 分钟的缺口（本地时间）——日休市作息，是与偏移扫描独立的第二条证据。"""
    t = pd.Series(hd_local_index).sort_values().reset_index(drop=True)
    gap = t.diff()
    m = (gap >= pd.Timedelta(minutes=45)) & (gap < pd.Timedelta(hours=6))
    out = pd.DataFrame({"last_before": t.shift()[m].values, "first_after": t[m].values})
    out["break_start_local"] = out["last_before"] + MINUTE
    out["hours"] = (out["first_after"] - out["break_start_local"]) / pd.Timedelta(hours=1)
    # 只保留工作日（周一至周五）的日休；排除周末休市窗口内的数据缺口（如欧洲夏令时切换日 03-29 的伪缺口）
    out = out[out["break_start_local"].dt.dayofweek < 5].reset_index(drop=True)
    return out
