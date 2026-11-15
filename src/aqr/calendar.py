# -*- coding: utf-8 -*-
"""现货黄金（外汇市场 XAU/USD）交易日历——只用事先公开可知的规则，不看 HistData 实际缺口。

用途：做特征（距收盘/距开盘、预测窗口内的休市分钟数）时必须事先可知，否则会把临时数据缺失漏进特征。
规则（美东时间 ET）：周五 17:00 收盘 → 周日 18:00 开盘；周一至周四 17:00–18:00 日休；
美国假日按公开日历（元旦、MLK/总统日/阵亡将士/劳工节 14:30–18:00 提前休市，耶稣受难日全天休市，
六月节与独立日观察日 13:00 提前收盘）。
validate() 用 HistData 的实际缺口回头核对日历——只做核对，不进入特征。
"""
import numpy as np
import pandas as pd

from . import config as C

US_DST_START_UTC = pd.Timestamp("2026-03-08 07:00")  # 美东 2026-03-08 02:00 EST
US_DST_END_UTC = pd.Timestamp("2026-11-01 06:00")
MINUTE = pd.Timedelta(minutes=1)


def et_offset(t_utc):
    t = pd.DatetimeIndex(t_utc)
    dst = (t >= US_DST_START_UTC) & (t < US_DST_END_UTC)
    return pd.to_timedelta(np.where(dst, -4, -5), unit="h")


def et_to_utc(s: str) -> pd.Timestamp:
    t = pd.Timestamp(s)
    for off in (-5, -4):
        u = t - pd.Timedelta(hours=off)
        if et_offset([u])[0] == pd.Timedelta(hours=off):
            return u
    raise ValueError(s)


# (休市开始 ET, 重新开盘 ET)
HOLIDAYS_ET = [
    ("2025-12-31 17:00", "2026-01-01 18:00"),  # 元旦
    ("2026-01-19 14:30", "2026-01-19 18:00"),  # MLK
    ("2026-02-16 14:30", "2026-02-16 18:00"),  # 总统日
    ("2026-04-02 17:00", "2026-04-05 18:00"),  # 耶稣受难日（连周末）
    ("2026-05-25 14:30", "2026-05-25 18:00"),  # 阵亡将士纪念日
    ("2026-06-19 13:00", "2026-06-21 18:00"),  # 六月节，提前收盘连周末
    ("2026-07-03 13:00", "2026-07-05 18:00"),  # 独立日观察日，提前收盘连周末
    ("2026-09-07 14:30", "2026-09-07 18:00"),  # 劳工节
]


def closures_utc(start=C.START - pd.Timedelta(days=7), end=C.END + pd.Timedelta(days=7)) -> pd.DataFrame:
    rows = []
    for d in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        ds = d.strftime("%Y-%m-%d")
        if d.dayofweek in (0, 1, 2, 3):
            rows.append((et_to_utc(f"{ds} 17:00"), et_to_utc(f"{ds} 18:00"), "daily"))
        elif d.dayofweek == 4:
            sun = (d + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
            rows.append((et_to_utc(f"{ds} 17:00"), et_to_utc(f"{sun} 18:00"), "weekend"))
    for a, b in HOLIDAYS_ET:
        rows.append((et_to_utc(a), et_to_utc(b), "holiday"))
    df = pd.DataFrame(rows, columns=["close", "open", "kind"]).sort_values("close")
    # 合并重叠区间（假日与周末相连时）
    merged = []
    for r in df.itertuples(index=False):
        if merged and r.close <= merged[-1][1]:
            a, b, k = merged[-1]
            merged[-1] = (a, max(b, r.open), "holiday" if "holiday" in (k, r.kind) else k)
        else:
            merged.append((r.close, r.open, r.kind))
    out = pd.DataFrame(merged, columns=["close", "open", "kind"])
    out["hours"] = (out["open"] - out["close"]) / pd.Timedelta(hours=1)
    out.loc[(out["kind"] == "daily") & (out["hours"] > 1.5), "kind"] = "holiday"
    out.loc[(out["kind"] == "weekend") & (out["hours"] > 49.5), "kind"] = "holiday"
    return out.reset_index(drop=True)


def open_mask(index: pd.DatetimeIndex, cal: pd.DataFrame = None) -> np.ndarray:
    """分钟（以结束时刻标记）是否处在开市时段：分钟起点 ∈ [open, 下一个 close)。"""
    cal = closures_utc() if cal is None else cal
    start = index - MINUTE
    closed = np.zeros(len(index), dtype=bool)
    for r in cal.itertuples(index=False):
        closed |= (start >= r.close) & (start < r.open)
    return ~closed


def schedule_features(times: pd.DatetimeIndex, horizon=pd.Timedelta(hours=C.H)) -> pd.DataFrame:
    """距下一次收盘、距上一次开盘（小时，截断）以及预测窗口 (t, t+4h] 内的休市分钟数。"""
    cal = closures_utc()
    closes, opens = cal["close"].to_numpy(), cal["open"].to_numpy()
    t = times.to_numpy()
    i_next = np.searchsorted(closes, t, side="left")
    h_to_close = (closes[np.minimum(i_next, len(closes) - 1)] - t) / np.timedelta64(1, "h")
    i_prev = np.searchsorted(opens, t, side="right") - 1
    h_since_open = (t - opens[np.maximum(i_prev, 0)]) / np.timedelta64(1, "h")
    end = t + horizon.to_timedelta64()
    closed_min = np.zeros(len(t))
    for a, b in zip(closes, opens):
        ov = (np.minimum(end, b) - np.maximum(t, a)) / np.timedelta64(1, "m")
        closed_min += np.clip(ov, 0, None)
    return pd.DataFrame({"h_to_close": np.clip(h_to_close, 0, 24),
                         "h_since_open": np.clip(h_since_open, 0, 24),
                         "closed_frac_window": closed_min / (horizon / pd.Timedelta(minutes=1))},
                        index=times)


def validate(hd_index_utc: pd.DatetimeIndex) -> dict:
    """日历 vs HistData 实际：日历开市但没数据的分钟（数据缺失）、日历休市却有数据的分钟（日历错）。"""
    grid = pd.date_range(C.START + MINUTE, C.END, freq="1min")
    om = pd.Series(open_mask(grid), index=grid)
    has = pd.Series(grid.isin(hd_index_utc), index=grid)
    return {"open_no_data": int((om & ~has).sum()), "closed_but_data": int((~om & has).sum()),
            "open_minutes": int(om.sum()), "data_minutes": int(has.sum()),
            "closed_but_data_list": has[~om & has].index}
