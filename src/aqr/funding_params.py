# -*- coding: utf-8 -*-
"""资金费参数时间表：参数随时间变化，分段生效，绝不用今天的参数回填历史。

生效语义（由数据决定）：一段参数作用于结算时刻 **严格晚于** 其生效时刻的结算期，
即"新参数从生效时刻开始的那一期起作用"。证据：XAU 利息项"自 2026-01-23 08:00 起为 0"，
但 08:00 结算的那一期官方值仍是 0.5000 bp = 0.01%/2（旧利息项）；若按"结算时刻 ≥ 08:00"
套用新参数，这一期会错 0.5 bp。

结算间隔 N 不在本表里：各期 N 以资金费数据中的 funding_interval_hours 字段为准（题面要求）。
cap/floor 为 None 表示样本期内从未触及、因而不可识别；复现时不施加。
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

BAND = 5e-4          # clamp 区间 ±0.05%（Binance 公式常数，复现中由死区拐点验证）
I_DEFAULT = 1e-4     # 默认利息项 0.01% / 8h


@dataclass(frozen=True)
class Seg:
    sym: str
    effective: pd.Timestamp
    I: float
    cap: Optional[float]
    floor: Optional[float]
    source: str
    evidence: str


SEGMENTS = [
    Seg("XAUUSDT", pd.Timestamp("2025-12-11"), I_DEFAULT, None, None, "数据",
        "死区内 F·8/N 恒等于 0.01%；01-25 20:00 官方 44.1736bp 与无截断公式值一致→上限若存在则≥0.4417%，"
        "与公告所称原上下限 ±0.375% 矛盾，记为未识别"),
    Seg("XAUUSDT", pd.Timestamp("2026-01-23 08:00"), 0.0, None, None, "题面+数据",
        "08:00 结算期官方 0.5000bp=旧 I/2；12:00 起死区内官方值恒为 0"),
    Seg("XAUUSDT", pd.Timestamp("2026-01-30 12:15"), 0.0, 5e-4, -5e-4, "公告+数据",
        "公告：UTC 12:15 起 8h 结算、上下限 ±0.05%；16:00 期字段 N=8，公式值 9.92bp，官方恰为 5.0000bp"),
    Seg("XAUUSDT", pd.Timestamp("2026-01-30 16:00"), 0.0, 5e-3, None, "数据",
        "20:00 起 N=4，官方 23.8248bp>0.05%→上下限已恢复；02-28 12:00 公式 52.08bp→官方 50.0000bp，"
        "上限=0.5%；下限未触及（最低官方 −33.5043bp 与无截断值一致）"),
    Seg("PAXGUSDT", pd.Timestamp("2025-01-01"), I_DEFAULT, None, None, "数据",
        "死区内官方值恒为 0.5000bp=0.01%/2；官方值全在 [−6.2491, 0.5000]bp，均与无截断值一致"),
]


def params_for(sym: str, settle: pd.Timestamp) -> Seg:
    segs = [s for s in SEGMENTS if s.sym == sym and s.effective < settle]
    if not segs:
        raise KeyError((sym, settle))
    return max(segs, key=lambda s: s.effective)


def interest_at(sym: str, index: pd.DatetimeIndex):
    """向量化：时刻 t 正在累计的那一期所适用的利息项（生效时刻严格早于 t 的最后一段）。"""
    import numpy as np
    segs = sorted((s for s in SEGMENTS if s.sym == sym), key=lambda s: s.effective)
    eff = np.array([s.effective.to_datetime64() for s in segs])
    pos = np.searchsorted(eff, pd.DatetimeIndex(index).to_numpy(), side="left") - 1
    return np.array([s.I for s in segs])[np.maximum(pos, 0)]


def table() -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in SEGMENTS])
