# -*- coding: utf-8 -*-
"""原始 zip 读取层。

约定：返回的 Binance 时间序列一律以 **UTC 分钟结束时刻** 为索引（open_time + 1 分钟）；
资金费以 **结算时刻**（calc_time 向下取整到整点）为索引；
HistData 保留文件中的本地时间标签，换算到 UTC 由 clock.py 负责（换算规则需要数据证据）。
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
              "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
MINUTE = pd.Timedelta(minutes=1)


def _read_zip_csv(path: Path) -> bytes:
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        return z.read(name)


def read_kline_zip(path: Path) -> pd.DataFrame:
    """读一个 Binance 1m K 线 zip（期货有表头、现货无表头；现货时间戳为微秒）。"""
    raw = _read_zip_csv(path)
    has_header = not raw[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    df.columns = KLINE_COLS
    ot = df["open_time"].astype("int64")
    unit = "us" if ot.iloc[0] > 10 ** 14 else "ms"
    df.index = pd.to_datetime(ot, unit=unit) + MINUTE
    df.index.name = "t"
    df["ts_unit"] = unit
    return df.drop(columns=["open_time", "close_time", "ignore"])


def _kline_dir(kind: str, sym: str) -> Path:
    if kind == "spot":
        return C.SPOT_DIR
    typ = {"perp": "klines", "index": "indexPriceKlines", "premium": "premiumIndexKlines"}[kind]
    return C.BN_FUT / C.TYP_DIR[typ] / C.SYM_DIR[sym]


def load_klines(kind: str, sym: str = "PAXGUSDT") -> pd.DataFrame:
    """kind ∈ {perp, index, premium, spot}。月度归档 + 日度补丁（若有），按时间排序。"""
    d = _kline_dir(kind, sym)
    files = [d / f"{sym}-1m-{m}.zip" for m in C.MONTHS]
    files += sorted((d / C.DAILY_DIR).glob("*.zip")) if (d / C.DAILY_DIR).exists() else []
    parts = []
    for f in files:
        x = read_kline_zip(f)
        x["source"] = "daily" if f.parent.name == C.DAILY_DIR else "monthly"
        parts.append(x)
    df = pd.concat(parts).sort_index()
    dup = df.index.duplicated(keep="first")
    if dup.any():  # 日度补丁与月度重叠时以月度为准（实测无重叠）
        df = df[~dup]
    return df


def load_funding(sym: str) -> pd.DataFrame:
    d = C.BN_FUT / C.TYP_DIR["fundingRate"] / C.SYM_DIR[sym]
    parts = [pd.read_csv(io.BytesIO(_read_zip_csv(d / f"{sym}-fundingRate-{m}.zip"))) for m in C.MONTHS]
    df = pd.concat(parts, ignore_index=True)
    df["calc_raw"] = pd.to_datetime(df["calc_time"], unit="ms")
    df["t"] = df["calc_raw"].dt.floor("h")
    df = df.rename(columns={"funding_interval_hours": "N", "last_funding_rate": "rate"})
    return df.set_index("t")[["N", "rate", "calc_raw", "calc_time"]].sort_index()


def load_histdata_raw() -> pd.DataFrame:
    """HistData 原始行（保留文件顺序 row 与来源月份）。本地时间标签 = 分钟起点（见 clock.py 证据）。"""
    parts = []
    for m in range(1, 10):
        f = C.HIST_DIR / f"HISTDATA_COM_ASCII_XAUUSD_M1_2026{m:02d}.zip"
        df = pd.read_csv(io.BytesIO(_read_zip_csv(f)), sep=";", header=None,
                         names=["label", "open", "high", "low", "close", "volume"])
        df["local"] = pd.to_datetime(df["label"], format="%Y%m%d %H%M%S")
        df["file_month"] = m
        df["row"] = np.arange(len(df))
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def merge_histdata(raw: pd.DataFrame) -> pd.DataFrame:
    """题面规则：同一分钟多行按文件顺序合并——开盘取第一行、最高/最低取极值、收盘取最后一行。"""
    raw = raw.sort_values(["file_month", "row"], kind="stable")
    g = raw.groupby("local", sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "n_rows": g.size()})
    return out


def histdata_status() -> pd.DataFrame:
    """HistData 官方缺口报告（本地时间），用于和我们自己扫描到的缺口互相核对。"""
    rows = []
    for m in range(1, 10):
        f = C.HIST_DIR / f"HISTDATA_COM_ASCII_XAUUSD_M1_2026{m:02d}_status.txt"
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Gap of"):
                p = line.split()
                rows.append((int(p[2].rstrip("s")), pd.to_datetime(p[6], format="%Y%m%d%H%M%S"),
                             pd.to_datetime(p[8].rstrip("."), format="%Y%m%d%H%M%S")))
    return pd.DataFrame(rows, columns=["gap_s", "from_local", "to_local"])


# ------------------------------------------------------------------ 核对数字：原样取原始文本
def raw_kline_line(kind: str, sym: str, t_end: pd.Timestamp) -> list:
    """返回结束于 t_end 的那根 1m K 线在原始 CSV 里的字段（字符串，不做任何数值转换）。"""
    start = t_end - MINUTE
    d = _kline_dir(kind, sym)
    f = d / f"{sym}-1m-{start:%Y-%m}.zip"
    key_ms, key_us = str(int(start.value // 10 ** 6)), str(int(start.value // 10 ** 3))
    for line in _read_zip_csv(f).decode().splitlines():
        head = line.split(",", 1)[0]
        if head in (key_ms, key_us):
            return line.split(",")
    raise KeyError((kind, sym, t_end))


def raw_histdata_lines(local_label: pd.Timestamp) -> list:
    f = C.HIST_DIR / f"HISTDATA_COM_ASCII_XAUUSD_M1_{local_label:%Y%m}.zip"
    key = local_label.strftime("%Y%m%d %H%M%S")
    return [l.split(";") for l in _read_zip_csv(f).decode().splitlines() if l.startswith(key)]


def raw_funding_line(sym: str, t: pd.Timestamp) -> list:
    d = C.BN_FUT / C.TYP_DIR["fundingRate"] / C.SYM_DIR[sym]
    for line in _read_zip_csv(d / f"{sym}-fundingRate-{t:%Y-%m}.zip").decode().splitlines()[1:]:
        ct = pd.to_datetime(int(line.split(",")[0]), unit="ms").floor("h")
        if ct == t:
            return line.split(",")
    raise KeyError((sym, t))


def cached(name: str, builder, rebuild: bool = False) -> pd.DataFrame:
    """parquet 缓存：缓存只是加速，删掉 cache/ 会从原始 zip 重建。"""
    p = C.CACHE / f"{name}.parquet"
    if p.exists() and not rebuild:
        return pd.read_parquet(p)
    df = builder()
    df.to_parquet(p)
    return df
