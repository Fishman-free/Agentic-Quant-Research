# -*- coding: utf-8 -*-
"""一分钟面板：把四个价格、参考指数、溢价指数放到同一个 UTC 时钟（分钟结束时刻）上。

基差 = 1e4·ln(永续/现货) 拆成两腿：
  溢价腿 prem_leg = 1e4·ln(永续/参考指数)   —— 资金费直接作用的那一段
  指数腿 index_leg = 1e4·ln(参考指数/现货)   —— 参考指数与基差所用"现货"之间的错位
"""
import numpy as np
import pandas as pd

from . import clock, config as C, io_raw as R

GRID = pd.date_range(C.START + pd.Timedelta(minutes=1), C.END, freq="1min", name="t")


def histdata_utc(rebuild=False) -> pd.DataFrame:
    def build():
        hd = R.merge_histdata(R.load_histdata_raw())
        hd["local"] = hd.index
        hd.index = clock.local_to_utc_end(hd.index)
        hd.index.name = "t"
        return hd
    return R.cached("histdata_utc", build, rebuild)


def _bn(kind, sym, rebuild):
    return R.cached(f"{kind}_{sym}", lambda: R.load_klines(kind, sym), rebuild)


def minute_panel(sym: str, rebuild=False) -> pd.DataFrame:
    def build():
        perp = _bn("perp", sym, rebuild)
        idx = _bn("index", sym, rebuild)
        pre = _bn("premium", sym, rebuild)
        if sym == "XAUUSDT":
            spot = histdata_utc(rebuild)["close"]
        else:
            spot = _bn("spot", "PAXGUSDT", rebuild)["close"]
        df = pd.DataFrame(index=GRID)
        df["perp"] = perp["close"]
        df["spot"] = spot
        df["index"] = idx["close"]
        for c in ("open", "high", "low", "close"):
            df[f"prem_{c}"] = pre[c]
        df["perp_vol"] = perp["volume"]
        df["perp_qvol"] = perp["quote_volume"]
        df["perp_trades"] = perp["count"]
        df["perp_taker_q"] = perp["taker_buy_quote_volume"]
        df["basis"] = 1e4 * np.log(df["perp"] / df["spot"])
        df["prem_leg"] = 1e4 * np.log(df["perp"] / df["index"])
        df["index_leg"] = 1e4 * np.log(df["index"] / df["spot"])
        return df
    return R.cached(f"panel_{sym}", build, rebuild)


def funding(sym: str, rebuild=False) -> pd.DataFrame:
    return R.cached(f"funding_{sym}", lambda: R.load_funding(sym), rebuild)
