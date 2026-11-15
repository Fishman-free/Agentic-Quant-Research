# -*- coding: utf-8 -*-
"""全局常量：路径、研究区间、切分日期。所有时刻均为 UTC、指一分钟 K 线结束的时刻。"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AQR_DATA_DIR", REPO / "data"))
CACHE = REPO / "cache"
OUT = REPO / "outputs"
FIG = REPO / "report" / "figures"
GEN = REPO / "report" / "generated"

MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]
START = pd.Timestamp("2026-01-01 00:00")
END = pd.Timestamp("2026-10-01 00:00")
SAMPLE_START = pd.Timestamp("2026-01-02 00:00")
TEST_START = pd.Timestamp("2026-07-01 00:00")
VAL_START = pd.Timestamp("2026-06-01 00:00")  # 选型阶段：t+4h<=06-01 训练、6 月验证

# 文档给出的事件时刻
XAU_INDEX_CLOSED_RULE = pd.Timestamp("2026-05-08 21:00")  # 休市期参考指数改由永续自身盘口生成
XAU_INDEX_USDT_CONV = pd.Timestamp("2026-07-16 08:00")    # 参考指数开始做 USDT/USD 换算

H = 4  # 预测期（小时）
SYMBOLS = ("XAUUSDT", "PAXGUSDT")
SEED = 20260701

# 原始数据目录（与 scripts/download_datasets.py 一致）
BN = DATA / "币安Binance"
BN_FUT = BN / "永续合约市场_USDM"
SYM_DIR = {"XAUUSDT": "XAUUSDT_黄金永续", "PAXGUSDT": "PAXGUSDT_黄金代币永续"}
TYP_DIR = {"klines": "一分钟K线_klines",
           "indexPriceKlines": "一分钟参考指数_indexPriceKlines",
           "premiumIndexKlines": "一分钟溢价指数_premiumIndexKlines",
           "fundingRate": "资金费历史_fundingRate"}
SPOT_DIR = BN / "现货市场" / "一分钟K线_klines" / "PAXGUSDT_PAXG代币现货"
HIST_DIR = DATA / "HistData真实金价" / "XAUUSD一分钟_GenericASCII"
DAILY_DIR = "日度补丁_daily"  # 月度归档缺整天时，从 daily 归档补拉，放在同类型目录下的子目录

for _p in (CACHE, OUT, FIG, GEN):
    _p.mkdir(parents=True, exist_ok=True)
