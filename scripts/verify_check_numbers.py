# -*- coding: utf-8 -*-
"""独立复核"核对数字"：只用标准库直接读原始 zip 文本，不经过 src/aqr 的任何代码。

复核三件事：
  1. 两个时刻四个收盘价与两个基差（时钟规则手写在这里：本地标签 + 5h/4h + 1 分钟）；
  2. XAUUSDT 2026-02-02 00:00 这一期资金费：240 根溢价 K 线按样本数加权的 OHLC4 平均，I=0，N=4；
  3. 与 report/generated/ 里管线产出的数字逐一比对。
用法：python scripts/verify_check_numbers.py
"""
import io
import math
import os
import re
import zipfile
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("AQR_DATA_DIR") or os.path.join(ROOT, "data")
FUT = os.path.join(DATA, "币安Binance", "永续合约市场_USDM")


def lines(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        return z.read(name).decode().splitlines()


def kline_close(path, t_end, us=False):
    start = int((t_end - timedelta(minutes=1)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    key = str(start * 1000 if us else start)
    for ln in lines(path):
        if ln.startswith(key + ","):
            return ln.split(",")[4]
    raise KeyError(t_end)


def histdata_close(t_end):
    off = 5 if t_end < datetime(2026, 3, 28, 12) else 4
    lab = t_end - timedelta(minutes=1) - timedelta(hours=off)
    f = os.path.join(DATA, "HistData真实金价", "XAUUSD一分钟_GenericASCII", f"HISTDATA_COM_ASCII_XAUUSD_M1_{lab:%Y%m}.zip")
    rows = [ln.split(";") for ln in lines(f) if ln.startswith(lab.strftime("%Y%m%d %H%M%S"))]
    return rows[-1][4], lab


def main():
    gen = open(os.path.join(ROOT, "report", "generated", "tab_check_prices.tex"), encoding="utf-8").read()
    ok = True
    for t in (datetime(2026, 1, 14, 15, 0), datetime(2026, 8, 19, 14, 0)):
        m = f"{t:%Y-%m}"
        xp = kline_close(os.path.join(FUT, "一分钟K线_klines", "XAUUSDT_黄金永续", f"XAUUSDT-1m-{m}.zip"), t)
        pp = kline_close(os.path.join(FUT, "一分钟K线_klines", "PAXGUSDT_黄金代币永续", f"PAXGUSDT-1m-{m}.zip"), t)
        ps = kline_close(os.path.join(DATA, "币安Binance", "现货市场", "一分钟K线_klines", "PAXGUSDT_PAXG代币现货",
                                      f"PAXGUSDT-1m-{m}.zip"), t, us=True)
        hd, lab = histdata_close(t)
        bx, bp = 1e4 * math.log(float(xp) / float(hd)), 1e4 * math.log(float(pp) / float(ps))
        row = f"{t:%m-%d %H:%M}"
        line = next(l for l in gen.splitlines() if l.startswith(row))
        nums = re.findall(r"texttt\{([^}]*)\}|(−?\d+\.\d{3})(?=\s*(?:&|\\\\))", line)
        print(f"{t}  本地标签 {lab:%Y%m%d %H%M%S}  XAU永续 {xp}  金价 {hd}  XAU基差 {bx:.3f}  "
              f"PAXG永续 {pp}  PAXG代币 {ps}  PAXG基差 {bp:.3f}")
        for v in (xp, hd, pp, ps, f"{bx:.3f}".replace("-", "−"), f"{bp:.3f}".replace("-", "−")):
            if v not in line:
                ok = False
                print("   ✗ 与管线不一致：", v)
    # 资金费 XAUUSDT 2026-02-02 00:00
    t = datetime(2026, 2, 2, 0, 0)
    pre = {}
    for m in ("2026-01", "2026-02"):
        for ln in lines(os.path.join(FUT, "一分钟溢价指数_premiumIndexKlines", "XAUUSDT_黄金永续", f"XAUUSDT-1m-{m}.zip"))[1:]:
            c = ln.split(",")
            pre[int(c[0])] = (float(c[1]), float(c[2]), float(c[3]), float(c[4]), int(c[8]))
    t0 = int((t - timedelta(hours=4)).replace(tzinfo=timezone.utc).timestamp() * 1000)
    seq, num, den = 0, 0.0, 0.0
    for k in range(240):
        o, h, l, c, n = pre[t0 + k * 60_000]
        w = n * seq + n * (n + 1) / 2
        seq += n
        num += w * (o + h + l + c) / 4
        den += w
    P = num / den
    F = (P + max(-5e-4, min(5e-4, 0.0 - P))) / 2
    off = next(l for l in lines(os.path.join(FUT, "资金费历史_fundingRate", "XAUUSDT_黄金永续",
                                             "XAUUSDT-fundingRate-2026-02.zip"))[1:]
               if datetime.fromtimestamp(int(l.split(",")[0]) / 1000, timezone.utc).replace(tzinfo=None) == t + timedelta(milliseconds=int(l.split(",")[0]) % 1000))
    print(f"XAUUSDT {t}  P̄={P * 1e4:.4f}bp  我算 {F * 1e4:.3f}bp  官方原值 {off.split(',')[2]}  差 {(F - float(off.split(',')[2])) * 1e4:+.3f}bp")
    fz = open(os.path.join(ROOT, "report", "generated", "tab_check_funding.tex"), encoding="utf-8").read()
    if f"{F * 1e4:.3f}".replace("-", "−") not in fz:
        ok = False
        print("   ✗ 资金费与管线不一致")
    print("全部一致 ✓" if ok else "存在不一致 ✗")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
