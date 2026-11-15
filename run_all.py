# -*- coding: utf-8 -*-
"""一条命令：从已下载的原始 zip 跑出报告里的全部数字、表格、图，并编译 PDF。

用法：
  python run_all.py            # 全流程约 6 分钟（其中神经网络对照约 5 分钟）
  python run_all.py --rebuild  # 删掉缓存、从原始 zip 重建（验证"缓存只是加速"）
  python run_all.py --no-pdf   # 只算数字与图，不调用 xelatex
  python run_all.py --no-mlp   # 跳过神经网络对照，沿用 outputs/summary.json 的上次结果（约 1 分钟）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aqr import (calendar, closure, clock, config as C, dataqa, dynamics, evaluate, figures, funding,  # noqa: E402
                 io_raw, mlp, models, panel, report_numbers)


def step(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-mlp", action="store_true", help="跳过神经网络对照，沿用 outputs/summary.json 的上次结果")
    a = ap.parse_args()
    t0 = time.time()
    if a.rebuild and C.CACHE.exists():
        shutil.rmtree(C.CACHE)
        C.CACHE.mkdir()
    for sym in C.SYMBOLS:
        panel.minute_panel(sym)
        panel.funding(sym)
    step("1/7 数据层：四个价格 + 指数 + 溢价 + 资金费已对齐到 UTC 分钟结束时刻", t0)

    res = {}
    hd = io_raw.merge_histdata(io_raw.load_histdata_raw())
    res["clock_scan"] = clock.offset_scan(hd["close"], io_raw.load_klines("perp", "XAUUSDT")["close"])
    res["clock_breaks"] = clock.daily_break_schedule(hd.index)
    res["calendar_validate"] = calendar.validate(panel.histdata_utc().index)
    step("2/7 时钟证据（逐日偏移扫描 + 日休作息）", t0)

    res["funding"] = funding.run_all()
    step("3/7 资金费逐期复现", t0)

    res["models"], res["eval"] = {}, {}
    for sym in C.SYMBOLS:
        r = models.fit_predict(sym)
        res["models"][sym] = r
        res["eval"][sym] = {"table": evaluate.rmse_table(r["pred"]), "sig": evaluate.significance(r["pred"])}
    if a.no_mlp:
        # 神经网络要跑约 5 分钟；调其他部分时沿用上次写进 summary.json 的结果，PDF 不变
        prev = json.loads((C.OUT / "summary.json").read_text(encoding="utf-8"))
        try:
            res["mlp"] = {sym: prev[sym]["mlp"] for sym in C.SYMBOLS}
        except KeyError:
            sys.exit("--no-mlp 需要含 mlp 字段的 outputs/summary.json：先完整跑一次 run_all.py")
        step("4/7 预测模型与检验（神经网络对照沿用上次结果）", t0)
    else:
        res["mlp"] = mlp.run_all(res["models"])
        step("4/7 预测模型与检验（含神经网络对照）", t0)

    res["dyn"] = {sym: dynamics.run(sym, res["models"][sym]) for sym in C.SYMBOLS}
    res["closure"] = closure.run()
    step("5/7 回归动态 + 休市深挖", t0)

    res["dataqa"] = dataqa.run_all()
    figures.run_all(res)
    macros = report_numbers.write_all(res)
    step(f"6/7 数据问题、图、{len(macros)} 个数字宏与表格", t0)

    for sym in C.SYMBOLS:
        print(f"\n== {sym}")
        print(res["eval"][sym]["table"].round(3).to_string())
        s = res["funding"][sym]["summary"]
        print(f"资金费复现：{s['n']} 期，最大误差 {s['max_abs']:.3f}bp，>0.1bp {s['n_gt_0.1']} 期")

    if not a.no_pdf:
        rep = C.REPO / "report"
        # 固定 PDF 内的时间戳与文件 ID，重跑后 main.pdf 逐字节不变
        env = {**os.environ, "SOURCE_DATE_EPOCH": "1789430400", "FORCE_SOURCE_DATE": "1"}
        for _ in range(2):  # 两遍以解析交叉引用
            r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                               cwd=rep, env=env, capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:])
            sys.exit("xelatex 失败，见 report/main.log")
        step("7/7 report/main.pdf 已编译", t0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 重定向到文件时默认是 GBK
    main()
