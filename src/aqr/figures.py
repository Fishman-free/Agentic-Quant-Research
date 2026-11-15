# -*- coding: utf-8 -*-
"""报告图表（matplotlib → 矢量 PDF）。

配色按"一个图里出现的系列"逐组用 dataviz 校验脚本验证过（全部 all-pairs 通过 CVD/对比度检查）：
  合约：XAU #2a78d6（蓝）· PAXG #008300（绿）
  制度：05-08 前 #eb6834（橙）· 05-08 后 #4a3aa7（紫）
  价格三线：永续 蓝 · 参考指数 橙 · 现货 紫
文字一律用墨色（主 #0b0b0b / 次 #52514e），系列色只给图形本身；网格与坐标轴保持退后。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from . import config as C

BLUE, GREEN, ORANGE, VIOLET = "#2a78d6", "#008300", "#eb6834", "#4a3aa7"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
SYM_COLOR = {"XAUUSDT": BLUE, "PAXGUSDT": GREEN}
SYM_NAME = {"XAUUSDT": "XAU 永续", "PAXGUSDT": "PAXG 永续"}
FULL_W = 6.9  # 英寸，对应 A4 正文宽


def setup():
    for f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"):
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
    fams = [f.name for f in font_manager.fontManager.ttflist]
    zh = next((n for n in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei") if n in fams), None)
    plt.rcParams.update({
        "font.family": [zh, "DejaVu Sans"] if zh else ["DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "lines.linewidth": 1.2, "lines.solid_capstyle": "round",
    })


def _save(fig, name):
    # 去掉创建时间，重跑后 PDF 逐字节不变
    fig.savefig(C.FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def _tag(ax, s):
    ax.set_title(s, loc="left", pad=4)


def _month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))


# ------------------------------------------------------------------ 图 1：时钟
def fig_clock(scan: pd.DataFrame, breaks: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for k, col, lab in ((300, ORANGE, "偏移 +5h（UTC−5）"), (240, VIOLET, "偏移 +4h（UTC−4）")):
        ax.plot(scan.index, scan[f"corr_{k}"], "o", ms=2.2, color=col, label=lab, mec="none")
    for x, txt, ls in ((pd.Timestamp("2026-03-08"), "美国夏令时 03-08", ":"), (pd.Timestamp("2026-03-28"), "实际切换 03-28/29 周末", "-")):
        ax.axvline(x, color=INK2, lw=0.7, ls=ls)
        ax.text(x, 0.52, " " + txt, rotation=90, va="center", ha="right" if ls == ":" else "left", fontsize=6.3, color=INK2)
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("与 XAU 永续分钟收益的相关系数")
    _month_axis(ax)
    ax.legend(loc="center right", ncol=1, bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
    _tag(ax, "(a) 两种偏移下的逐日相关：k 恰为 300→240 分钟")
    ax = axes[1]
    b = breaks[(breaks["hours"] > 0.9) & (breaks["hours"] < 1.2)]
    ax.plot(b["break_start_local"], b["break_start_local"].dt.hour + b["break_start_local"].dt.minute / 60,
            "s", ms=2.6, color=INK2, mec="none")
    ax.axvspan(pd.Timestamp("2026-03-08"), pd.Timestamp("2026-03-28"), color=ORANGE, alpha=0.12, lw=0)
    ax.text(pd.Timestamp("2026-03-18"), 16.45, "美国已进夏令时、\nHistData 仍是 UTC−5", ha="center", fontsize=6.2, color=INK2)
    ax.set_ylim(15.5, 18.5)
    ax.set_yticks([16, 17, 18])
    ax.set_yticklabels(["16:00", "17:00", "18:00"])
    ax.set_ylabel("日休开始时刻（HistData 本地）")
    _month_axis(ax)
    _tag(ax, "(b) 独立证据：每日 1 小时休市的本地时刻")
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig1_clock")


# ------------------------------------------------------------------ 图 2：资金费复现
def fig_funding(rep: dict):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.55]})
    for ax, sym, tag in ((axes[0], "XAUUSDT", "(a) XAU：官方 F·8/N 与 P̄"), (axes[1], "PAXGUSDT", "(b) PAXG：同上")):
        m = rep[sym]
        if sym == "XAUUSDT":
            m = m[m.index > pd.Timestamp("2026-01-23 12:00")]
        P, Fo = m["P"] * 1e4, m["official"] * 8 / m["N"] * 1e4
        I = m["I"].iloc[-1] * 1e4
        lim, ylim = ((-12, 14), (-8, 10)) if sym == "XAUUSDT" else ((-12, 6), (-8, 3))
        ax.axvspan(I - 5, I + 5, color=GRID, alpha=0.8, lw=0)
        xs = np.linspace(*lim, 200)
        ax.plot(xs, xs + np.clip(I - xs, -5, 5), color=INK2, lw=0.8, ls="--", zorder=1)
        ax.plot(P, Fo, "o", ms=2.0, color=SYM_COLOR[sym], alpha=0.55, mec="none", zorder=2)
        ax.set_xlim(*lim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("P̄（溢价指数加权均值, bp）")
        ax.text(I, ylim[0] + 0.6, f"死区 [{I - 5:.0f}, {I + 5:.0f}]\nF ≡ I = {I:.0f}bp", ha="center",
                va="bottom", fontsize=6.2, color=INK2)
        _tag(ax, tag)
    axes[0].set_ylabel("官方 F·8/N（bp）")
    ax = axes[2]
    for sym in C.SYMBOLS:
        m = rep[sym]
        ax.plot(m.index, m["err_bp"], "o", ms=1.6, color=SYM_COLOR[sym], alpha=0.6, mec="none", label=SYM_NAME[sym])
    ax.axhspan(-0.1, 0.1, color=GRID, alpha=0.8, lw=0)
    big = rep["XAUUSDT"][rep["XAUUSDT"]["err_bp"].abs() > 0.1]
    offs = [(8, 10), (8, -9), (10, -2), (8, 0)]  # 逐个错开，避免标签重叠
    for (t, r), o in zip(big.iterrows(), offs):
        ax.annotate(t.strftime("%m-%d %H:%M"), (t, r["err_bp"]), xytext=o, textcoords="offset points",
                    fontsize=5.8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_ylabel("我算的 − 官方（bp）")
    ax.set_ylim(-1.0, 0.4)
    _month_axis(ax)
    ax.legend(loc="lower right", ncol=2, markerscale=2.5)
    _tag(ax, "(c) 逐期复现误差（灰带 ±0.1bp）")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig2_funding")


# ------------------------------------------------------------------ 图 3：拉得多快、谁在动
def fig_speed(dyn: dict, preds: dict):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.15), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    ax = axes[0]
    for sym in C.SYMBOLS:
        lp = dyn[sym]["lp_train"]
        ax.fill_between(lp.index, lp["beta"] - 1.96 * lp["se"], lp["beta"] + 1.96 * lp["se"],
                        color=SYM_COLOR[sym], alpha=0.15, lw=0)
        ax.plot(lp.index, lp["beta"], "-o", ms=2.2, color=SYM_COLOR[sym], label=SYM_NAME[sym])
    ax.axhline(-0.5, color=INK2, lw=0.7, ls=":")
    ax.text(23.8, -0.48, "回吐一半", ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax.axvline(4, color=INK2, lw=0.6, ls="--")
    ax.set_xlabel("h（小时）")
    ax.set_ylabel("β_h")
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(-1.0, 0)
    ax.legend(loc="upper right")
    _tag(ax, "(a) 局部投影：偏离之后 h 小时回吐多少")

    ax = axes[1]
    rows = [("XAU 全样本", dyn["XAUUSDT"]["legs"]["全样本"]), ("XAU 05-08 前", dyn["XAUUSDT"]["legs"]["05-08 前"]),
            ("XAU 05-08 后", dyn["XAUUSDT"]["legs"]["05-08 后"]), ("PAXG 全样本", dyn["PAXGUSDT"]["legs"]["全样本"])]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        sp = float(np.clip(r["share_perp"], 0, 1))
        ax.barh(yi, sp, color=BLUE, height=0.55, edgecolor=SURF, lw=1)
        ax.barh(yi, 1 - sp, left=sp, color=VIOLET, height=0.55, edgecolor=SURF, lw=1)
        ax.text(0.02, yi, f"{sp:.0%}", va="center", color="white", fontsize=6.5, fontweight="bold")
        ax.text(0.98, yi, f"{1 - sp:.0%}", va="center", ha="right", color="white", fontsize=6.5, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.grid(False)
    ax.text(0, len(rows) - 0.35, "■ 永续向现货靠", color=BLUE, fontsize=6.5)
    ax.text(1, len(rows) - 0.35, "现货向永续靠 ■", color=VIOLET, fontsize=6.5, ha="right")
    ax.set_ylim(-0.6, len(rows) - 0.1)
    _tag(ax, "(b) 4h 回归由哪一腿完成")

    ax = axes[2]
    for sym in C.SYMBOLS:
        p = preds[sym]
        d = ((p["y"] - p["base"]) ** 2 - (p["y"] - p["model"]) ** 2).cumsum()
        ax.plot(p.index, d / d.iloc[-1], color=SYM_COLOR[sym], label=SYM_NAME[sym])
    p = preds[C.SYMBOLS[0]]
    ax.plot([p.index[0], p.index[-1]], [0, 1], color=MUTED, lw=0.7, ls=":")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    ax.set_ylabel("平方误差改进的累计占比")
    ax.legend(loc="upper left")
    _tag(ax, "(c) 测试期改进是匀速还是靠几次跳跃")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig3_speed")


# ------------------------------------------------------------------ 图 4：结算锯齿
def fig_event(dyn: dict):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.85), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ev, pl = dyn["XAUUSDT"]["event"], dyn["XAUUSDT"]["event_placebo"]
    n = ev.attrs.get("n", {})
    for g, col in (("F>利息项", BLUE), ("F<利息项", ORANGE)):
        ax.plot(ev.index, ev[g], color=col, lw=1.3, label=f"结算时刻，{g.replace('利息项', 'I')}（n={n.get(g, '')}）")
        ax.plot(pl.index, pl[g], color=col, lw=0.9, ls="--", alpha=0.8)
    ax.plot([], [], color=INK2, ls="--", lw=0.9, label="安慰剂：结算前 2h 的整点")
    ax.axvline(0, color=INK2, lw=0.7)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("相对结算时刻（分钟）")
    ax.set_ylabel("溢价指数变化（bp）")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-3.6, 6.2)
    ax.legend(loc="upper center", fontsize=6.0, ncol=3, columnspacing=0.8, handlelength=1.6)
    _tag(ax, "(a) XAU：结算前溢价被压向 0，结算后 1–5 分钟弹回")
    ax = axes[1]
    ev = dyn["PAXGUSDT"]["event"]
    for g, col in (("F<利息项", ORANGE), ("F=利息项", GREEN)):
        if g in ev:
            ax.plot(ev.index, ev[g], color=col, lw=1.2, label=g.replace("利息项", "I"))
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_ylim(-4, 4)
    ax.set_xlim(-60, 60)
    ax.set_xlabel("相对结算时刻（分钟）")
    ax.legend(loc="upper left")
    _tag(ax, "(b) PAXG：同一纵轴下几乎没有锯齿")
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig4_event")


# ------------------------------------------------------------------ 图 5：休市
def fig_closure(cl: dict):
    fig = plt.figure(figsize=(FULL_W, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1])
    paths = cl["paths"]
    for i, key in enumerate(("05-08 前", "05-08 后")):
        ax = fig.add_subplot(gs[0, i])
        r, w = paths[key]
        base = w["perp"].dropna().iloc[0]
        f = lambda s: 1e4 * np.log(s / base)
        ax.plot(w.index, f(w["perp"]), color=BLUE, lw=0.9, label="XAU 永续")
        ax.plot(w.index, f(w["index"]), color=ORANGE, lw=1.1, label="参考指数")
        ax.plot(w.index, f(w["spot"]), color=VIOLET, lw=1.1, label="真实金价")
        ax.axvspan(r["close"], r["reopen"], color=GRID, alpha=0.6, lw=0)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.set_ylabel("相对起点（bp）" if i == 0 else "")
        if i == 0:
            ax.legend(loc="lower left", fontsize=6.2)
        _tag(ax, f"({'ab'[i]}) {key}：{r['close']:%m-%d} 周末")
    ax = fig.add_subplot(gs[0, 2])
    ep = cl["episodes"]
    wk = ep[ep["kind"] == "周末/假日"]
    lim = 260
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.7, ls=":")
    for post, col, lab in ((False, ORANGE, "05-08 前"), (True, VIOLET, "05-08 后")):
        g = wk[wk["post58"] == post]
        ax.plot(g["xau_perp_move"], g["gold_gap"], "o", ms=3.4, color=col, mec="white", mew=0.5, label=lab)
        b, a = np.polyfit(g["xau_perp_move"], g["gold_gap"], 1)
        xs = np.array([-lim, lim])
        ax.plot(xs, a + b * xs, color=col, lw=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("休市期间 XAU 永续涨跌（bp）")
    ax.set_ylabel("开盘时金价缺口（bp）")
    ax.legend(loc="upper left")
    _tag(ax, "(c) 周末永续走势预告开盘缺口")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig5_closure")


def run_all(res: dict):
    setup()
    fig_clock(res["clock_scan"], res["clock_breaks"])
    fig_funding({s: res["funding"][s]["main"] for s in C.SYMBOLS})
    fig_speed(res["dyn"], {s: res["models"][s]["pred"] for s in C.SYMBOLS})
    fig_event(res["dyn"])
    fig_closure(res["closure"])
