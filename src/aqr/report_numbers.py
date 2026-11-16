# -*- coding: utf-8 -*-
"""把全部结果写成 LaTeX 宏与表格（report/generated/），报告正文不手写任何数字。

同时输出 outputs/summary.json（关键数字）与 docs/DATA_ISSUES.md（隐藏数据问题完整清单）。
核对数字附录里：价格与官方资金费原样取自原始文件文本，不做四舍五入；其余保留 3 位小数。
"""
import json
from decimal import Decimal

import numpy as np
import pandas as pd

from . import clock, config as C, io_raw as R, funding_params as FP

SYMS = C.SYMBOLS
SN = {"XAUUSDT": "XAU", "PAXGUSDT": "PAXG"}
CHECK_TIMES = [pd.Timestamp("2026-01-14 15:00"), pd.Timestamp("2026-08-19 14:00")]
CHECK_FUNDING = [("XAUUSDT", pd.Timestamp("2026-02-02 00:00")), ("XAUUSDT", pd.Timestamp("2026-08-05 12:00")),
                 ("PAXGUSDT", pd.Timestamp("2026-08-11 04:00"))]


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def pct(x, d=1):
    return f(100 * x, d) + "\\%"


def raw_bp(rate_str: str) -> str:
    """官方资金费原始字符串 → bp，十进制精确移位（0.00046480 → 4.6480）。"""
    v = Decimal(rate_str) * 10000
    return f"{v:.4f}".replace("-", "−")


class Macros:
    def __init__(self):
        self.d = {}

    def __setitem__(self, k, v):
        assert k.isalpha(), k
        self.d[k] = v

    def write(self, path):
        lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in self.d.items()]
        path.write_text("% 自动生成：src/aqr/report_numbers.py，请勿手改\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _tex_table(path, header, rows, align, caption_note=None):
    body = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "mid":
            body.append("\\midrule")
        else:
            body.append(" & ".join(str(x) for x in r) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 核对数字
def check_prices(m: Macros):
    rows = []
    for i, t in enumerate(CHECK_TIMES):
        xp = R.raw_kline_line("perp", "XAUUSDT", t)[4]
        lab = clock.utc_end_to_local_label(t)
        hd_rows = R.raw_histdata_lines(lab)
        hd = hd_rows[-1][4]  # 多行时收盘取最后一行（题面规则）
        pp = R.raw_kline_line("perp", "PAXGUSDT", t)[4]
        ps = R.raw_kline_line("spot", "PAXGUSDT", t)[4]
        bx = 1e4 * np.log(float(xp) / float(hd))
        bp = 1e4 * np.log(float(pp) / float(ps))
        tt = lambda v: "\\texttt{" + v + "}"
        rows.append([t.strftime("%m-%d %H:%M"), tt(xp), tt(hd), f(bx), tt(pp), tt(ps), f(bp)])
        tag = "A" if i == 0 else "B"
        m[f"chk{tag}local"] = lab.strftime("%Y%m%d %H%M%S")
        m[f"chk{tag}hdrows"] = str(len(hd_rows))
    _tex_table(C.GEN / "tab_check_prices.tex",
               ["时刻（UTC，分钟结束）", "XAU 永续", "真实金价", "XAU 基差", "PAXG 永续", "PAXG 代币", "PAXG 基差"],
               rows, "lrrrrrr")


def check_funding(res, m: Macros):
    rows = []
    for sym, t in CHECK_FUNDING:
        raw = R.raw_funding_line(sym, t)
        mine = res["funding"][sym]["main"].loc[t]
        rows.append([sym, t.strftime("%Y-%m-%d %H:%M"), "\\texttt{" + raw[2] + "}", raw_bp(raw[2]), f(mine["calc"] * 1e4),
                     f(mine["err_bp"], 3, True)])
    _tex_table(C.GEN / "tab_check_funding.tex", ["合约", "结算时刻", "官方原值", "官方 (bp)", "复现值 (bp)", "差 (bp)"],
               rows, "llrrrr")


def check_model(res, m: Macros):
    # 预测模型测试期 RMSE（零 / 基线 / 我的模型 × 全期 / 7 / 8 / 9 月）
    rows = []
    for sym in SYMS:
        r, tab = res["models"][sym], res["eval"][sym]["table"]
        for name, col in (("零", "zero"), ("基线", "base"), ("我的模型", "model")):
            rows.append([SN[sym] if name == "零" else "", name] + [f(tab.loc[col, c]) for c in ("全期", "7月", "8月", "9月")])
        rows.append(["", "样本数（测试）"] + [str(int(tab.loc["n", c])) for c in ("全期", "7月", "8月", "9月")])
        if sym == SYMS[0]:
            rows.append("mid")
    _tex_table(C.GEN / "tab_rmse.tex", ["合约", "预测", "全期", "7 月", "8 月", "9 月"], rows, "llrrrr")

    # 复现误差：每个合约的结算期数、最大绝对误差、误差超过 0.1 bp 的期数
    rows = []
    for sym in SYMS:
        s = res["funding"][sym]["summary"]
        rows.append([SN[sym], str(s["n"]), f(s["max_abs"]), str(s["n_gt_0.1"])])
    _tex_table(C.GEN / "tab_check_repro.tex",
               ["合约", "结算期数", "最大绝对误差 (bp)", "误差$>$0.1bp 期数"],
               rows, "lrrr")

    # 预测模型参数：训练 / 测试样本数、基线 α、β
    rows = []
    for sym in SYMS:
        r = res["models"][sym]
        rows.append([SN[sym], str(len(r["train"])), str(len(r["test"])), f(r["ab"][0]), f(r["ab"][1])])
    _tex_table(C.GEN / "tab_check_model.tex",
               ["合约", "训练样本", "测试样本", "基线 $\\alpha$", "基线 $\\beta$"],
               rows, "lrrrr")


# ------------------------------------------------------------------ 正文表格
def params_table():
    rows = []
    for s in FP.SEGMENTS:
        if s.cap is None:
            cap = "未触及"
        elif s.floor is not None:
            cap = f"±{s.cap * 100:g}\\%"
        else:
            cap = f"+{s.cap * 100:g}\\%（下限未触及）"
        rows.append([SN[s.sym], "样本起点" if s.effective < C.START else s.effective.strftime("%m-%d %H:%M"),
                     f"{s.I * 100:g}\\%", cap])
    _tex_table(C.GEN / "tab_params.tex", ["合约", "生效（结算晚于）", "利息项", "上下限"], rows, "llrl")


def repro_table(res):
    rows = []
    for sym in SYMS:
        fr = res["funding"][sym]
        variants = [("本文", fr["summary"]), ("收盘价加权", fr["methods"]["close_lin"])]
        if sym == "XAUUSDT":  # PAXG 参数从未变过，"回填"与本文相同，不重复列
            variants.append(("样本末参数回填", fr["naive"]))
        for lab, s in variants:
            rows.append([SN[sym] if lab == "本文" else "", lab, str(s["n_gt_0.1"]), f(s["max_abs"]),
                         f(s["rmse"]), pct(s["share_in_env"])])
        if sym == SYMS[0]:
            rows.append("mid")
    _tex_table(C.GEN / "tab_repro.tex", ["", "口径", "$>$0.1bp", "最大误差", "RMSE", "包络内"], rows, "llrrrr")


# 每组特征的构造方式与纳入理由（正文表 4 的文字列；数字列来自交叉验证消融）
FEATURE_DOC = {
    "偏离": ("基差减过去 1、4、24、72 小时均值", "基线的来源；区分新偏离和旧偏离"),
    "噪声": ("基差减 15 分钟均值；成交价减盘口价", "瞬间的报价误差几分钟内就消失"),
    "溢价与指数差": ("两部分各自相对 24 小时均值的偏离", "资金费只作用于溢价"),
    "资金费": ("本期估计值、上期官方值、距结算时间、死区外×偏离", "结算前后推动溢价；死区外才随溢价变"),
    "动量": ("永续、现货各自 15 分钟和 1 小时涨跌", "区分偏离的驱动方"),
    "波动与流动性": ("基差波动及其×偏离、主动买入占比、零成交占比", "成交稀少时是旧报价，偏离不可信"),
    "休市日历": ("距收盘、距开盘时间，窗口内休市占比及其×偏离", "跨越休市时现货不动；按公开规则写成"),
}


def features_table(res):
    """特征组说明表：构造方式、纳入理由、交叉验证上去掉该组后 RMSE 的增幅。"""
    a = {sym: res["models"][sym]["ablation"].set_index("组") for sym in SYMS}
    rows = [[g, *FEATURE_DOC[g], *(f(a[sym].loc[g, "剔除损失"], 3, True) for sym in SYMS)]
            for g in a[SYMS[0]].index]
    cols = "".join(">{\\raggedright\\arraybackslash}p{%s}" % w for w in ("2.0cm", "6.2cm", "5.6cm")) + "rr"
    _tex_table(C.GEN / "tab_features.tex", ["特征组", "构造方式", "纳入理由", "XAU", "PAXG"], rows, cols)


def issues_table(res, m: Macros):
    q = res["dataqa"]
    ck = q["clock"]
    i5, dp, cm, us = q["idx0305"], q["dups"], q["close_minute"], q["usdt"]
    t624 = q["funding_archive"]["XAUUSDT"]["test_0624"]
    rows = [
        ["HistData 时钟 03-29 才切到 UTC−4", "图 1 的两条独立证据",
         f"分段换算。若恒按 UTC−5，XAU 测试期基线 RMSE 从 {f(ck.iloc[0]['rmse_base'], 2)} 变成 {f(ck.iloc[1]['rmse_base'], 2)}；"
         f"按美国夏令时为 {f(ck.iloc[2]['rmse_base'], 2)}"],
        ["XAU 参考指数 03-05 04:40–05:54 停滞", f"指数比金价最多高 {f(i5['index_vs_spot_max_bp'], 1)}bp，永续与金价之差中位数只有 {f(i5['perp_vs_spot_med_bp'], 1)}bp",
         f"不改数据。该期官方值 {f(i5['official_bp'], 4)}bp 落在误差范围 [{f(i5['env_lo'], 2)}, {f(i5['env_hi'], 2)}] 外，是唯一的一期"],
        ["HistData 同一分钟多行是两路报价交错", f"{dp['n_minutes']} 个分钟里 {dp['nonmono_rows']} 行时间倒序，同分钟收盘最多差 {f(dp['close_spread_bp_max'], 1)}bp",
         f"按题面规则合并。影响 {dp['samples_touched']} 个样本；改用首行收盘，XAU 基线 RMSE {f(dp['rule']['rmse_base'])}→{f(dp['alt_first_close']['rmse_base'])}"],
        ["每日停盘前最后一分钟金价偏低", f"{cm['n_breaks']} 次停盘平均：倒数第二分钟基差 {f(cm['basis_close_m1'], 2)}bp，最后一分钟 {f(cm['basis_close'], 2)}bp",
         f"按规则保留，用距收盘特征吸收。{cm['samples_on_close_minute']} 个样本的 $t$ 或 $t$+4h 落在这一分钟"],
        ["07-16 换算实际在 08:33 生效", f"溢价指数在 08:33 跳 {f(us['prem_jump_bp'], 1)}bp，此前冻结 {us['prem_frozen_min']} 分钟",
         f"特征只用偏离，不用水平。指数差均值 {f(us['index_leg_before'], 2)}→{f(us['index_leg_after'], 2)}bp"],
        ["资金费 06-24 04:00 缺失，但实际已结算", f"下一期按 4h 窗口算误差 {f(t624['err_4h_bp'], 3)}bp，按 8h 算 {f(t624['err_8h_bp'], 3)}bp",
         f"剔除该期，下一期仍用 4h 窗口；误当成未结算会多出 {f(t624['err_8h_bp'], 2)}bp 误差"],
    ]
    _tex_table(C.GEN / "tab_issues.tex", ["问题", "数据证据", "处理与影响"], rows,
               "".join(">{\\raggedright\\arraybackslash}p{%s}" % w for w in ("4.3cm", "5.5cm", "6.9cm")))


# ------------------------------------------------------------------ 宏
def macros(res, m: Macros):
    fz = res["funding"]
    for sym in SYMS:
        t = SN[sym]
        s = fz[sym]["summary"]
        m[f"rep{t}n"], m[f"rep{t}max"], m[f"rep{t}big"] = str(s["n"]), f(s["max_abs"]), str(s["n_gt_0.1"])
        m[f"rep{t}rmse"], m[f"rep{t}env"] = f(s["rmse"]), pct(s["share_in_env"])
        m[f"rep{t}naivebig"], m[f"rep{t}naivemax"] = str(fz[sym]["naive"]["n_gt_0.1"]), f(fz[sym]["naive"]["max_abs"])
        m[f"rep{t}closebig"] = str(fz[sym]["methods"]["close_lin"]["n_gt_0.1"])
        r = res["models"][sym]
        m[f"m{t}ntrain"], m[f"m{t}ntest"] = str(len(r["train"])), str(len(r["test"]))
        m[f"m{t}alpha"], m[f"m{t}beta"] = f(r["ab"][0]), f(r["ab"][1])
        m[f"m{t}ridge"] = f"{r['alpha']:g}"
        m[f"m{t}groups"] = "、".join(r["groups"])
        m[f"m{t}cv"], m[f"m{t}cvbase"] = f(r["cv_best"]), f(r["cv_base"])
        tab, sig = res["eval"][sym]["table"], res["eval"][sym]["sig"]
        for col, nm in (("zero", "zero"), ("base", "base"), ("model", "model"), ("hgb", "hgb")):
            m[f"m{t}{nm}"] = f(tab.loc[col, "全期"])
        ml = res["mlp"][sym]
        m[f"m{t}mlp"], m[f"m{t}mlpsd"] = f(ml["test_mean"]), f(ml["test_sd"])
        m[f"m{t}mlphidden"] = str(ml["hidden"][0])
        m[f"m{t}gain"], m[f"m{t}gainpct"] = f(sig["gain"]), pct(sig["gain_pct"])
        m[f"m{t}dm"], m[f"m{t}dmp"] = f(sig["dm24"][0], 2), f(sig["dm24"][1], 4)
        m[f"m{t}dmfour"], m[f"m{t}dmfourp"] = f(sig["dm4"][0], 2), f(sig["dm4"][1], 3)
        m[f"m{t}cw"], m[f"m{t}cwp"] = f(sig["cw24"][0], 2), f(sig["cw24"][1], 4)
        m[f"m{t}bootlo"], m[f"m{t}boothi"] = f(sig["boot_ci"][0]), f(sig["boot_ci"][1])
        m[f"m{t}bootp"] = f(sig["boot_p_le0"], 3)
        m[f"m{t}trim"], m[f"m{t}topshare"] = f(sig["trim_gain"]), pct(sig["top1_share"], 0)
        m[f"m{t}mingain"] = f(min(sig["month_gain"].values()))
        m[f"m{t}basegain"] = pct(1 - tab.loc["base", "全期"] / tab.loc["zero", "全期"])
        d = res["dyn"][sym]
        m[f"d{t}beta"], m[f"d{t}hl"] = f(d["beta_train"]), f(d["hl_beta"], 1)
        m[f"d{t}betaiv"], m[f"d{t}hliv"] = f(d["beta_iv"]), f(d["hl_iv"], 1)
        mr = d["model_response"]
        m[f"d{t}fresh"], m[f"d{t}old"] = pct(-mr["fresh"], 0), pct(-mr["old"], 0)
        m[f"d{t}hlold"] = f(mr["hl_old"], 1)
        lp = d["lp_train"]["beta"]
        m[f"d{t}lpone"], m[f"d{t}lpmax"], m[f"d{t}lpmaxh"] = f(lp.loc[1], 2), f(lp.min(), 2), str(int(lp.idxmin()))
        m[f"d{t}lpend"] = f(lp.loc[24], 2)
        for k, nm in (("全样本", "all"), ("训练", "train"), ("测试", "test")):
            m[f"d{t}share{nm}"] = pct(d["legs"][k]["share_perp"], 0)
        m[f"d{t}aperp"], m[f"d{t}aspot"] = f(d["legs"]["全样本"]["alpha_perp"]), f(d["legs"]["全样本"]["alpha_spot"])
        m[f"d{t}seperp"], m[f"d{t}sespot"] = f(d["legs"]["全样本"]["se_perp"]), f(d["legs"]["全样本"]["se_spot"])
        m[f"d{t}rawseperp"] = f(d["legs_raw"]["全样本"]["se_perp"], 2)
        m[f"d{t}aperpiv"], m[f"d{t}aspotiv"] = f(d["legs_iv"]["全样本"]["alpha_perp"]), f(d["legs_iv"]["全样本"]["alpha_spot"])
        pl = d["prem_index_legs"]
        m[f"d{t}premleg"], m[f"d{t}idxleg"] = f(pl["prem"]), f(pl["idx"])
        m[f"d{t}premshare"] = pct(pl["prem"] / (pl["prem"] + pl["idx"]), 0)
        pdist = d["pdist"]
        m[f"d{t}Pmed"], m[f"d{t}dead"] = f(pdist["P_median"], 2), pct(pdist["share_dead"], 0)
        m[f"d{t}below"], m[f"d{t}above"] = pct(pdist["share_below"], 0), pct(pdist["share_above"], 0)
        ps = d["premium_state"][1].set_index("状态")
        m[f"d{t}betadead"], m[f"d{t}betaout"] = f(ps.loc["死区内", "beta"]), f(ps.loc["死区外", "beta"])
    dx = res["dyn"]["XAUUSDT"]
    m["dXAUsharepre"], m["dXAUsharepost"] = pct(dx["legs"]["05-08 前"]["share_perp"], 0), pct(dx["legs"]["05-08 后"]["share_perp"], 0)
    m["dXAUshareivpost"] = pct(dx["legs_iv"]["05-08 后"]["share_perp"], 0)
    ev, pl = dx["event"], dx["event_placebo"]
    m["evUpPre"], m["evUpJump"] = f(ev.loc[0, "F>利息项"] - ev.loc[-60, "F>利息项"], 2), f(ev.loc[5, "F>利息项"] - ev.loc[0, "F>利息项"], 2)
    m["evDnPre"], m["evDnJump"] = f(ev.loc[0, "F<利息项"], 2), f(ev.loc[5, "F<利息项"] - ev.loc[0, "F<利息项"], 2)
    m["evUpOne"], m["evDnOne"] = f(ev.loc[1, "F>利息项"] - ev.loc[0, "F>利息项"], 2), f(ev.loc[1, "F<利息项"] - ev.loc[0, "F<利息项"], 2)
    m["evPlUpJump"], m["evPlDnJump"] = f(pl.loc[5, "F>利息项"] - pl.loc[0, "F>利息项"], 2), f(pl.loc[5, "F<利息项"] - pl.loc[0, "F<利息项"], 2)
    nu, nd = ev.attrs["n"]["F>利息项"], ev.attrs["n"]["F<利息项"]
    m["evNup"], m["evNdn"] = str(nu), str(nd)
    ju, jd = ev.loc[5, "F>利息项"] - ev.loc[0, "F>利息项"], ev.loc[0, "F<利息项"] - ev.loc[5, "F<利息项"]
    m["evAvgJump"] = f((nu * ju + nd * jd) / (nu + nd), 2)
    m["evPerYear"] = str(int(round((nu + nd) * 365 / 272)))
    evp = res["dyn"]["PAXGUSDT"]["event"]
    m["evPaxgMax"] = f(evp.abs().max().max(), 2)
    # 资金费参数证据
    ih = fz["XAUUSDT"]["interval_0130"]
    row = ih[(ih["窗口(h)"] == 8) & (ih["N"] == 8) & (ih["上下限"] == "无")].iloc[0]
    m["fzRawOneThirty"] = f(row["F_bp"], 2)
    row = ih[(ih["窗口(h)"] == 8) & (ih["N"] == 4) & (ih["上下限"] == "无")].iloc[0]
    m["fzAltOneThirty"], m["fzAltOneThirtyErr"] = f(row["F_bp"], 2), f(row["误差_bp"], 2, True)
    mx = fz["XAUUSDT"]["main"]
    m["fzCapRaw"] = f(mx.loc[pd.Timestamp("2026-02-28 12:00"), "calc_uncapped"] * 1e4, 2)
    m["fzBoundNaive"] = f(res["dataqa"]["funding_archive"]["boundary_err_naive"], 3, True)
    im = fz["XAUUSDT"]["implied_I"]
    m["fzIpre"] = str(int((im.index <= pd.Timestamp("2026-01-23 08:00")).sum()))
    m["fzIpost"] = str(int((im.index > pd.Timestamp("2026-01-23 08:00")).sum()))
    m["fzIpostmax"] = f(im.loc[im.index > pd.Timestamp("2026-01-23 08:00"), "I_implied_bp"].abs().max(), 4)
    mp = fz["PAXGUSDT"]["implied_I"]
    m["fzIpaxg"] = str(len(mp))
    m["fzIpaxgdev"] = f((mp["I_implied_bp"] - 1).abs().max(), 4)
    m["fzOutsideDate"] = "03-05 08:00"
    # 时钟
    sc = res["clock_scan"]
    pre, post = sc[sc.index <= pd.Timestamp("2026-03-27")], sc[sc.index >= pd.Timestamp("2026-03-30")]
    m["ckPreDays"], m["ckPostDays"] = str(int((pre["best"] == 300).sum())), str(int((post["best"] == 240).sum()))
    m["ckPreN"], m["ckPostN"] = str(len(pre)), str(len(post))
    m["ckCorrPre"], m["ckCorrPost"] = f(pre.loc[pre.index >= "2026-02-01", "corr_300"].median(), 3), f(post["corr_240"].median(), 3)
    m["ckNeighbor"] = f(np.nanmedian(np.r_[sc["corr_m1"], sc["corr_p1"]]), 3)
    cv = res["calendar_validate"]
    m["calOpenNoData"], m["calClosedData"] = str(cv["open_no_data"]), str(cv["closed_but_data"])
    # 休市
    cl = res["closure"]
    pr = cl["pred"]
    for key, nm in ((("周末/假日", "05-08 前", "xau_perp_move"), "WkPre"), (("周末/假日", "05-08 后", "xau_perp_move"), "WkPost"),
                    (("周末/假日", "05-08 后", "paxg_spot_move"), "WkPostPaxg"), (("日休", "05-08 后", "xau_perp_move"), "DayPost"),
                    (("日休", "05-08 前", "xau_perp_move"), "DayPre")):
        p = pr[key]
        m[f"cl{nm}n"], m[f"cl{nm}slope"], m[f"cl{nm}rsq"] = str(p["n"]), f(p["slope"], 2), f(p["r2"], 2)
        m[f"cl{nm}lo"], m[f"cl{nm}hi"], m[f"cl{nm}hit"] = f(p["ci"][0], 2), f(p["ci"][1], 2), pct(p["hit"], 0)
        m[f"cl{nm}sd"], m[f"cl{nm}resid"], m[f"cl{nm}icpt"] = f(p["gap_sd"], 1), f(p["resid_sd"], 1), f(p["intercept"], 1)
    ww = cl["wrong_way"]
    m["clFrozenPre"], m["clFrozenPost"] = pct(ww[False]["frozen_share"], 1), pct(ww[True]["frozen_share"], 1)
    m["clPremPre"], m["clPremPost"] = f(ww[False]["prem_absmed"], 1), f(ww[True]["prem_absmed"], 1)
    m["clAgreePre"], m["clWkFundAbs"] = pct(ww[False]["sign_agree_prem_gap"], 0), f(ww[False]["funding_sum_mean_abs"], 1)
    ep = cl["episodes"]
    wk = ep[ep["kind"] == "周末/假日"]
    m["clWkFundSigned"] = f(wk.loc[~wk["post58"], "funding_sum_bp"].mean(), 1)
    m["clWkExcess"] = f((wk.loc[~wk["post58"], "xau_perp_move"] - wk.loc[~wk["post58"], "gold_gap"]).mean(), 1)
    m["clWkExcessPost"] = f((wk.loc[wk["post58"], "xau_perp_move"] - wk.loc[wk["post58"], "gold_gap"]).mean(), 1)
    m["clNwk"], m["clNday"] = str(len(wk)), str(int((ep["kind"] == "日休").sum()))
    fu = cl["funding"].set_index(["post58", "state"])
    m["clFpreClosed"], m["clFpreOpen"] = f(fu.loc[(False, "全休市"), "absmean"], 2), f(fu.loc[(False, "全开市"), "absmean"], 2)
    m["clFpostClosed"], m["clNpostClosed"] = f(fu.loc[(True, "全休市"), "absmean"], 2), str(int(fu.loc[(True, "全休市"), "n"]))
    m["clNpreClosed"], m["clFpreNZ"] = str(int(fu.loc[(False, "全休市"), "n"])), pct(fu.loc[(False, "全休市"), "nonzero"], 0)
    ro = cl["reopen"]
    m["clReopenPre"], m["clReopenPost"] = f(ro["05-08 前"].iloc[0], 1), f(ro["05-08 后"].iloc[0], 1)
    m["clReopenPreEnd"], m["clReopenPostEnd"] = f(ro["05-08 前"].iloc[-1], 1), f(ro["05-08 后"].iloc[-1], 1)
    sp = cl["spread"]
    m["clSpreadPost"], m["clSpreadRev"] = f(sp["post_abs"], 1), pct(-sp["post_slope"], 0)
    m["clSpreadGross"] = f(-sp["post_slope"] * sp["post_abs"], 1)
    g = cl["pred"][("周末/假日", "05-08 后", "xau_perp_move")]
    m["clVarCut"] = pct(1 - (g["resid_sd"] / g["gap_sd"]) ** 2, 0)
    m["clVaRun"], m["clVaRhe"] = f(1.645 * g["gap_sd"], 0), f(1.645 * g["resid_sd"], 0)
    for sym in SYMS:
        o = fz[sym]["main"]["official"] * 1e4
        m[f"fz{SN[sym]}Fmean"], m[f"fz{SN[sym]}Fann"] = f(o.mean(), 2), pct(o.mean() * 6 * 365 / 1e4, 1)
    # 数据问题
    q = res["dataqa"]
    m["qDupMin"], m["qDupFirst"] = str(q["dups"]["n_minutes"]), q["dups"]["first_local"].strftime("%m-%d %H:%M")
    m["qUsdtBefore"], m["qUsdtAfter"] = f(q["usdt"]["index_leg_before"], 2), f(q["usdt"]["index_leg_after"], 2)
    m["qHdBias"] = f(q["hd_bias"]["pre_0716_median"], 2)
    m["qStaleJan"] = pct(q["stale"]["zero_vol_share_jan"], 1)
    m["qBetaExJan"] = f(q["stale"]["beta_ex_jan"])
    m["qCountLt"], m["qCountZero"] = str(q["premium_gaps"]["XAUUSDT"]["count_lt12"]), str(q["premium_gaps"]["XAUUSDT"]["count0"])
    m["qMaxNaive"], m["qMaxCnt"] = f(q["premium_gaps"]["max_err_naive"]), f(q["premium_gaps"]["max_err_cnt"])
    ck = q["clock"]
    m["qClockBase"], m["qClockEst"], m["qClockUS"] = f(ck.iloc[0]["rmse_base"], 2), f(ck.iloc[1]["rmse_base"], 1), f(ck.iloc[2]["rmse_base"], 2)
    m["qClockSdRule"], m["qClockSdEst"] = f(ck.iloc[0]["basis_sd"], 1), f(ck.iloc[1]["basis_sd"], 1)
    m["dXAUrawshare"] = pct(res["dyn"]["XAUUSDT"]["legs_raw"]["全样本"]["share_perp"], 0)
    pb = res["dyn"]["XAUUSDT"]["premium_state"][0].set_index(["状态", "|偏离|档(bp)"])["beta"]
    m["dXAUbinDead"], m["dXAUbinOut"] = f(pb.loc[("死区内", "[2.0, 4.0)")], 2), f(pb.loc[("死区外", "[2.0, 4.0)")], 2)
    m["chkFzErr"] = f(res["funding"]["XAUUSDT"]["main"].loc[CHECK_FUNDING[0][1], "err_bp"], 3, True)
    notional = 1e7
    m["clVaRsave"] = f((1.645 * g["gap_sd"] - 1.645 * g["resid_sd"]) * 1e-4 * notional / 1e4, 1)
    m["qCloseMinA"], m["qCloseMinB"] = f(q["close_minute"]["basis_close_m1"], 2), f(q["close_minute"]["basis_close"], 2)
    m["qOpenMin"] = f(q["close_minute"]["basis_open"], 2)


def data_issues_md(res):
    q = res["dataqa"]
    ck = q["clock"]
    lines = ["# 隐藏数据问题清单（自动生成：`python run_all.py`）", "",
             "每条：问题 · 证据（可复现的数字）· 处理 · 对结论数字的影响。", ""]
    lines += ["## 1. HistData 时钟不是\"EST 无夏令时\"，也不是按美国夏令时切换",
              "逐日最优偏移 300 分钟（至 03-27）→ 240 分钟（03-30 起）；03-09~03-26 日休在本地 16:00（美国已夏令时、HistData 仍 UTC−5）。",
              "", "```", ck.round(3).to_string(), "```", ""]
    d = q["dups"]
    lines += ["## 2. 同分钟多行是两路报价交错", f"首次出现于本地 {d['first_local']}；{d['n_minutes']} 分钟/{d['n_rows']} 行；文件内时间倒序 {d['nonmono_rows']} 行；"
              f"同分钟收盘最大相差 {d['close_spread_bp_max']:.1f}bp（均值 {d['close_spread_bp_mean']:.2f}bp）。按题面规则合并；"
              f"改用首行收盘时 XAU 测试基线 RMSE {d['rule']['rmse_base']:.3f}→{d['alt_first_close']['rmse_base']:.3f}，触及 {d['samples_touched']} 个样本。", ""]
    lines += ["## 3. 月度文件缺的最后一分钟恰好落在整点样本上",
              "| 月份 | 缺失本地标签 | 对应 UTC 结束时刻 | 是整点 |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in q["lastmin"]["rows"]] + [""]
    lines += ["## 4. 参考指数/溢价指数 6 月月度归档缺 06-29 整天",
              f"日度归档补回 {q['missing_day']}（分钟数）；挽回每个合约 {q['missing_day']['funding_periods_rescued_per_symbol']} 期资金费复现。", ""]
    pg = q["premium_gaps"]
    lines += ["## 5. 溢价指数 K 线有缺样本分钟（count<12）与空分钟（count=0，数值是前值顺延）",
              f"XAU count<12 共 {pg['XAUUSDT']['count_lt12']} 分钟、count=0 共 {pg['XAUUSDT']['count0']} 分钟；PAXG {pg['PAXGUSDT']}。"
              f"按样本数加权后 XAU 最大复现误差 {pg['max_err_naive']:.3f}→{pg['max_err_cnt']:.3f}bp。", ""]
    a = q["idx0305"]
    lines += ["## 6. 2026-03-05 04:40–05:54 XAU 参考指数停滞（成分源中断）",
              f"指数−金价最大 {a['index_vs_spot_max_bp']:.1f}bp，而永续−金价中位 {a['perp_vs_spot_med_bp']:.1f}bp：错的是指数；溢价指数最低 {a['prem_min_bp']:.1f}bp；"
              f"期间 {a['two_dec_minutes']} 分钟指数只有两位小数（退化为单一报价源）。08:00 期官方 {a['official_bp']:.4f}bp 落在我们包络 [{a['env_lo']:.3f}, {a['env_hi']:.3f}] 之外——"
              "交易所结算时显然没有使用归档里的这段溢价。", ""]
    u = q["usdt"]
    lines += ["## 7. 07-16 USDT/USD 换算实际生效于 08:33", f"溢价指数 08:33 跳变 {u['prem_jump_bp']:.2f}bp，之前冻结 {u['prem_frozen_min']} 分钟；"
              f"指数差（指数−金价）均值 {u['index_leg_before']:.2f}→{u['index_leg_after']:.2f}bp。", ""]
    fa = q["funding_archive"]
    lines += ["## 8. 资金费归档", f"两合约均缺 06-24 04:00；下一期 4h/8h 窗口误差：XAU {fa['XAUUSDT']['test_0624']}，PAXG {fa['PAXGUSDT']['test_0624']} → 结算发生了，只是归档缺行。"
              f"利息项切换边界：若把 01-23 08:00 结算期也按 I=0 计算，误差 {fa['boundary_err_naive']:+.3f}bp。", ""]
    cm = q["close_minute"]
    lines += ["## 9. 日休前最后一分钟 HistData 报价偏低", f"{cm['n_breaks']} 次日休平均：收盘前一分钟基差 {cm['basis_close_m1']:.2f}、最后一分钟 {cm['basis_close']:.2f}、"
              f"开盘第一分钟 {cm['basis_open']:.2f}、开盘 5 分钟后 {cm['basis_open_p5']:.2f}bp；{cm['samples_on_close_minute']} 个整点样本落在收盘分钟。", ""]
    s = q["stale"]
    lines += ["## 10. 其他", f"- XAU 永续 1 月零成交分钟 {s['zero_vol_share_jan']:.1%}（2 月 {s['zero_vol_share_feb']:.1%}）；去掉 1 月后基线 β {s['beta_all']:.3f}→{s['beta_ex_jan']:.3f}。",
              f"- HistData 相对多源参考指数系统性偏低（07-16 前中位 {q['hd_bias']['pre_0716_median']:.2f}bp），与买价报价一致，抬高 XAU 基差水平但不影响 y。",
              "- 现货 K 线时间戳为微秒、期货为毫秒；期货 CSV 有表头、现货没有。",
              "- 资金费原始文件用 Java BigDecimal 格式：零写作 `0E-8`（XAU 806 期），绝对值 <1e-6 的写成科学计数法（如 `-8.2E-7`）；按字符串截取小数位的解析会出错。",
              "- 公告称 XAU 原上下限 ±0.375%，但 01-25 20:00 官方 0.4417% 与无截断公式值一致——以数据为准，记为未识别。", ""]
    (C.REPO / "docs" / "DATA_ISSUES.md").write_text("\n".join(lines), encoding="utf-8")


def summary_json(res):
    # mlp 只留正文用到的三项，供 run_all.py --no-mlp 复用上次结果
    out = {sym: {"funding": res["funding"][sym]["summary"], "rmse": res["eval"][sym]["table"].to_dict(),
                 "alpha_beta": res["models"][sym]["ab"], "sig": res["eval"][sym]["sig"],
                 "mlp": {"test_mean": res["mlp"][sym]["test_mean"], "test_sd": res["mlp"][sym]["test_sd"],
                         "hidden": list(res["mlp"][sym]["hidden"])}} for sym in SYMS}
    (C.OUT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def write_all(res):
    m = Macros()
    check_prices(m)
    check_funding(res, m)
    check_model(res, m)
    params_table()
    repro_table(res)
    features_table(res)
    issues_table(res, m)
    macros(res, m)
    m.write(C.GEN / "numbers.tex")
    data_issues_md(res)
    summary_json(res)
    return m.d
