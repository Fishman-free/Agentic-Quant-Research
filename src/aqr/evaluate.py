# -*- coding: utf-8 -*-
"""测试期比较：RMSE 表 + "是真的还是运气"的检验。

- Diebold-Mariano：损失差 d_t = e_base² − e_model²，HAC（Newey-West, Bartlett）方差。
  目标是 4h 变化、样本按小时重叠，误差至少 MA(3)；日内还有季节性，所以主口径滞后取 24，另报 4/48。
- Clark-West：我的模型包含基线的信息（dev_24h 也在特征里），嵌套比较下 DM 偏保守，CW 做校正。
- 块自助法：按 24 个连续样本（约一天）为块重抽，给 RMSE 差的 95% 区间。
- 稳健性：剔除 |d_t| 最大的 1% 样本后再比；分月看。
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .models import rmse


def nw_var(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    T = len(x)
    v = x @ x / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / T
    return v


def dm_test(e1, e2, lag=24):
    """H0: 两个预测平方误差期望相等。正的统计量表示 e2（第二个）更好。"""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    s = d.mean() / np.sqrt(nw_var(d, lag) / len(d))
    return float(s), float(2 * (1 - stats.norm.cdf(abs(s))))


def clark_west(y, f_small, f_big, lag=24):
    y, f1, f2 = map(np.asarray, (y, f_small, f_big))
    adj = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    s = adj.mean() / np.sqrt(nw_var(adj, lag) / len(adj))
    return float(s), float(1 - stats.norm.cdf(s))


def block_bootstrap(y, f1, f2, block=24, B=2000, seed=C.SEED):
    """RMSE(f1) − RMSE(f2) 的移动块自助分布。"""
    rng = np.random.default_rng(seed)
    y, f1, f2 = map(np.asarray, (y, f1, f2))
    T = len(y)
    nb = int(np.ceil(T / block))
    out = np.empty(B)
    for b in range(B):
        st = rng.integers(0, T - block + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T]
        out[b] = rmse(y[idx] - f1[idx]) - rmse(y[idx] - f2[idx])
    return out


def rmse_table(pred: pd.DataFrame, cols=("zero", "base", "model", "hgb")) -> pd.DataFrame:
    rows = {"全期": {c: rmse(pred["y"] - pred[c]) for c in cols}}
    for m in (7, 8, 9):
        p = pred[pred["month"] == m]
        rows[f"{m}月"] = {c: rmse(p["y"] - p[c]) for c in cols}
    t = pd.DataFrame(rows)
    t.loc["n"] = [len(pred)] + [int((pred["month"] == m).sum()) for m in (7, 8, 9)]
    return t


def significance(pred: pd.DataFrame) -> dict:
    y, b, m = pred["y"].to_numpy(), pred["base"].to_numpy(), pred["model"].to_numpy()
    eb, em = y - b, y - m
    boot = block_bootstrap(y, b, m)
    d = eb ** 2 - em ** 2
    keep = np.abs(d) <= np.quantile(np.abs(d), 0.99)
    out = {
        "rmse_base": rmse(eb), "rmse_model": rmse(em),
        "gain": rmse(eb) - rmse(em), "gain_pct": 1 - rmse(em) / rmse(eb),
        "dm24": dm_test(eb, em, 24), "dm4": dm_test(eb, em, 4), "dm48": dm_test(eb, em, 48),
        "cw24": clark_west(y, b, m, 24),
        "boot_ci": (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))),
        "boot_p_le0": float((boot <= 0).mean()),
        "trim_gain": rmse(eb[keep]) - rmse(em[keep]),
        "month_gain": {mm: rmse(eb[pred["month"] == mm]) - rmse(em[pred["month"] == mm]) for mm in (7, 8, 9)},
        # 优势集中度：损失差最大的 1% 样本贡献了总改进的多少
        "top1_share": float(d[~keep].sum() / d.sum()) if d.sum() != 0 else np.nan,
        "dm_base_vs_zero": dm_test(y, eb, 24),
    }
    return out
