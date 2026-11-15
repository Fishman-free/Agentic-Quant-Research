# -*- coding: utf-8 -*-
"""神经网络对照：回答"更大的模型是否值得"。

协议与 models.fit_predict 完全一致：同样的 FE.ALL 特征、同样的训练/测试切分、
同样的滚动起点 CV（3–6 月四折）选超参数，测试期只评一次。
网络用 sklearn 的 MLPRegressor（L2 正则），5 个随机种子重训，报告测试 RMSE 的
均值、标准差，以及 5 个种子平均预测的 RMSE（小集成）。

结果只用于报告 §5 末尾的一句对照，不参与主结论。
"""
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C, features as FE, models as M

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HIDDEN = [(8,), (16,), (32,), (64,), (64, 32)]
ALPHAS = (1e-3, 1e-2, 1e-1)
SEEDS = tuple(C.SEED + i for i in range(5))
MAX_ITER = 5000


def _net(hidden, alpha, seed):
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=hidden, alpha=alpha, solver="adam", learning_rate_init=1e-3,
        max_iter=MAX_ITER, early_stopping=False, batch_size=256, random_state=seed))


def _cv_rmse(d, hidden, alpha) -> float:
    errs = []
    for fit, val in M.folds(d):
        m = _net(hidden, alpha, C.SEED).fit(fit[FE.ALL], fit["y"])
        errs.append(val["y"].to_numpy() - m.predict(val[FE.ALL]))
    return M.rmse(np.concatenate(errs))


def compare(sym: str, d=None) -> dict:
    """选型（四折 CV）→ 5 个种子重训 → 测试期 RMSE。"""
    d = FE.samples(sym) if d is None else d
    train, test = d[d["split"] == "train"], d[d["split"] == "test"]
    y = test["y"].to_numpy()

    t0 = time.time()
    grid = {(h, a): _cv_rmse(d, h, a) for h in HIDDEN for a in ALPHAS}
    hidden, alpha = min(grid, key=grid.get)

    rmses, preds = [], []
    for s in SEEDS:
        m = _net(hidden, alpha, s).fit(train[FE.ALL], train["y"])
        p = m.predict(test[FE.ALL])
        rmses.append(M.rmse(y - p))
        preds.append(p)
    ens = np.mean(preds, axis=0)
    nparam = sum(x.size for x in m[-1].coefs_) + sum(x.size for x in m[-1].intercepts_)

    return {
        "hidden": hidden, "alpha": alpha, "cv_best": grid[(hidden, alpha)], "cv": grid,
        "nparam": nparam, "seeds": SEEDS, "test_mean": float(np.mean(rmses)),
        "test_sd": float(np.std(rmses)), "test_min": float(np.min(rmses)),
        "test_max": float(np.max(rmses)), "test_ens": float(M.rmse(y - ens)),
        "seconds": time.time() - t0,
    }


def run_all(models_res: dict = None) -> dict:
    out = {}
    for sym in C.SYMBOLS:
        r = compare(sym, models_res[sym]["d"] if models_res else None)
        out[sym] = r
        print(f"  {sym}: 选中 hidden={r['hidden']} alpha={r['alpha']:g}，"
              f"CV {r['cv_best']:.3f}，测试 {r['test_mean']:.3f}±{r['test_sd']:.3f}"
              f"（{r['seconds']:.0f}s）", flush=True)
    return out
