# -*- coding: utf-8 -*-
"""三个预测：零、基线（OLS 回到 24h 均值）、我的模型（Ridge + 特征工程 + 状态交互）。

调参只用训练期：滚动起点交叉验证——3、4、5、6 月依次做验证月，每折只用 t+4h ≤ 该月月初的
样本训练（自然隔开 4h），四折验证误差合并算 RMSE。只用 6 月一个月选型太容易碰运气。
在 CV 上选 Ridge 的 α，并按特征组做向后剔除（"偏离"组含基线变量，必须保留）。
选定后用全部训练样本重估，再在 7–9 月测试期一次性评估。
HistGradientBoosting 只作稳健性对照（看非线性能否再挖出东西），不参与主结论。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C, features as FE

ALPHAS = [0.1, 1, 3, 10, 30, 100, 300, 1000, 3000]


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2)))


CV_MONTHS = (3, 4, 5, 6)


def folds(d: pd.DataFrame):
    tr = d[d["split"] == "train"]
    for m in CV_MONTHS:
        start = pd.Timestamp(f"2026-{m:02d}-01")
        yield tr[tr.index + C.H * FE.HOUR <= start], tr[(tr.index >= start) & (tr["month"] == m)]


def inner_split(d: pd.DataFrame):
    """兼容：最后一折（6 月验证）。"""
    return list(folds(d))[-1]


def cv_rmse(d: pd.DataFrame, feats, alpha) -> float:
    """计算指定特征子集与正则化参数 alpha 下的滚动时序交叉验证 RMSE。
    
    参数：
        d: 包含训练与验证样本的面板数据
        feats: 候选特征名称列表
        alpha: Ridge 模型的 L2 正则化系数
    返回：
        合并四折验证集后的全期样本外预测 RMSE（bp）
    """
    errs = []
    for fit, val in folds(d):
        errs.append(val["y"].to_numpy() - ridge(alpha).fit(fit[feats], fit["y"]).predict(val[feats]))
    return rmse(np.concatenate(errs))


def cv_baseline(d: pd.DataFrame) -> float:
    """计算简单均值回归基线在四折滚动验证集上的 RMSE。"""
    return rmse(np.concatenate([val["y"] - baseline_predict(val, baseline_fit(fit)) for fit, val in folds(d)]))


def best_alpha(d, feats):
    """在预设网格中搜索时序交叉验证 RMSE 最小的 Ridge L2 正则化系数 alpha。"""
    sc = {a: cv_rmse(d, feats, a) for a in ALPHAS}
    a = min(sc, key=sc.get)
    return a, sc[a], sc


def select_groups(d: pd.DataFrame):
    """基于验证集 RMSE 的向后特征组贪心剔除算法。
    
    原则：
        - "偏离"组包含基准变量，作为基础核心特征强制保留；
        - 每轮遍历剔除剩余的一组特征，若验证集 RMSE 改善则执行剔除，直到剔除任何组均无法提升泛化性能为止。
    """
    groups = list(FE.GROUPS)
    feats = lambda gs: [f for g in gs for f in FE.GROUPS[g]]
    a, cur, _ = best_alpha(d, feats(groups))
    path = [("全部", cur)]
    while True:
        trials = {}
        for g in groups:
            if g == "偏离":
                continue
            gs = [x for x in groups if x != g]
            trials[g] = best_alpha(d, feats(gs))
        if not trials:
            break
        g, (a2, sc2, _) = min(trials.items(), key=lambda kv: kv[1][1])
        if sc2 >= cur:
            break
        groups.remove(g)
        a, cur = a2, sc2
        path.append((f"去掉{g}", cur))
    return groups, feats(groups), a, cur, path


def baseline_fit(train: pd.DataFrame):
    """拟合均值回归基准模型：y_t = alpha + beta * d_t (OLS 最小二乘法)"""
    X = np.c_[np.ones(len(train)), train["dev_24h"]]
    coef, *_ = np.linalg.lstsq(X, train["y"].to_numpy(), rcond=None)
    return float(coef[0]), float(coef[1])


def baseline_predict(d, ab):
    """基于基准模型参数输出 4 小时基差变动预测值"""
    return ab[0] + ab[1] * d["dev_24h"].to_numpy()


def ridge(alpha):
    """构造包含标准化（StandardScaler）与 L2 正则化（Ridge）的预测管道"""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def ablation(d: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """特征重要性消融分析：
    计算各特征组被单项剔除后导致的验证集 RMSE 恶化幅度，以及仅保留该特征组时的独立预测改善，
    用于科学评估每个维度的经济学信息贡献。
    """
    full = cv_rmse(d, FE.ALL, alpha)
    base = cv_baseline(d)
    rows = []
    for g, cols in FE.GROUPS.items():
        keep = [c for c in FE.ALL if c not in cols]
        only = sorted(set(cols) | {"dev_24h"})
        drop, add = cv_rmse(d, keep, alpha), cv_rmse(d, only, alpha)
        rows.append({"组": g, "剔除后CV": drop, "剔除损失": drop - full,
                     "基线+本组CV": add, "相对基线增益": base - add})
    out = pd.DataFrame(rows)
    out.attrs.update(full=full, base=base)
    return out


def hgb_select(d: pd.DataFrame):
    fit, val = inner_split(d)  # 树模型只作对照，用 6 月一折选迭代次数即可
    best = None
    for lr in (0.03, 0.1):
        for depth in (2, 3):
            m = HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=600,
                                              l2_regularization=1.0, min_samples_leaf=40,
                                              early_stopping=False, random_state=C.SEED)
            m.fit(fit[FE.ALL], fit["y"])
            # 逐轮评估验证误差，选迭代次数
            errs = [rmse(val["y"] - p) for p in m.staged_predict(val[FE.ALL])]
            k = int(np.argmin(errs))
            if best is None or errs[k] < best[0]:
                best = (errs[k], dict(learning_rate=lr, max_depth=depth, max_iter=k + 1))
    return best[1]


def fit_predict(sym: str, d: pd.DataFrame = None) -> dict:
    d = FE.samples(sym) if d is None else d
    train, test = d[d["split"] == "train"], d[d["split"] == "test"]
    ab = baseline_fit(train)
    groups, feats, alpha, cv_best, path = select_groups(d)
    model = ridge(alpha).fit(train[feats], train["y"])
    hp = hgb_select(d)
    hgb = HistGradientBoostingRegressor(l2_regularization=1.0, min_samples_leaf=40,
                                        early_stopping=False, random_state=C.SEED, **hp)
    hgb.fit(train[FE.ALL], train["y"])
    full_alpha = best_alpha(d, FE.ALL)[0]
    model_all = ridge(full_alpha).fit(train[FE.ALL], train["y"])
    pred = pd.DataFrame({"y": test["y"], "zero": 0.0, "base": baseline_predict(test, ab),
                         "model": model.predict(test[feats]), "model_all": model_all.predict(test[FE.ALL]),
                         "hgb": hgb.predict(test[FE.ALL]), "month": test["month"]}, index=test.index)
    coefs = pd.Series(model[-1].coef_ / model[0].scale_, index=feats)  # 换回原始单位
    return {"d": d, "train": train, "test": test, "ab": ab, "alpha": alpha, "groups": groups,
            "feats": feats, "cv_best": cv_best, "cv_path": path, "cv_base": cv_baseline(d),
            "model": model, "coef_raw": coefs, "hgb_params": hp,
            "pred": pred, "ablation": ablation(d, full_alpha)}
