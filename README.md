# 资金费能把黄金永续拉回现货吗？

AGENTIC 多资产量化研究实习申请项目 · Zhuqi Ma

- **报告**：[`report/main.pdf`](report/main.pdf)（正文 5 页，附录 3 页：下一步 / 核对数字 / AI 使用说明）
- **题目**：[`screening_test_资金费能把黄金永续拉回现货吗.md`](screening_test_资金费能把黄金永续拉回现货吗.md)
- **隐藏数据问题完整清单**：[`docs/DATA_ISSUES.md`](docs/DATA_ISSUES.md)

## 阅读顺序

**先读报告。** 报告按论文结构组织，每一节回答一个问题：

| 报告章节 | 回答什么 | 对应代码 |
|---|---|---|
| 摘要、§1 引言、§2 相关工作 | 研究什么问题，分哪五步，和已有研究的关系 | — |
| §3 数据与对齐 | 四个价格怎样放到同一个 UTC 时钟上；HistData 时间戳的换算规则及证据；题面没列出的数据问题 | `io_raw.py` → `clock.py` → `panel.py` → `dataqa.py` |
| §4 资金费机制复现 | 资金费公式和死区；参数怎样随时间变化；逐期重算与官方值比对 | `funding_params.py` → `funding.py` |
| §5 预测模型 | 未来 4 小时基差变化的预测；特征为什么这样设计；比基线好多少，是不是运气；神经网络为什么没有更好 | `features.py` → `models.py` → `mlp.py` → `evaluate.py` |
| §6 回归机制 | 偏离多久回来；是永续在动还是现货在动；资金费什么时候在拉 | `dynamics.py` |
| §7 深挖休市 | 现货休市时资金费和永续价格发生了什么 | `calendar.py` → `closure.py` |
| §8 结论 | 把前面的结果合起来回答研究问题 | — |
| 附录 A / B / C | 下一步研究方向 / 核对数字 / AI 使用说明 | `report_numbers.py` |

**再看代码。** 入口是 [`run_all.py`](run_all.py)，它按上表的顺序调用各模块，每一步打印一行进度。所有代码在 `src/aqr/` 下，每个文件开头的说明写了它做什么、为什么这样做。图在 `figures.py`，报告里的每个数字由 `report_numbers.py` 写成 LaTeX 宏。

**最后看验证。** `tests/` 是自动测试（时钟、资金费边界期、防前视等），`scripts/verify_check_numbers.py` 不经过分析代码、直接读原始文件复核附录 B 的数字。

## 一条命令复现

```bash
pip install -r requirements.txt
python scripts/download_datasets.py   # 约 80 MB，幂等；自动补拉月度归档缺失的整天
python run_all.py                     # 原始 zip → 全部数字、表格、图 → report/main.pdf（约 6 分钟，其中神经网络对照约 5 分钟）
```

- `python run_all.py --rebuild`：删掉 `cache/`，从原始 zip 重建（缓存只用来加速）
- `python run_all.py --no-pdf`：只算数字和图，不调用 xelatex
- `python run_all.py --no-mlp`：跳过神经网络对照，沿用 `outputs/summary.json` 里上次的结果（约 1 分钟，PDF 与完整运行一致）
- `python -m pytest -q`：23 项测试，覆盖时钟、合并规则、资金费边界期与上限期、截断不变性（防前视），还有一个故意偷看未来的特征，测试必须能抓到它
- `python scripts/verify_check_numbers.py`：只用标准库直接读原始 zip，独立复核"核对数字"附录

报告正文里的数字**全部**由 `src/aqr/report_numbers.py` 写进 `report/generated/numbers.tex`（270 个宏），正文中不手写数字。

## 依赖

Python 3.14.5；numpy 2.4.6 · pandas 2.3.3 · pyarrow 24.0.0 · scipy 1.18.0 · statsmodels 0.14.6 · scikit-learn 1.9.0 · matplotlib 3.11.0 · requests 2.34.2 · pytest 9.0.3（见 `requirements.txt`）。
编译 PDF 需要 TeX Live（xelatex + ctex，已在 TeX Live 2026 上测试）。中文字体由 ctex 按操作系统自动选择；图中中文优先使用微软雅黑，找不到时回退到 Noto/思源黑体。

## 数据存放位置

数据放在 `data/` 下，不入库。也可以用环境变量 `AQR_DATA_DIR` 指向其他位置。目录与官方文件名保持一致：

```
data/
├── HistData真实金价/XAUUSD一分钟_GenericASCII/HISTDATA_COM_ASCII_XAUUSD_M1_2026MM.zip (+ _status.txt)
└── 币安Binance/
    ├── 现货市场/一分钟K线_klines/PAXGUSDT_PAXG代币现货/PAXGUSDT-1m-2026-MM.zip
    └── 永续合约市场_USDM/
        ├── 一分钟K线_klines/{XAUUSDT_黄金永续,PAXGUSDT_黄金代币永续}/<SYM>-1m-2026-MM.zip
        ├── 一分钟参考指数_indexPriceKlines/…（同上；日度补丁_daily/<SYM>-1m-2026-06-29.zip）
        ├── 一分钟溢价指数_premiumIndexKlines/…（同上；日度补丁_daily/…）
        └── 资金费历史_fundingRate/…/<SYM>-fundingRate-2026-MM.zip
```

## 代码结构

| 模块 | 作用 |
|---|---|
| `src/aqr/io_raw.py` | 读原始 zip，统一用"UTC 分钟结束时刻"作索引（现货时间戳为微秒、期货为毫秒）；HistData 同一分钟多行按题面规则合并 |
| `src/aqr/clock.py` | HistData → UTC：逐日偏移扫描与日休作息，两条独立证据 |
| `src/aqr/calendar.py` | 事先可知的外汇黄金交易日历，供特征使用，不看实际数据缺口 |
| `src/aqr/panel.py` | 一分钟面板；基差拆成溢价（永续−指数）与指数差（指数−现货） |
| `src/aqr/funding_params.py` | **时变资金费参数表**（利息项、上下限，按结算期分段生效，每段附证据） |
| `src/aqr/funding.py` | 逐期复现、$\bar P$ 近似方法对比、误差包络、不匹配期的归因 |
| `src/aqr/features.py` · `models.py` · `mlp.py` · `evaluate.py` | 样本与特征、基线 / Ridge / 梯度提升树 / 神经网络对照、滚动交叉验证选型、DM / Clark–West / 块自助法 |
| `src/aqr/dynamics.py` | 半衰期、工具变量去噪、局部投影、永续与现货谁在动（相对第三市场）、结算事件研究 |
| `src/aqr/closure.py` | 深挖角度：现货休市与 05-08 指数规则变更的自然实验、开盘缺口的预测 |
| `src/aqr/dataqa.py` | 隐藏数据问题：检测、证据、处理方式，以及对结论数字的影响 |
| `src/aqr/figures.py` · `report_numbers.py` | 图（配色经色盲安全校验）、LaTeX 宏与表格 |

## 主要结论（详见报告）

- **时钟**：HistData 的标签是分钟起点。03-27 之前时区为 UTC−5，03-29 起为 UTC−4（欧洲夏令时切换的那个周末）。它既不是"恒为 EST"，也不按美国夏令时切换。如果按"恒为 UTC−5"处理，XAU 测试期基线 RMSE 会从 2.28 bp 膨胀到 29.1 bp。
- **资金费复现**：两个合约各 1636 期。用时变参数表复现，XAU 最大误差 0.749 bp，误差超过 0.1 bp 的有 4 期；PAXG 最大误差 0.081 bp，没有超过 0.1 bp 的期。如果用样本末的参数回填全部 9 个月，XAU 会有 89 期误差超过 0.1 bp，最大 4.92 bp。
- **预测**：测试期 RMSE 从基线降到模型：XAU 2.275 → 2.166（DM 检验 p=0.0007），PAXG 1.226 → 1.197（p=0.03）。
- **谁向谁靠**：基差回归主要是现货追上永续。XAU 永续在 4 小时回归中只承担 16%。资金费的作用体现为结算前后的"锯齿"，结算后 5 分钟内溢价弹回 1.8 bp。
- **拉不动的情形**：一是溢价落在 ±0.05% 死区内，PAXG 的溢价长期停在死区下沿；二是现货休市，05-08 之后休市期间的资金费恒为 0；三是参考指数与真实金价本身就有错位，比如 07-16 之后的 USDT/USD 偏离。
