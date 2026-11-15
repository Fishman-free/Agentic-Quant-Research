# -*- coding: utf-8 -*-
"""时钟换算、HistData 合并规则、核对时刻落在哪根 K 线。"""
import numpy as np
import pandas as pd
import pytest

from aqr import clock, io_raw as R, panel


def test_local_to_utc_rule():
    lab = pd.DatetimeIndex(["2026-01-14 09:59", "2026-03-27 16:58", "2026-03-29 18:00", "2026-08-19 09:59"])
    got = clock.local_to_utc_end(lab)
    want = pd.DatetimeIndex(["2026-01-14 15:00", "2026-03-27 21:59", "2026-03-29 22:01", "2026-08-19 14:00"])
    assert (got == want).all()


@pytest.mark.parametrize("t", ["2026-01-14 15:00", "2026-08-19 14:00"])
def test_check_times_map_to_0959_bar(t):
    assert clock.utc_end_to_local_label(pd.Timestamp(t)).strftime("%H:%M") == "09:59"


def test_merge_rule_on_synthetic_rows():
    raw = pd.DataFrame({"local": pd.to_datetime(["2026-07-01 10:00"] * 3 + ["2026-07-01 10:01"]),
                        "open": [1.0, 2.0, 3.0, 9.0], "high": [5.0, 7.0, 4.0, 9.0], "low": [0.5, 0.2, 0.9, 9.0],
                        "close": [4.0, 6.0, 2.5, 9.0], "volume": 0, "file_month": 7, "row": [0, 1, 2, 3]})
    m = R.merge_histdata(raw).loc[pd.Timestamp("2026-07-01 10:00")]
    assert (m["open"], m["high"], m["low"], m["close"], m["n_rows"]) == (1.0, 7.0, 0.2, 2.5, 3)


@pytest.mark.needs_data
def test_offset_scan_switches_exactly_once():
    hd = R.merge_histdata(R.load_histdata_raw())
    sc = clock.offset_scan(hd["close"], R.load_klines("perp", "XAUUSDT")["close"])
    assert (sc.loc[:"2026-03-27", "best"] == 300).all()
    assert (sc.loc["2026-03-30":, "best"] == 240).all()
    # 相邻偏移（±1 分钟）几乎不相关：标签就是分钟起点，没有半分钟的模糊
    assert np.nanmedian(np.abs(sc["corr_p1"])) < 0.1 and sc["corr"].median() > 0.95


@pytest.mark.needs_data
def test_panel_alignment_no_duplicates():
    p = panel.minute_panel("XAUUSDT")
    assert p.index.is_unique and p.index.is_monotonic_increasing
    assert p.index[0] == pd.Timestamp("2026-01-01 00:01") and p.index[-1] == pd.Timestamp("2026-10-01 00:00")
    # 06-29 由日度补丁补回
    assert p.loc["2026-06-29", "index"].notna().all()
