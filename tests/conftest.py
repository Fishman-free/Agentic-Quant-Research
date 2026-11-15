import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqr import config as C  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if not (C.HIST_DIR.exists() and C.BN_FUT.exists()):
        skip = pytest.mark.skip(reason="原始数据不在 data/（先运行 scripts/download_datasets.py）")
        for it in items:
            if "needs_data" in it.keywords:
                it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: 需要 data/ 下的原始 zip")
