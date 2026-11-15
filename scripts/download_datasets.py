# -*- coding: utf-8 -*-
"""
筛选题《资金费能把黄金永续拉回现货吗》数据集下载脚本
区间：2026-01-01 00:00 至 2026-10-01 00:00 (UTC) -> 月度归档 2026-01 .. 2026-09

来源：
  1. HistData.com        XAU/USD 一分钟 Generic ASCII（含状态说明文件 getStatus.php）
  2. data.binance.vision 现货 PAXGUSDT 1m klines
  3. data.binance.vision USDⓈ-M 期货 XAUUSDT / PAXGUSDT：
     klines / indexPriceKlines / premiumIndexKlines / fundingRate
  4. 日度补丁：月度归档缺整天时（实测 2026-06 的参考指数/溢价指数缺 06-29），
     从 data.binance.vision 的 daily 归档补拉该日，放在同目录的 日度补丁_daily/ 下

用法：python scripts/download_datasets.py   （幂等：已存在且非空的文件自动跳过）
数据目录：<仓库>/data，可用环境变量 AQR_DATA_DIR 覆盖
"""
import calendar
import hashlib
import io
import os
import re
import sys
import time
import zipfile

import requests

ROOT = os.environ.get("AQR_DATA_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]  # 2026-01 .. 2026-09
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

S = requests.Session()
S.headers.update(UA)

ok, skipped, failed = [], [], []


def fetch(url, tries=3, timeout=120, stream=False):
    last = None
    for i in range(tries):
        try:
            r = S.get(url, timeout=timeout, stream=stream)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code == 404:
                return r  # 不重试
        except Exception as e:  # 网络抖动重试
            last = str(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET {url} 失败: {last}")


def save_stream(url, dest):
    """下载到 dest，已存在且非空则跳过。返回 'ok'/'skip'。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = fetch(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    os.replace(tmp, dest)
    return "ok"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_verified(url, rel, tag):
    """下载 zip + .CHECKSUM 并校验 SHA256。"""
    dest = os.path.join(ROOT, rel)
    try:
        state = save_stream(url, dest)
        save_stream(url + ".CHECKSUM", dest + ".CHECKSUM")
        want = open(dest + ".CHECKSUM", encoding="utf-8").read().split()[0].lower()
        got = sha256_of(dest)
        if got != want:
            failed.append((rel, f"SHA256 不匹配 want={want} got={got}"))
            print(f"  {tag} 校验失败 {rel}", flush=True)
            return
        (skipped if state == "skip" else ok).append(rel)
        print(f"  {tag} {state.upper():5s} {rel}  ({os.path.getsize(dest)//1024} KB)", flush=True)
    except Exception as e:
        failed.append((rel, str(e)))
        print(f"  {tag} FAIL  {rel}: {e}", flush=True)


# ---------------------------------------------------------------- Binance
# 本地目录已中文化（见 data/数据说明.md），zip 保持官方原名
SYM_DIR = {"XAUUSDT": "XAUUSDT_黄金永续",
           "PAXGUSDT": "PAXGUSDT_黄金代币永续"}
TYP_DIR = {"klines": "一分钟K线_klines",
           "indexPriceKlines": "一分钟参考指数_indexPriceKlines",
           "premiumIndexKlines": "一分钟溢价指数_premiumIndexKlines"}
FUT_ROOT = "币安Binance/永续合约市场_USDM"
SPOT_REL = "币安Binance/现货市场/一分钟K线_klines/PAXGUSDT_PAXG代币现货"
DAILY_SUB = "日度补丁_daily"


def binance_jobs():
    jobs = []
    for m in MONTHS:
        jobs.append((
            f"https://data.binance.vision/data/spot/monthly/klines/PAXGUSDT/1m/PAXGUSDT-1m-{m}.zip",
            f"{SPOT_REL}/PAXGUSDT-1m-{m}.zip"))
    for sym in ("XAUUSDT", "PAXGUSDT"):
        for typ in ("klines", "indexPriceKlines", "premiumIndexKlines"):
            for m in MONTHS:
                jobs.append((
                    f"https://data.binance.vision/data/futures/um/monthly/{typ}/{sym}/1m/{sym}-1m-{m}.zip",
                    f"{FUT_ROOT}/{TYP_DIR[typ]}/{SYM_DIR[sym]}/{sym}-1m-{m}.zip"))
        for m in MONTHS:
            jobs.append((
                f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip",
                f"{FUT_ROOT}/资金费历史_fundingRate/{SYM_DIR[sym]}/{sym}-fundingRate-{m}.zip"))
    return jobs


def download_binance():
    jobs = binance_jobs()
    print(f"[binance] 共 {len(jobs)} 个 zip（含 .CHECKSUM 校验）", flush=True)
    for i, (url, rel) in enumerate(jobs, 1):
        save_verified(url, rel, f"[{i}/{len(jobs)}]")


def missing_days(zip_path, month):
    """月度 1m 归档里没有任何一行的日期（YYYY-MM-DD）。"""
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(z.namelist()[0])
    days = set()
    for line in io.BytesIO(raw):
        head = line.split(b",", 1)[0]
        if not head[:1].isdigit():
            continue  # 期货文件有表头
        ts = int(head)
        ts = ts // 1000 if ts > 10 ** 14 else ts  # 现货自 2025 年起为微秒
        days.add(time.strftime("%Y-%m-%d", time.gmtime(ts / 1000)))
    y, m = map(int, month.split("-"))
    want = {f"{month}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)}
    return sorted(want - days)


def download_daily_patches():
    """月度归档缺整天 -> 从 daily 归档补拉（同样做 SHA256 校验）。"""
    print("[binance-daily] 扫描月度归档缺失的整天", flush=True)
    targets = [("spot", "PAXGUSDT", "klines", SPOT_REL)]
    for sym in SYM_DIR:
        for typ in TYP_DIR:
            targets.append(("um", sym, typ, f"{FUT_ROOT}/{TYP_DIR[typ]}/{SYM_DIR[sym]}"))
    for market, sym, typ, rel_dir in targets:
        for m in MONTHS:
            zp = os.path.join(ROOT, rel_dir, f"{sym}-1m-{m}.zip")
            if not os.path.exists(zp):
                continue
            for day in missing_days(zp, m):
                base = ("https://data.binance.vision/data/spot/daily/klines" if market == "spot"
                        else f"https://data.binance.vision/data/futures/um/daily/{typ}")
                save_verified(f"{base}/{sym}/1m/{sym}-1m-{day}.zip",
                              f"{rel_dir}/{DAILY_SUB}/{sym}-1m-{day}.zip", "[daily]")


# ---------------------------------------------------------------- HistData
def parse_hidden_fields(html):
    fields = {}
    for name in ("tk", "date", "datemonth", "platform", "timeframe", "fxpair"):
        m = re.search(rf'name="{name}"[^>]*value="([^"]*)"', html) or \
            re.search(rf'id="{name}"[^>]*value="([^"]*)"', html)
        if m:
            fields[name] = m.group(1)
    return fields


def download_histdata():
    print(f"[histdata] XAU/USD M1 Generic ASCII 2026-01 .. 2026-09", flush=True)
    for mm in range(1, 10):
        rel = f"HistData真实金价/XAUUSD一分钟_GenericASCII/HISTDATA_COM_ASCII_XAUUSD_M1_2026{mm:02d}.zip"
        dest = os.path.join(ROOT, rel)
        page_url = (f"https://www.histdata.com/download-free-forex-historical-data/"
                    f"?/ascii/1-minute-bar-quotes/xauusd/2026/{mm}")
        try:
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                skipped.append(rel)
                print(f"  [{mm}/9] SKIP  {rel}", flush=True)
            else:
                html = fetch(page_url).text
                fields = parse_hidden_fields(html)
                need = {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}
                if set(fields) != need:
                    raise RuntimeError(f"表单字段不全: {fields}")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # get.php 校验 Referer 防盗链，必须带上下载页来源
                hd = {"Referer": page_url, "Origin": "https://www.histdata.com"}
                r = S.post("https://www.histdata.com/get.php", data=fields,
                           headers=hd, timeout=120)
                if r.status_code != 200 or not r.content.startswith(b"PK"):
                    raise RuntimeError(f"POST get.php 返回异常 status={r.status_code} "
                                       f"head={r.content[:40]!r}")
                with open(dest, "wb") as f:
                    f.write(r.content)
                ok.append(rel)
                print(f"  [{mm}/9] OK    {rel}  ({len(r.content)//1024} KB)", flush=True)
            # 数据状态说明文件（列缺失区间，对"数据质量"分析有用）
            srel = rel.replace(".zip", "_status.txt")
            sdest = os.path.join(ROOT, srel)
            if not (os.path.exists(sdest) and os.path.getsize(sdest) > 0):
                fields = parse_hidden_fields(fetch(page_url).text)
                r = S.post("https://www.histdata.com/getStatus.php", data=fields,
                           headers={"Referer": page_url, "Origin": "https://www.histdata.com"},
                           timeout=60)
                if r.status_code == 200 and not r.content.startswith(b"<"):
                    with open(sdest, "wb") as f:
                        f.write(r.content)
                    ok.append(srel)
            time.sleep(0.8)  # 对免费站点保持礼貌
        except Exception as e:
            failed.append((rel, str(e)))
            print(f"  [{mm}/9] FAIL  {rel}: {e}", flush=True)


# ---------------------------------------------------------------- 校验
def verify_zips():
    print("\n[verify] zip CRC 与内容抽查", flush=True)
    bad = 0
    for dirpath, _, files in os.walk(ROOT):
        for fn in sorted(files):
            if not fn.endswith(".zip"):
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, ROOT)
            try:
                with zipfile.ZipFile(p) as z:
                    corrupt = z.testzip()
                    if corrupt:
                        print(f"  BAD  {rel}: CRC 损坏于 {corrupt}", flush=True)
                        bad += 1
            except Exception as e:
                print(f"  BAD  {rel}: {e}", flush=True)
                bad += 1
    return bad


if __name__ == "__main__":
    t0 = time.time()
    download_binance()
    download_daily_patches()
    download_histdata()
    bad = verify_zips()
    print(f"\n==== 汇总 ====", flush=True)
    print(f"新下载 {len(ok)} · 跳过(已存在) {len(skipped)} · 失败 {len(failed)} · 损坏 zip {bad}"
          f" · 耗时 {time.time()-t0:.0f}s", flush=True)
    for rel, why in failed:
        print(f"  FAIL {rel}: {why}", flush=True)
    sys.exit(1 if (failed or bad) else 0)
