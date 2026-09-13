#!/usr/bin/env python3
"""
actions_fetch.py — v3 (Sep 13). Automated Market Data Fetcher.
Runs on GitHub Actions runner or local machine, commits results to docs/Market files/.
Resumable across runs: existing files are skipped or topped-up.

Includes:
  1. Futures UDiFF patch / top-up
  2. NSE Promoter Pledge disclosures (SAST 31(1)/(2) Oct 2019 – Present)
  3. NSE SLB (Securities Lending & Borrowing) daily snapshot & series data
  4. Chittorgarh Mainboard IPO listing dates & Anchor lock-in schedules (2022–2026)
  5. RBI Daily Money Market Operations series (Net LAF, WACR Oct 2019 – Present)
  6. NSE Surveillance current lists (ASM / GSM)
"""

import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from scripts.download_market_datasets import (
    fetch_chittorgarh_ipos,
    fetch_pledge_disclosures,
    fetch_rbi_mmo_series,
    fetch_slb_data,
    get_nse_session,
)

BASE = "docs/Market files"
os.makedirs(BASE, exist_ok=True)
REPORT = []


def log(s):
    print(s)
    REPORT.append(f"- {s}")


def weekdays(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ---------------------------------------------------------- 1. Futures Top-up
def futures_patch():
    out = f"{BASE}/fo_futures"
    os.makedirs(out, exist_ok=True)
    yfile = f"{out}/futures_{date.today().year}.csv.gz"
    S = get_nse_session()
    pat = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
    start = date(2024, 7, 8)
    if os.path.exists(yfile):
        start = date.today() - timedelta(days=14)
    got = 0
    by_year = {}
    for d in weekdays(start, date.today() - timedelta(days=1)):
        tag = d.strftime("%Y%m%d")
        try:
            r = S.get(pat.format(d=tag), timeout=25)
            if r.status_code == 200 and len(r.content) > 1000:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                col = next((c for c in df.columns if c.strip().upper() == "FININSTRMTP"), None)
                if col is not None:
                    by_year.setdefault(d.year, []).append(
                        df[df[col].astype(str).str.upper().isin(["STF", "IDF"])]
                    )
                    got += 1
        except Exception:
            pass
        time.sleep(0.4)
    for y, ch in by_year.items():
        new = pd.concat(ch, ignore_index=True)
        sfx = "_udiff" if y == 2024 else ""
        fp = f"{out}/futures_{y}{sfx}.csv.gz"
        if os.path.exists(fp):
            old = pd.read_csv(fp, low_memory=False)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates()
        new.to_csv(fp, index=False, compression="gzip")
    log(f"futures: {got} days fetched/topped-up")


# ---------------------------------------------------------- 2. Pledge Disclosures (SAST 31(1)/(2))
def run_pledge():
    try:
        fetch_pledge_disclosures(start_date=date(2019, 10, 1))
        csv_path = f"{BASE}/pledge/pledge_disclosures_2019_2026.csv.gz"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log(f"pledge SAST 31(1)/(2): {len(df)} total historical disclosures on disk")
        else:
            log("pledge: completed with no records")
    except Exception as e:
        log(f"pledge failed: {e}")


# ---------------------------------------------------------- 3. SLB Daily Data
def run_slb():
    try:
        fetch_slb_data()
        csv_path = f"{BASE}/slb/slb_snapshot_latest.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log(f"slb: {len(df)} active quotes saved across series")
        else:
            log("slb: completed")
    except Exception as e:
        log(f"slb failed: {e}")


# ---------------------------------------------------------- 4. Chittorgarh IPOs & Lockups
def run_lockups():
    try:
        fetch_chittorgarh_ipos(years=(2022, 2023, 2024, 2025, 2026))
        csv_path = f"{BASE}/lockups/chittorgarh_ipo_listings_2022_2026.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log(f"lockups: {len(df)} mainboard IPO listings & anchor lock-in schedules saved (2022-2026)")
        else:
            log("lockups: completed")
    except Exception as e:
        log(f"lockups failed: {e}")


# ---------------------------------------------------------- 5. Surveillance (ASM / GSM)
def surveillance():
    out = f"{BASE}/surveillance"
    os.makedirs(out, exist_ok=True)
    S = get_nse_session()
    for name, url in {
        "asm_current": "https://www.nseindia.com/api/reportASM",
        "gsm_current": "https://www.nseindia.com/api/reportGSM",
    }.items():
        fp = f"{out}/{name}_{date.today():%Y%m%d}.json"
        if not os.path.exists(fp):
            try:
                r = S.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 50:
                    with open(fp, "wb") as f:
                        f.write(r.content)
                    log(f"surveillance: {name} saved")
            except Exception as e:
                log(f"surveillance {name} failed: {e}")
            time.sleep(1)


# ---------------------------------------------------------- 6. RBI MMO Series
def run_rbi_mmo():
    try:
        fetch_rbi_mmo_series(min_prid=48250, max_prid=65100, max_workers=15)
        csv_path = f"{BASE}/rbi_mmo/rbi_mmo_daily_series.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            log(f"rbi_mmo: {len(df)} daily Money Market Operations series rows on disk")
        else:
            log("rbi_mmo: completed")
    except Exception as e:
        log(f"rbi_mmo failed: {e}")


if __name__ == "__main__":
    for fn in (futures_patch, run_pledge, run_slb, run_lockups, surveillance, run_rbi_mmo):
        try:
            fn()
        except Exception as e:
            log(f"SECTION FAILED {fn.__name__}: {type(e).__name__}: {e}")
    with open(f"{BASE}/FETCH_REPORT.md", "w") as f:
        f.write(
            f"# Fetch report v3 — {datetime.utcnow():%Y-%m-%d %H:%M} UTC\n\n"
            + "\n".join(REPORT)
            + "\n"
        )
    print("\nREPORT SUMMARY:\n" + "\n".join(REPORT))
