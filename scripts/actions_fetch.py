#!/usr/bin/env python3
"""
actions_fetch.py — runs on a GitHub Actions runner (unrestricted network).
Fetches everything Phase 2 still needs and writes into docs/Market files/.
Every section fails soft: whatever succeeds is committed; a status report
(docs/Market files/FETCH_REPORT.md) records exactly what worked.
Resumable across runs: existing files are skipped.
"""

import os, io, time, json, zipfile
from datetime import date, timedelta, datetime
import requests, pandas as pd

BASE = "docs/Market files"
os.makedirs(BASE, exist_ok=True)
REPORT = []
def log(s):
    print(s); REPORT.append(f"- {s}")

def session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/"})
    for warm in ("https://www.nseindia.com",
                 "https://www.nseindia.com/market-data/securities-lending-and-borrowing"):
        try: s.get(warm, timeout=20)
        except requests.RequestException: pass
        time.sleep(1)
    return s

S = session()

def weekdays(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5: yield d
        d += timedelta(days=1)

# ---------------------------------------------------------------- 1. futures (UDiFF era)
def futures_patch():
    out = f"{BASE}/fo_futures"; os.makedirs(out, exist_ok=True)
    have_2025 = os.path.exists(f"{out}/futures_2025.csv.gz")
    start = date(2024, 7, 8)
    end = date.today() - timedelta(days=1)
    pat = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
    by_year, got, miss = {}, 0, 0
    if have_2025:
        log("futures: patch files already present, skipping"); return
    for d in weekdays(start, end):
        tag = d.strftime("%Y%m%d")
        try:
            r = S.get(pat.format(d=tag), timeout=25)
            if r.status_code == 200 and len(r.content) > 1000:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                col = next((c for c in df.columns if c.strip().upper()=="FININSTRMTP"), None)
                if col is not None:
                    fut = df[df[col].astype(str).str.upper().isin(["STF","IDF"])]
                    by_year.setdefault(d.year, []).append(fut); got += 1
                else: miss += 1
        except Exception: miss += 1
        time.sleep(0.4)
    for y, ch in by_year.items():
        allr = pd.concat(ch, ignore_index=True)
        sfx = "_udiff" if y == 2024 else ""
        allr.to_csv(f"{out}/futures_{y}{sfx}.csv.gz", index=False, compression="gzip")
    log(f"futures UDiFF patch: {got} days fetched, {miss} missed -> {sorted(by_year)}")

# ---------------------------------------------------------------- 2. pledge history (API)
def pledge_history():
    out = f"{BASE}/pledge"; os.makedirs(out, exist_ok=True)
    url = "https://www.nseindia.com/api/corporate-pledgedata?index=equities&from_date={a}&to_date={b}"
    start, end = date(2019, 10, 1), date.today()
    got = 0
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=89), end)
        fp = f"{out}/pledge_{cur:%Y%m%d}_{nxt:%Y%m%d}.json"
        if not os.path.exists(fp):
            try:
                r = S.get(url.format(a=cur.strftime("%d-%m-%Y"),
                    b=nxt.strftime("%d-%m-%Y")), timeout=30)
                if r.status_code == 200 and len(r.content) > 50:
                    open(fp, "wb").write(r.content); got += 1
            except Exception as e:
                log(f"pledge window {cur} failed: {type(e).__name__}")
            time.sleep(1.2)
        cur = nxt + timedelta(days=1)
    n = len([f for f in os.listdir(out) if f.startswith('pledge_')])
    log(f"pledge: {got} new windows this run; {n} windows on disk (90-day chunks since Oct 2019)")

# ---------------------------------------------------------------- 3. surveillance lists
def surveillance():
    out = f"{BASE}/surveillance"; os.makedirs(out, exist_ok=True)
    endpoints = {
        "asm_current": "https://www.nseindia.com/api/reportASM",
        "gsm_current": "https://www.nseindia.com/api/reportGSM",
    }
    for name, url in endpoints.items():
        try:
            r = S.get(url, timeout=25)
            if r.status_code == 200 and len(r.content) > 50:
                open(f"{out}/{name}_{date.today():%Y%m%d}.json", "wb").write(r.content)
                log(f"surveillance: {name} saved")
            else:
                log(f"surveillance: {name} HTTP {r.status_code}")
        except Exception as e:
            log(f"surveillance: {name} failed ({type(e).__name__})")
        time.sleep(1)
    # historical: REG_IND-style daily surveillance indicator archives
    pat = "https://nsearchives.nseindia.com/content/equities/REG_IND_{d}.csv"
    got = miss = 0
    for d in weekdays(date(2022, 4, 1), date.today() - timedelta(days=1)):
        fp = f"{out}/reg_ind/REG_IND_{d:%d%m%Y}.csv"
        os.makedirs(f"{out}/reg_ind", exist_ok=True)
        if os.path.exists(fp): continue
        try:
            r = S.get(pat.format(d=d.strftime("%d%m%Y")), timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                open(fp, "wb").write(r.content); got += 1
            else: miss += 1
        except Exception: miss += 1
        time.sleep(0.3)
    log(f"surveillance archive sweep (REG_IND pattern): {got} hits, {miss} misses "
        f"{'— pattern may be wrong; hits>0 means F3 archives EXIST' if got==0 else ''}")

# ---------------------------------------------------------------- 4. SLB
def slb():
    out = f"{BASE}/slb"; os.makedirs(out, exist_ok=True)
    try:
        r = S.get("https://www.nseindia.com/api/slb-marketwatch", timeout=25)
        if r.status_code == 200:
            open(f"{out}/slb_{date.today():%Y%m%d}.json", "wb").write(r.content)
            log("slb: snapshot saved")
    except Exception as e:
        log(f"slb failed ({type(e).__name__})")

# ---------------------------------------------------------------- 5. lock-up calendars (best effort)
def lockups():
    out = f"{BASE}/lockups"; os.makedirs(out, exist_ok=True)
    ok = 0
    for year in range(2022, 2027):
        fp = f"{out}/anchor_lockin_{year}.html"
        if os.path.exists(fp): ok += 1; continue
        try:
            r = requests.get(
                f"https://www.chittorgarh.com/report/anchor-investor-lock-in-end-dates/156/all/?year={year}",
                headers={"User-Agent": S.headers["User-Agent"]},
                timeout=30)
            if r.status_code == 200 and len(r.content) > 5000:
                open(fp, "wb").write(r.content); ok += 1
        except Exception:
            pass
        time.sleep(2)
    log(f"lockups: {ok}/5 year pages saved (HTML; tables may be JS-loaded — "
        f"Claude will check if data is embedded; if not, manual export remains)")

if __name__ == "__main__":
    for fn in (futures_patch, pledge_history, surveillance, slb, lockups):
        try: fn()
        except Exception as e:
            log(f"SECTION FAILED {fn.__name__}: {type(e).__name__}: {e}")
    with open(f"{BASE}/FETCH_REPORT.md", "w") as f:
        f.write(f"# Fetch report — {datetime.utcnow():%Y-%m-%d %H:%M} UTC\n\n" + "\n".join(REPORT) + "\n")
    print("REPORT:\n" + "\n".join(REPORT))
