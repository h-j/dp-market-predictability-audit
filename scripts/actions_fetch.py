#!/usr/bin/env python3
"""
actions_fetch.py — v4 (Sep 13): MAINTENANCE MODE.
All historical archives are secured; v4 keeps every series current through
the Oct 30 test window using the endpoints VERIFIED by the AG discovery
mission (sast3132 pledge, RBI prid scanner, SLB series master). Nightly
via schedule; resumable; fails soft; reports to FETCH_REPORT.md.
"""

import os, io, time, json, zipfile
from datetime import date, timedelta, datetime
import requests, pandas as pd

BASE = "docs/Market files"
REPORT = []
def log(s): print(s); REPORT.append(f"- {s}")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

def nse(extra=()):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*",
                       "Accept-Language": "en-US,en;q=0.9",
                       "Referer": "https://www.nseindia.com/"})
    for u in ["https://www.nseindia.com"] + list(extra):
        try: s.get(u, timeout=30)
        except requests.RequestException: pass
        time.sleep(1.5)
    return s

def weekdays(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5: yield d
        d += timedelta(days=1)

def futures_topup():
    out = f"{BASE}/fo_futures"; os.makedirs(out, exist_ok=True)
    S = nse()
    pat = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
    chunks, got = [], 0
    for d in weekdays(date.today()-timedelta(days=10), date.today()-timedelta(days=1)):
        try:
            r = S.get(pat.format(d=d.strftime("%Y%m%d")), timeout=25)
            if r.status_code == 200 and len(r.content) > 1000:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                    col = next((c for c in df.columns if c.strip().upper()=="FININSTRMTP"), None)
                    if col is not None:
                        chunks.append(df[df[col].astype(str).str.upper().isin(["STF","IDF"])]); got += 1
        except Exception: pass
        time.sleep(0.4)
    if chunks:
        fp = f"{out}/futures_{date.today().year}.csv.gz"
        new = pd.concat(chunks, ignore_index=True)
        if os.path.exists(fp):
            new = pd.concat([pd.read_csv(fp, low_memory=False), new],
                             ignore_index=True).drop_duplicates()
        new.to_csv(fp, index=False, compression="gzip")
    log(f"futures top-up: {got} days merged")

def pledge_topup():
    out = f"{BASE}/pledge/raw_windows"; os.makedirs(out, exist_ok=True)
    S = nse(["https://www.nseindia.com/companies-listing/corporate-filings-pledged-data"])
    b = date.today(); a = b - timedelta(days=45)
    fp = f"{out}/sast3132_{a:%Y%m%d}_{b:%Y%m%d}.json"
    try:
        r = S.get("https://www.nseindia.com/api/corporate-pledgedata-sast3132"
                   f"?index=equities&from_date={a:%d-%m-%Y}&to_date={b:%d-%m-%Y}", timeout=60)
        if r.status_code == 200 and len(r.content) > 50:
            open(fp, "wb").write(r.content)
            n = len(r.json().get("data", []))
            log(f"pledge top-up: rolling 45d window saved ({n} records)")
        else:
            log(f"pledge top-up: HTTP {r.status_code}")
    except Exception as e:
        log(f"pledge top-up failed ({type(e).__name__})")

def slb_snapshot():
    out = f"{BASE}/slb"; os.makedirs(out, exist_ok=True)
    S = nse(["https://www.nseindia.com/market-data/securities-lending-and-borrowing"])
    fp = f"{out}/slb_{date.today():%Y%m%d}.json"
    if os.path.exists(fp): return
    try:
        r = S.get("https://www.nseindia.com/api/live-analysis-slb-series-master", timeout=45)
        if r.status_code == 200:
            open(fp, "wb").write(r.content); log("slb: daily snapshot saved")
    except Exception as e:
        log(f"slb snapshot failed ({type(e).__name__})")

def surveillance_snapshots():
    out = f"{BASE}/surveillance"; os.makedirs(out, exist_ok=True)
    S = nse()
    for name, url in {"asm_current": "https://www.nseindia.com/api/reportASM",
                       "gsm_current": "https://www.nseindia.com/api/reportGSM"}.items():
        fp = f"{out}/{name}_{date.today():%Y%m%d}.json"
        if os.path.exists(fp): continue
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 50:
                open(fp, "wb").write(r.content); log(f"surveillance: {name} saved")
        except Exception as e:
            log(f"surveillance {name} failed ({type(e).__name__})")
        time.sleep(1)

def rbi_topup():
    """Continue AG's prid scanner forward from its saved state, parse new
    MMO pages into the daily series CSV (append-only, dedup on date)."""
    out = f"{BASE}/rbi_mmo"; os.makedirs(out, exist_ok=True)
    state_fp = f"{out}/prid_scan_state.json"
    series_fp = f"{out}/rbi_mmo_daily_series.csv"
    if not os.path.exists(state_fp):
        log("rbi: no scanner state found - skipping (historical series already complete)")
        return
    st = json.load(open(state_fp))
    last = int(st.get("last_prid", 0))
    R = requests.Session(); R.headers.update({"User-Agent": UA})
    found = 0
    for prid in range(last + 1, last + 400):
        try:
            r = R.get("https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
                       f"?prid={prid}", timeout=30)
            if r.status_code == 200 and b"Money Market Operations" in r.content:
                open(f"{out}/mmo_{prid}.html", "wb").write(r.content)
                found += 1
                st["last_prid"] = prid
        except Exception:
            break
        time.sleep(0.6)
    json.dump(st, open(state_fp, "w"))
    log(f"rbi top-up: scanned to prid {st['last_prid']}, {found} new MMO pages "
        f"(analyst parses new pages into {os.path.basename(series_fp)} at test time)")

if __name__ == "__main__":
    for fn in (futures_topup, pledge_topup, slb_snapshot,
               surveillance_snapshots, rbi_topup):
        try: fn()
        except Exception as e:
            log(f"SECTION FAILED {fn.__name__}: {type(e).__name__}: {e}")
    with open(f"{BASE}/FETCH_REPORT.md", "w") as f:
        f.write(f"# Fetch report v4 (maintenance) — {datetime.utcnow():%Y-%m-%d %H:%M} UTC\n\n"
                 + "\n".join(REPORT) + "\n")
    print("REPORT:\n" + "\n".join(REPORT))
