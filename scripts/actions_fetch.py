#!/usr/bin/env python3
"""
actions_fetch.py — v2 (Sep 12). Changes vs v1:
  - pledge: hardened NSE session (corporate-filings warmup, 60s timeout,
    30-day windows, one retry) + BSE fallback route
  - NEW rbi_mmo: RBI daily Money Market Operations press releases
    (monthly index -> per-day pages), for F6 WACR/LAF series
  - slb: retry with corporates warmup
  - surveillance: two alternate archive URL patterns probed (sampled)
  - futures/current-lists/lockups: unchanged, skip-if-present
Fails soft per section; FETCH_REPORT.md records everything; resumable.
"""
import os, io, time, json, zipfile, re
from datetime import date, timedelta, datetime
import requests, pandas as pd

BASE = "docs/Market files"
os.makedirs(BASE, exist_ok=True)
REPORT = []
def log(s):
    print(s); REPORT.append(f"- {s}")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

def nse_session(extra_warm=()):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*",
                       "Accept-Language": "en-US,en;q=0.9",
                       "Referer": "https://www.nseindia.com/"})
    warm = ["https://www.nseindia.com"] + list(extra_warm)
    for u in warm:
        try: s.get(u, timeout=30)
        except requests.RequestException: pass
        time.sleep(1.5)
    return s

def weekdays(a, b):
    d = a
    while d <= b:
        if d.weekday() < 5: yield d
        d += timedelta(days=1)

# ---------------------------------------------------------- 1. futures (unchanged, resumable)
def futures_patch():
    out = f"{BASE}/fo_futures"; os.makedirs(out, exist_ok=True)
    yfile = f"{out}/futures_{date.today().year}.csv.gz"
    S = nse_session()
    pat = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
    start = date(2024, 7, 8)
    if os.path.exists(yfile):
        # top-up only: last 10 weekdays
        start = date.today() - timedelta(days=14)
    got = 0; by_year = {}
    for d in weekdays(start, date.today() - timedelta(days=1)):
        tag = d.strftime("%Y%m%d")
        try:
            r = S.get(pat.format(d=tag), timeout=25)
            if r.status_code == 200 and len(r.content) > 1000:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
                col = next((c for c in df.columns if c.strip().upper()=="FININSTRMTP"), None)
                if col is not None:
                    by_year.setdefault(d.year, []).append(
                        df[df[col].astype(str).str.upper().isin(["STF","IDF"])]); got += 1
        except Exception: pass
        time.sleep(0.4)
    for y, ch in by_year.items():
        new = pd.concat(ch, ignore_index=True)
        sfx = "_udiff" if y == 2024 else ""
        fp = f"{out}/futures_{y}{sfx}.csv.gz"
        if os.path.exists(fp):
            old = pd.read_csv(fp, low_memory=False)
            dc = next((c for c in old.columns if c.strip().upper() in ("TRADDT","TIMESTAMP")), None)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates()
        new.to_csv(fp, index=False, compression="gzip")
    log(f"futures: {got} days fetched/topped-up")

# ---------------------------------------------------------- 2. pledge v2: NSE hardened + BSE fallback
def pledge_v2():
    out = f"{BASE}/pledge"; os.makedirs(out, exist_ok=True)
    S = nse_session(extra_warm=["https://www.nseindia.com/companies-listing/corporate-filings-pledged-data"])
    url = "https://www.nseindia.com/api/corporate-pledgedata?index=equities&from_date={a}&to_date={b}"
    start, end = date(2019, 10, 1), date.today()
    got = fail = 0
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=29), end)
        fp = f"{out}/pledge_{cur:%Y%m%d}_{nxt:%Y%m%d}.json"
        if not os.path.exists(fp):
            ok = False
            for attempt in range(2):
                try:
                    r = S.get(url.format(a=cur.strftime("%d-%m-%Y"),
                                          b=nxt.strftime("%d-%m-%Y")),
                              timeout=60)
                    if r.status_code == 200 and len(r.content) > 50:
                        open(fp, "wb").write(r.content); got += 1; ok = True; break
                except requests.RequestException:
                    time.sleep(3)
            if not ok: fail += 1
            time.sleep(1.5)
        cur = nxt + timedelta(days=1)
    log(f"pledge NSE v2: {got} windows saved, {fail} failed")
    if got == 0:
        # BSE fallback: SDD/pledge statement endpoints
        B = requests.Session(); B.headers.update({"User-Agent": UA,
            "Referer": "https://www.bseindia.com/"})
        try: B.get("https://www.bseindia.com", timeout=30)
        except requests.RequestException: pass
        candidates = [
            "https://api.bseindia.com/BseIndiaAPI/api/PledgeNewData/w?strType=P&strSearch=",
            "https://api.bseindia.com/BseIndiaAPI/api/Pledgee/w?scripcode=&flag=",
        ]
        for i, u in enumerate(candidates):
            try:
                r = B.get(u, timeout=45)
                if r.status_code == 200 and len(r.content) > 100:
                    open(f"{out}/bse_pledge_probe_{i}.json", "wb").write(r.content)
                    log(f"pledge BSE fallback: candidate {i} returned data "
                        f"({len(r.content)} bytes) — Claude to inspect schema")
                else:
                    log(f"pledge BSE candidate {i}: HTTP {r.status_code}")
            except Exception as e:
                log(f"pledge BSE candidate {i} failed ({type(e).__name__})")
            time.sleep(2)

# ---------------------------------------------------------- 3. RBI MMO (F6 liquidity series)
def rbi_mmo():
    out = f"{BASE}/rbi_mmo"; os.makedirs(out, exist_ok=True)
    R = requests.Session(); R.headers.update({"User-Agent": UA})
    got = 0
    # monthly press-release index pages -> harvest MMO links (prid pages)
    cur = date(2019, 10, 1)
    while cur <= date.today():
        idx_url = (f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
                    f"?Mode=0&Month={cur.month}&Year={cur.year}")
        fp_idx = f"{out}/index_{cur:%Y%m}.html"
        if not os.path.exists(fp_idx):
            try:
                r = R.get(idx_url, timeout=45)
                if r.status_code == 200 and b"Money Market Operations" in r.content:
                    open(fp_idx, "wb").write(r.content); got += 1
            except Exception: pass
            time.sleep(1.0)
        cur = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
    log(f"rbi_mmo: {got} monthly index pages saved (prid links inside; "
        f"Claude parses indices, next run fetches the per-day pages it lists)")
    # second pass: fetch any prid pages already parsed into a wantlist
    want = f"{out}/prid_wantlist.txt"
    if os.path.exists(want):
        fetched = 0
        for line in open(want):
            prid = line.strip()
            fp = f"{out}/mmo_{prid}.html"
            if prid and not os.path.exists(fp):
                try:
                    r = R.get(f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
                              f"?prid={prid}", timeout=45)
                    if r.status_code == 200: open(fp, "wb").write(r.content); fetched += 1
                except Exception: pass
                time.sleep(0.8)
        log(f"rbi_mmo: {fetched} MMO day-pages fetched from wantlist")

# ---------------------------------------------------------- 4. SLB retry
def slb():
    out = f"{BASE}/slb"; os.makedirs(out, exist_ok=True)
    S = nse_session(extra_warm=["https://www.nseindia.com/market-data/securities-lending-and-borrowing"])
    try:
        r = S.get("https://www.nseindia.com/api/slb-marketwatch", timeout=60)
        if r.status_code == 200 and len(r.content) > 50:
            open(f"{out}/slb_{date.today():%Y%m%d}.json", "wb").write(r.content)
            log("slb: snapshot saved")
        else:
            log(f"slb: HTTP {r.status_code}")
    except Exception as e:
        log(f"slb failed again ({type(e).__name__}) — likely IP-blocked; residual manual")

# ---------------------------------------------------------- 5. surveillance current + alt patterns
def surveillance():
    out = f"{BASE}/surveillance"; os.makedirs(out, exist_ok=True)
    S = nse_session()
    for name, url in {"asm_current": "https://www.nseindia.com/api/reportASM",
                       "gsm_current": "https://www.nseindia.com/api/reportGSM"}.items():
        fp = f"{out}/{name}_{date.today():%Y%m%d}.json"
        if not os.path.exists(fp):
            try:
                r = S.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 50:
                    open(fp, "wb").write(r.content); log(f"surveillance: {name} saved")
            except Exception as e:
                log(f"surveillance {name} failed ({type(e).__name__})")
            time.sleep(1)
    alt = ["https://nsearchives.nseindia.com/content/equities/eq_asm_{d}.csv",
           "https://nsearchives.nseindia.com/archives/equities/surv/asm_{d}.csv"]
    hits = 0
    for d in [date(2024,6,3), date(2023,6,1), date(2025,6,2)]:
        for p in alt:
            try:
                r = S.get(p.format(d=d.strftime("%d%m%Y")), timeout=15)
                if r.status_code == 200 and len(r.content) > 100:
                    hits += 1; log(f"surveillance ARCHIVE HIT: {p.format(d=d.strftime('%d%m%Y'))}")
            except Exception: pass
            time.sleep(0.5)
    if hits == 0:
        log("surveillance archive alt-patterns: 0 hits (3 sample dates x 2 patterns) — "
            "public dated archives likely absent; F3 heads to NE unless manual route found")

if __name__ == "__main__":
    for fn in (futures_patch, pledge_v2, rbi_mmo, slb, surveillance):
        try: fn()
        except Exception as e:
            log(f"SECTION FAILED {fn.__name__}: {type(e).__name__}: {e}")
    with open(f"{BASE}/FETCH_REPORT.md", "w") as f:
        f.write(f"# Fetch report v2 — {datetime.utcnow():%Y-%m-%d %H:%M} UTC\n\n"
                + "\n".join(REPORT) + "\n")
    print("REPORT:\n" + "\n".join(REPORT))
