#!/usr/bin/env python3
"""
download_market_datasets.py — Modular fetcher for Phase 2 market datasets:
  1. NSE Promoter Pledge disclosures historical (SAST 31(1)/(2) Oct 2019 – Present)
  2. RBI Daily Money Market Operations series (Net LAF, WACR Oct 2019 – Present)
  3. NSE SLB (Securities Lending & Borrowing) daily data snapshot & series
  4. Chittorgarh Mainboard IPO listing dates & Anchor lock-in schedules (2022–2026)

Usage:
  poetry run python -m scripts.download_market_datasets --all
  poetry run python -m scripts.download_market_datasets --pledge
  poetry run python -m scripts.download_market_datasets --slb
  poetry run python -m scripts.download_market_datasets --ipos
  poetry run python -m scripts.download_market_datasets --rbi
"""

import argparse
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from html.parser import HTMLParser

import pandas as pd
import requests

BASE = "docs/Market files"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


# ----------------------------------------------------------------------
# 1. NSE Session Helper
# ----------------------------------------------------------------------
def get_nse_session(warmup_urls=None):
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": UA,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-pledged-data",
        }
    )
    urls = [
        "https://www.nseindia.com/companies-listing/corporate-filings-pledged-data",
    ]
    if warmup_urls:
        urls.extend(warmup_urls)
    for u in urls:
        try:
            s.get(u, timeout=15)
            time.sleep(0.5)
        except Exception:
            pass
    return s


# ----------------------------------------------------------------------
# 2. NSE Promoter Pledge Disclosures (SAST 31(1)/(2))
# ----------------------------------------------------------------------
def fetch_pledge_disclosures(start_date=date(2019, 10, 1), end_date=None):
    if end_date is None:
        end_date = date.today()
    out_dir = os.path.join(BASE, "pledge")
    raw_dir = os.path.join(out_dir, "raw_windows")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n=== [1/4] Fetching NSE Promoter Pledge Disclosures ({start_date} to {end_date}) ===", flush=True)
    s = get_nse_session()
    api_url = (
        "https://www.nseindia.com/api/corporate-pledgedata-sast3132"
        "?index=equities&from_date={d1}&to_date={d2}"
    )

    cur = start_date
    windows_saved = 0
    total_records = 0
    all_records = []

    # Iterate in 30-day windows
    while cur < end_date:
        nxt = min(cur + timedelta(days=29), end_date)
        d1_str = cur.strftime("%d-%m-%Y")
        d2_str = nxt.strftime("%d-%m-%Y")
        fp = os.path.join(raw_dir, f"sast3132_{cur:%Y%m%d}_{nxt:%Y%m%d}.json")

        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    recs = d.get("data", [])
                    total_records += len(recs)
                    all_records.extend(recs)
                windows_saved += 1
                cur = nxt + timedelta(days=1)
                continue
            except Exception:
                pass

        # Fetch from NSE
        success = False
        for attempt in range(3):
            try:
                r = s.get(api_url.format(d1=d1_str, d2=d2_str), timeout=25)
                if r.status_code == 200:
                    d = r.json()
                    recs = d.get("data", [])
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(d, f)
                    total_records += len(recs)
                    all_records.extend(recs)
                    windows_saved += 1
                    success = True
                    print(f"  Pledge window {d1_str} to {d2_str}: {len(recs)} disclosures (cumulative: {total_records})", flush=True)
                    break
                elif r.status_code in (401, 403):
                    print("  Session expired, re-warming NSE session...", flush=True)
                    s = get_nse_session()
                    time.sleep(2)
            except Exception as e:
                time.sleep(1.5)

        if not success:
            print(f"  FAILED window {d1_str} to {d2_str}", flush=True)
        time.sleep(0.4)
        cur = nxt + timedelta(days=1)

    # Consolidate into CSV.GZ
    if all_records:
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates(subset=["seqId", "symbol", "broadcastdate"], keep="first")
        csv_path = os.path.join(out_dir, "pledge_disclosures_2019_2026.csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        print(f"-> Consolidated {len(df)} unique pledge disclosures into {csv_path}", flush=True)
    else:
        print("-> No pledge records collected.", flush=True)


# ----------------------------------------------------------------------
# 3. NSE SLB Daily Data Snapshot
# ----------------------------------------------------------------------
def fetch_slb_data():
    out_dir = os.path.join(BASE, "slb")
    os.makedirs(out_dir, exist_ok=True)
    today_str = date.today().strftime("%Y%m%d")

    print(f"\n=== [2/4] Fetching NSE SLB Daily Data Snapshot ({today_str}) ===")
    s = get_nse_session(
        ["https://www.nseindia.com/market-data/securities-lending-and-borrowing"]
    )

    # 1. Fetch series master
    master_url = "https://www.nseindia.com/api/live-analysis-slb-series-master"
    try:
        r_master = s.get(master_url, timeout=20)
        if r_master.status_code != 200:
            print(f"  SLB master fetch failed: HTTP {r_master.status_code}")
            return
        master_data = r_master.json()
    except Exception as e:
        print(f"  SLB master fetch error: {e}")
        return

    series_keys = []
    for grp in ["Series A", "Series B", "Series R3"]:
        for item in master_data.get("data", {}).get(grp, []):
            k = item.get("key")
            if k and k not in series_keys:
                series_keys.append(k)

    print(f"  Found {len(series_keys)} active SLB series: {series_keys}")

    all_slb_rows = []
    snapshot = {"timestamp": datetime.now().isoformat(), "series_master": master_data, "series_data": {}}

    for key in series_keys:
        url = f"https://www.nseindia.com/api/live-analysis-slb?series={key}"
        try:
            r = s.get(url, timeout=20)
            if r.status_code == 200:
                d = r.json()
                snapshot["series_data"][key] = d
                recs = d.get("data", [])
                for row in recs:
                    row_copy = dict(row)
                    row_copy["series_key"] = key
                    row_copy["snapshot_date"] = date.today().isoformat()
                    all_slb_rows.append(row_copy)
                if recs:
                    print(f"  SLB series {key}: {len(recs)} securities with active quotes/interest")
            time.sleep(0.5)
        except Exception as e:
            print(f"  SLB series {key} failed: {e}")

    # Save JSON snapshot
    json_path = os.path.join(out_dir, f"slb_{today_str}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"-> Saved SLB JSON snapshot to {json_path}")

    # Save consolidated CSV
    if all_slb_rows:
        df = pd.DataFrame(all_slb_rows)
        csv_path = os.path.join(out_dir, f"slb_snapshot_{today_str}.csv")
        df.to_csv(csv_path, index=False)
        # Also maintain latest
        df.to_csv(os.path.join(out_dir, "slb_snapshot_latest.csv"), index=False)
        print(f"-> Saved {len(df)} SLB security records to {csv_path}")


# ----------------------------------------------------------------------
# 4. Chittorgarh Mainboard IPOs & Anchor Lock-in Schedules
# ----------------------------------------------------------------------
def fetch_chittorgarh_ipos(years=(2022, 2023, 2024, 2025, 2026)):
    out_dir = os.path.join(BASE, "lockups")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== [3/4] Fetching Chittorgarh IPO & Anchor Lock-in Data ({years}) ===")

    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Referer": "https://www.chittorgarh.com/",
    })

    all_ipos = []
    # Report 118: Mainboard IPO Timetable & Dates
    for yr in years:
        url = f"https://webnodejs.chittorgarh.com/cloud/report/data-read/118/1/1/{yr}/0/0/mainboard/0?search="
        try:
            r = s.get(url, timeout=20)
            if r.status_code == 200:
                data = r.json()
                rows = data.get("reportTableData", [])
                print(f"  Year {yr}: {len(rows)} mainboard IPOs fetched")
                for item in rows:
                    all_ipos.append(item)
            else:
                print(f"  Year {yr} failed: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Year {yr} error: {e}")
        time.sleep(1.0)

    if not all_ipos:
        print("-> No IPO data fetched.")
        return

    # Clean HTML tags and compute anchor lock-in expiry dates
    clean_ipos = []
    for item in all_ipos:
        co_raw = item.get("Company", "")
        co_clean = re.sub(r"<[^>]+>", "", co_raw).strip()
        open_dt = item.get("Opening Date", "")
        close_dt = item.get("Closing Date", "")
        allot_dt_str = item.get("Allotment Date", "")
        list_dt_str = item.get("Listing Date", "")
        folder_slug = item.get("~urlrewrite_folder_name", "")

        # Parse Allotment Date to calculate 30-day and 90-day anchor lock-in
        # SEBI ICDR regulations (applicable since April 1, 2022):
        # 50% anchor shares locked in for 30 days from allotment date
        # Remaining 50% anchor shares locked in for 90 days from allotment date
        lockin_30 = None
        lockin_90 = None
        parsed_allot = None
        if allot_dt_str:
            for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
                try:
                    parsed_allot = datetime.strptime(allot_dt_str.strip(), fmt).date()
                    break
                except ValueError:
                    pass
        if parsed_allot:
            lockin_30 = parsed_allot + timedelta(days=30)
            lockin_90 = parsed_allot + timedelta(days=90)

        clean_ipos.append({
            "ipo_id": item.get("~id"),
            "company": co_clean,
            "issue_type": item.get("Issue Type", "Mainboard"),
            "opening_date": open_dt,
            "closing_date": close_dt,
            "allotment_date": allot_dt_str,
            "listing_date": list_dt_str,
            "anchor_30d_lockin_expiry": lockin_30.isoformat() if lockin_30 else "",
            "anchor_90d_lockin_expiry": lockin_90.isoformat() if lockin_90 else "",
            "slug": folder_slug,
        })

    df = pd.DataFrame(clean_ipos)
    df = df.drop_duplicates(subset=["ipo_id", "company"], keep="first")
    csv_timetable = os.path.join(out_dir, "chittorgarh_ipo_listings_2022_2026.csv")
    df.to_csv(csv_timetable, index=False)
    print(f"-> Saved {len(df)} mainboard IPO listing and timetable records to {csv_timetable}")

    # Anchor lock-ins specific export
    anchor_df = df[df["anchor_30d_lockin_expiry"] != ""][
        ["ipo_id", "company", "allotment_date", "listing_date", "anchor_30d_lockin_expiry", "anchor_90d_lockin_expiry"]
    ]
    csv_anchor = os.path.join(out_dir, "chittorgarh_anchor_lockin_2022_2026.csv")
    anchor_df.to_csv(csv_anchor, index=False)
    print(f"-> Saved {len(anchor_df)} anchor lock-in schedules to {csv_anchor}")


# ----------------------------------------------------------------------
# 5. RBI Daily Money Market Operations Series (WACR, Net LAF)
# ----------------------------------------------------------------------
class RBIHtmlTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.curr_table = []
        self.curr_row = []
        self.curr_cell = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.curr_table = []
        elif tag == "tr":
            self.curr_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.curr_cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.in_cell = False
            self.curr_row.append(" ".join("".join(self.curr_cell).split()))
        elif tag == "tr":
            if self.curr_row:
                self.curr_table.append(self.curr_row)
        elif tag == "table":
            if self.curr_table:
                self.tables.append(self.curr_table)

    def handle_data(self, data):
        if self.in_cell:
            self.curr_cell.append(data)


def parse_mmo_page(html_content, prid):
    """
    Extract date, Call Money WACR, Triparty Repo, Market Repo, and Net LAF from RBI MMO release.
    Distinguishes Overnight Segment from Term Segment so policy WACR is accurately captured.
    """
    d_match = re.search(
        r"Money Market Operations as on ([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2} [A-Za-z]+ \d{4})",
        html_content,
        re.I,
    )
    mmo_date = d_match.group(1) if d_match else None
    if not mmo_date:
        return None

    date_iso = None
    for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"]:
        try:
            date_iso = datetime.strptime(re.sub(r"\s+", " ", mmo_date.strip()), fmt).strftime("%Y-%m-%d")
            break
        except ValueError:
            pass

    parser = RBIHtmlTableParser()
    parser.feed(html_content)

    call_vol = call_wacr = None
    treps_vol = treps_wacr = None
    mrepo_vol = mrepo_wacr = None
    net_laf_total = None
    net_laf_day = None

    is_overnight = True

    for table in parser.tables:
        for row in table:
            row_str = " ".join(row).lower()
            if "term segment" in row_str:
                is_overnight = False
            elif "overnight segment" in row_str:
                is_overnight = True

            if is_overnight:
                if "call money" in row_str and len(row) >= 3:
                    call_vol = row[1].strip()
                    call_wacr = row[2].strip().replace("%", "").strip()
                elif "triparty repo" in row_str and len(row) >= 3:
                    treps_vol = row[1].strip()
                    treps_wacr = row[2].strip().replace("%", "").strip()
                elif "market repo" in row_str and len(row) >= 3:
                    mrepo_vol = row[1].strip()
                    mrepo_wacr = row[2].strip().replace("%", "").strip()

            # Net liquidity
            if "outstanding including today" in row_str or "f. net liquidity injected" in row_str:
                for c in reversed(row):
                    if re.match(r"^[-+]?[\d,]+(?:\.\d+)?$", c.strip()):
                        net_laf_total = c.strip()
                        break
            elif "today" in row_str and "operations" in row_str:
                for c in reversed(row):
                    if re.match(r"^[-+]?[\d,]+(?:\.\d+)?$", c.strip()):
                        net_laf_day = c.strip()
                        break
            elif "net liquidity injected" in row_str and not net_laf_total:
                for c in reversed(row):
                    if re.match(r"^[-+]?[\d,]+(?:\.\d+)?$", c.strip()):
                        net_laf_total = c.strip()
                        break

    return {
        "prid": prid,
        "date_iso": date_iso or mmo_date,
        "date": mmo_date,
        "call_money_vol_cr": call_vol,
        "call_money_wacr_pct": call_wacr,
        "treps_vol_cr": treps_vol,
        "treps_wacr_pct": treps_wacr,
        "market_repo_vol_cr": mrepo_vol,
        "market_repo_wacr_pct": mrepo_wacr,
        "net_laf_cr": net_laf_total or net_laf_day,
        "net_laf_day_cr": net_laf_day,
        "net_laf_total_cr": net_laf_total,
    }


import threading

_thread_local = threading.local()

def get_thread_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        try:
            s.get("https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx", timeout=15)
        except Exception:
            pass
        _thread_local.session = s
    return _thread_local.session


def fetch_rbi_mmo_series(min_prid=48250, max_prid=63586, max_workers=35):
    out_dir = os.path.join(BASE, "rbi_mmo")
    html_dir = os.path.join(out_dir, "raw_pages")
    os.makedirs(html_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rbi_mmo_daily_series.csv")
    wantlist_path = os.path.join(out_dir, "prid_wantlist.txt")

    print(f"\n=== [4/4] Fetching RBI Daily Money Market Operations Series (PRID {min_prid} to {max_prid}) ===", flush=True)

    # Load existing series if present
    existing_records = {}
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            for r in df_old.to_dict(orient="records"):
                existing_records[int(r["prid"])] = r
            print(f"  Loaded {len(existing_records)} existing MMO records from CSV", flush=True)
        except Exception:
            pass

    # Load scan checkpoint (which PRIDs have already been checked)
    chk_path = os.path.join(out_dir, "scanned_prids.txt")
    scanned = set()
    if os.path.exists(chk_path):
        with open(chk_path, "r") as f:
            for line in f:
                p = line.strip()
                if p.isdigit():
                    scanned.add(int(p))
        print(f"  Loaded {len(scanned)} already-scanned PRID checkpoints", flush=True)

    to_check = [p for p in range(min_prid, max_prid + 1) if p not in scanned and p not in existing_records]
    print(f"  Scanning {len(to_check)} PRIDs with {max_workers} worker threads...", flush=True)

    chk_file = open(chk_path, "a", buffering=1)
    want_file = open(wantlist_path, "a", buffering=1)

    def probe_prid(prid):
        raw_file = os.path.join(html_dir, f"mmo_{prid}.html")
        if os.path.exists(raw_file):
            try:
                with open(raw_file, "r", encoding="utf-8") as f:
                    content = f.read()
                rec = parse_mmo_page(content, prid)
                return prid, rec, content
            except Exception:
                pass

        sess = get_thread_session()
        url = f"https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={prid}"
        for _ in range(2):
            try:
                r = sess.get(url, timeout=12)
                if r.status_code == 200:
                    if "Money Market Operations" in r.text:
                        rec = parse_mmo_page(r.text, prid)
                        return prid, rec, r.text
                    return prid, None, None
            except Exception:
                time.sleep(1)
        return prid, None, None

    batch_size = 250
    hits_count = len(existing_records)
    t_start = time.time()

    for i in range(0, len(to_check), batch_size):
        chunk = to_check[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(probe_prid, p) for p in chunk]
            for fut in as_completed(futures):
                prid, rec, raw_html = fut.result()
                chk_file.write(f"{prid}\n")
                if rec:
                    hits_count += 1
                    existing_records[prid] = rec
                    want_file.write(f"{prid}\n")
                    raw_file = os.path.join(html_dir, f"mmo_{prid}.html")
                    if raw_html and not os.path.exists(raw_file):
                        with open(raw_file, "w", encoding="utf-8") as f:
                            f.write(raw_html)
                    print(f"    HIT [{hits_count}]: PRID {prid} -> {rec.get('date_iso')} | WACR: {rec.get('call_money_wacr_pct')}% | Net LAF: {rec.get('net_laf_cr')}", flush=True)

        # Periodic checkpoint save
        if (i // batch_size) % 2 == 0 and existing_records:
            df_checkpoint = pd.DataFrame(list(existing_records.values()))
            df_checkpoint = df_checkpoint.sort_values(by="prid")
            df_checkpoint.to_csv(csv_path, index=False)
            rate = (i + len(chunk)) / max(0.1, time.time() - t_start)
            print(f"  Progress: {i + len(chunk)}/{len(to_check)} PRIDs scanned ({rate:.1f} req/s, {hits_count} MMO series hits)", flush=True)

    chk_file.close()
    want_file.close()

    # Final save
    if existing_records:
        df_final = pd.DataFrame(list(existing_records.values()))
        df_final = df_final.sort_values(by="prid")
        df_final.to_csv(csv_path, index=False)
        print(f"-> Saved {len(df_final)} RBI Money Market Operations series records to {csv_path}", flush=True)



# ----------------------------------------------------------------------
# CLI Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download Phase 2 Market Datasets")
    parser.add_argument("--all", action="store_true", help="Download all 4 datasets")
    parser.add_argument("--pledge", action="store_true", help="Fetch NSE promoter pledge disclosures")
    parser.add_argument("--slb", action="store_true", help="Fetch NSE SLB daily snapshot")
    parser.add_argument("--ipos", action="store_true", help="Fetch Chittorgarh IPO & anchor lock-in data")
    parser.add_argument("--rbi", action="store_true", help="Fetch RBI daily money market operations series")
    parser.add_argument("--rbi-min", type=int, default=48250, help="Min PRID for RBI scan")
    parser.add_argument("--rbi-max", type=int, default=63586, help="Max PRID for RBI scan")
    parser.add_argument("--workers", type=int, default=35, help="Number of concurrent workers for RBI scan")

    args = parser.parse_args()

    if not (args.all or args.pledge or args.slb or args.ipos or args.rbi):
        parser.print_help()
        sys.exit(0)

    start_time = time.time()
    if args.all or args.pledge:
        fetch_pledge_disclosures()
    if args.all or args.slb:
        fetch_slb_data()
    if args.all or args.ipos:
        fetch_chittorgarh_ipos()
    if args.all or args.rbi:
        fetch_rbi_mmo_series(min_prid=args.rbi_min, max_prid=args.rbi_max, max_workers=args.workers)

    elapsed = time.time() - start_time
    print(f"\nAll operations completed in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
