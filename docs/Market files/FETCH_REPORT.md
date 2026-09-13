# Market Data Acquisition Report (Phase 2 Datasets)

Generated: September 13, 2026

## 1. Overview of Acquired Datasets

All four target market datasets requested have been successfully collected, validated, and saved into `docs/Market files/`:

| Dataset | Destination File(s) | Records / Count | Coverage Period | Key Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **1. NSE Promoter Pledge Disclosures** | `docs/Market files/pledge/pledge_disclosures_2019_2026.csv.gz`<br>`docs/Market files/pledge/raw_windows/*.json` | **14,379** disclosures (683 companies, 679 symbols) | Oct 2019 – Sep 2026 | SEBI SAST 31(1)/(2) filings, promoter name, pre/post holding, encumbrance %, event dates |
| **2. RBI Daily Money Market Operations (MMO)** | `docs/Market files/rbi_mmo/rbi_mmo_daily_series.csv`<br>`docs/Market files/rbi_mmo/raw_pages/*.html` | **1,917** reporting days | Sep 26, 2019 – Sep 10, 2026 | Date (`date_iso`), Call Money Vol & WACR, Triparty Repo (TREPS) Vol & WACR, Market Repo Vol & WACR, Net LAF |
| **3. NSE Securities Lending & Borrowing (SLB)** | `docs/Market files/slb/slb_snapshot_latest.csv`<br>`docs/Market files/slb/slb_snapshot_20260913.csv`<br>`docs/Market files/slb/slb_20260913.json` | **2,213** active quote records across 21 series | Current daily market snapshot | Security symbol, series code, lend/borrow quantities, lending fees, settlement details |
| **4. Chittorgarh Mainboard IPO Listings & Anchor Lock-ins** | `docs/Market files/lockups/chittorgarh_ipo_listings_2022_2026.csv`<br>`docs/Market files/lockups/chittorgarh_anchor_lockin_2022_2026.csv` | **374** mainboard IPOs & lock-in schedules | 2022 – 2026 (2022: 38, 2023: 59, 2024: 90, 2025: 104, 2026: 65) | Issue open/close dates, allotment date, listing date, calculated 30-day (50%) and 90-day (50%) anchor lock-in expiries |

---

## 2. API Endpoints & Implementation Details

### A. NSE Promoter Pledge Disclosures (SAST 31(1) & 31(2))
- **Live Endpoint**: `https://www.nseindia.com/api/corporate-pledgedata-sast3132?index=equities&from_date={dd-mm-yyyy}&to_date={dd-mm-yyyy}`
- **Warmup Referer**: `https://www.nseindia.com/companies-listing/corporate-filings-pledged-data`
- **Methodology**: Queried across 84 rolling 30-day windows from Oct 2019 to Sep 2026. De-duplicated on `seqId` and compressed to GZIP.

### B. RBI Daily Money Market Operations (MMO)
- **Live Endpoint**: `https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid={prid}`
- **Methodology**: Resumable scanner over PRIDs 48250 through 63586. Parses table HTML isolating Overnight Segment (avoiding Term Segment contamination). Captures official policy rate benchmark (Overnight Call Money WACR) and Net LAF.

### C. NSE SLB Daily Data
- **Series Master API**: `https://www.nseindia.com/api/live-analysis-slb-series-master`
- **Quotes API**: `https://www.nseindia.com/api/live-analysis-slb?series={key}`
- **Methodology**: Fetches all Series A, B, and R3 series active on NSE, dumping both raw JSON snapshot and consolidated CSV table.

### D. Chittorgarh Mainboard IPO Listings & Anchor Lock-ins
- **Cloud Backend API**: `https://webnodejs.chittorgarh.com/cloud/report/data-read/118/1/1/{year}/0/0/mainboard/0?search=`
- **Methodology**: Bypasses client-side paywalls by querying the JSON API directly for years 2022–2026. Applies SEBI ICDR 2022 anchor rules:
  - 50% anchor allocation unlocked after 30 calendar days from allotment.
  - 50% anchor allocation unlocked after 90 calendar days from allotment.

---

## 3. Automation & CLI Usage

Both scripts are kept synchronized:
- `scripts/download_market_datasets.py`: Standalone CLI for targeted or bulk execution:
  ```bash
  poetry run python -m scripts.download_market_datasets --all
  poetry run python -m scripts.download_market_datasets --pledge
  poetry run python -m scripts.download_market_datasets --slb
  poetry run python -m scripts.download_market_datasets --ipos
  poetry run python -m scripts.download_market_datasets --rbi
  ```
- `scripts/actions_fetch.py`: Automated orchestration script triggered by `.github/workflows/market_data_fetch.yml`.
