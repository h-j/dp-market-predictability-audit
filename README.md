# DP-Core — Reflective Cognition Substrate (`v1.0-research-complete`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Hermetic Unit Tests](https://img.shields.io/badge/Unit%20Tests-48%20PASSED%20(Offline)-brightgreen)](#hermetic-unit-testing)
[![Release Tag](https://img.shields.io/badge/Git%20Tag-v1.0--research--complete-blue)](https://github.com/h-j/dp-market-predictability-audit/releases/tag/v1.0-research-complete)

> **Citable Empirical Research Repository**: A reproducible negative-results research artifact evaluating market predictability bounds (NIFTY, RELIANCE, TCS, NIFTY 100) and LLM-guided hypothesis reasoning value.

---

## 1. Executive Research Overview

The **DP-Core Reflective Cognition Substrate** is an experimental quantitative intelligence framework. This repository contains the complete codebase, point-in-time datasets, pre-registrations, frozen hypothesis arms, and programmatic reports for **all seven empirical research tracks**:

1. **Daily / 3-Day Single-Name Direction**: 🔴 **`DIRECTION IS DEAD`** ($MCC \approx 0.00$ out-of-sample).
2. **5-Day Realized Volatility Forecasting**: 🟢 **`VOLATILITY IS ALIVE`** ($R^2_{\text{vs\_pers}} = +0.26 \text{ to } +0.40$).
3. **India VIX Encompassing Test**: 🟢 **`H-V3 CONFIRMED (NO EDGE BEYOND VIX)`** ($\beta_B = 0.0410$, 95% CI includes zero).
4. **Options Volatility Strategy**: 🔴 **`G-OPT FAILED`** (Unconditional short-strangle selling beats model-filtered selling).
5. **Cross-Sectional Ranking Study**: 🔴 **`CROSS-SECTIONAL NULL`** (Composite IC = $-0.0063$, top quintile Sharpe $0.45$ vs equal-weight benchmark $0.72$).
6. **Substrate LLM Hypothesis Survival**: 🔴 **`LLM REASONING NULL`** (LLM survivor edge $= +0.1948$ vs random survivor edge $= +0.2493$, Mann-Whitney $p = 0.6876$).

See the 5-minute technical overview in [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) and master report in [volatility_options_track_report.md](volatility_options_track_report.md).

---

## 2. Installation & Quickstart

### Prerequisites
- Python $\ge 3.12$
- Poetry package manager (`pip install poetry`)

### Setup
```bash
git clone https://github.com/h-j/dp-market-predictability-audit.git
cd dp-market-predictability-audit
poetry install
```

---

## 3. Reproducing Research Studies End-to-End

Every research study can be run end-to-end programmatically from committed point-in-time CSV datasets in `data/`.

### 1. Phase 3 Rebuild: LLM Hypothesis Survival Study (Gate G-P3)
Backtests frozen treatment arm (150 LLM hypotheses) vs control arm (150 Random hypotheses) across 60/40 In-Sample/Out-of-Sample splits:
```bash
poetry run python -m bootstrap.run_phase3_evaluation
```
*Outputs*: [`llm_hypothesis_survival_report.md`](llm_hypothesis_survival_report.md)

### 2. Phase 2b: Cross-Sectional Ranking Study (Gates G-XS1 & G-XS2)
Evaluates 5 cross-sectional signals over NIFTY 100 constituents across 35 rebalance months with 0.25% transaction costs:
```bash
poetry run python -m bootstrap.run_cross_sectional_study
```
*Outputs*: [`cross_sectional_ranking_report.md`](cross_sectional_ranking_report.md)

### 3. Options Volatility Strategy Study (Gate G-OPT)
Simulates short-strangle and iron-condor options selling policies under corrected annualized volatility conversion ($\hat{\sigma}_{ann} = \hat{\sigma}_{daily} \times \sqrt{252}$):
```bash
poetry run python -m bootstrap.run_options_simulation_study
```
*Outputs*: [`options_simulation_study_report.md`](options_simulation_study_report.md)

### 4. India VIX Mincer-Zarnowitz Encompassing Study (Gate G-VIX / H-V3)
Ingests India VIX history and executes Mincer-Zarnowitz joint regressions with 1,000 block bootstrap iterations:
```bash
poetry run python -m bootstrap.run_vix_encompassing_study
```
*Outputs*: [`vix_encompassing_study_report.md`](vix_encompassing_study_report.md)

### 5. Multi-Asset Volatility Harness & Positive-Control Study
Evaluates HAR-RV, EWMA, and Gradient Boosting volatility models vs persistence on stride-5 non-overlapping folds:
```bash
poetry run python -m bootstrap.run_volatility_positive_control_study
```
*Outputs*: [`volatility_positive_control_report.md`](volatility_positive_control_report.md)

### 6. Single-Name & Index Direction Walk-Forward Study
Evaluates daily and 3-day directional prediction across OHLCV, market breadth, delivery, and institutional flow feature tiers:
```bash
poetry run python -m bootstrap.run_walkforward_direction_study
```
*Outputs*: [`walkforward_direction_study_report.md`](walkforward_direction_study_report.md)

---

## 4. Hermetic Unit Testing

The repository includes a 100% hermetic test suite executing with network sockets explicitly disabled:

```bash
# Run unit tests hermetically with network sockets disabled
poetry run python -c "import socket, sys; socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('NETWORK DISABLED')); import pytest; sys_exit = pytest.main(['tests/']); raise SystemExit(sys_exit)"
```

Or using standard pytest:
```bash
poetry run pytest tests/
```

---

## 5. Repository Structure & Navigation Map

```text
dp-market-predictability-audit/
├── README.md                              # Repository overview & execution guide
├── RESEARCH_FINDINGS.md                   # 5-minute technical overview of all 7 tracks
├── LICENSE                                # MIT Open Source License
├── pyproject.toml                         # Pinned dependencies & Poetry configuration
├── bootstrap/                             # Executable study runners & entry points
│   ├── run_phase3_evaluation.py           # Phase 3 LLM Hypothesis Survival runner
│   ├── run_cross_sectional_study.py       # Cross-sectional ranking study runner
│   ├── run_options_simulation_study.py    # Corrected options simulation runner
│   ├── run_vix_encompassing_study.py      # India VIX encompassing study runner
│   └── run_volatility_positive_control_study.py # Volatility harness runner
├── cognition/                             # Reflective cognition & hypothesis grammar
│   └── grammar/                           # Formal context-free hypothesis grammar G
├── experiments/preregistration/           # Frozen pre-registration documents
├── market/                                # Market memory, cross-sectional engine & options sim
├── data/                                  # Committed point-in-time CSVs & frozen hypothesis arms
│   └── hypotheses/                        # llm_arm.json, random_arm.json, llm_generation_log.json
└── tests/                                 # 48 hermetic offline unit tests
```

---

## 6. Citation

If referencing this repository or its negative empirical results in quantitative research:

```bibtex
@misc{dp_market_predictability_audit_2026,
  author = {DP-Core Research Team},
  title = {DP-Core Reflective Cognition Substrate: Empirical Market Predictability Audit},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/h-j/dp-market-predictability-audit}},
  tag = {v1.0-research-complete}
}
```

---

## 7. License

Distributed under the [MIT License](LICENSE).
