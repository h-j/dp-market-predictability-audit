"""
Cross-Sectional Ranking Study Runner (WS-C).

Executes monthly cross-sectional study over NIFTY 100 universe (~34 rebalance dates).
Evaluates Signals 1-5 and Composite against Pre-Registered Gates G-XS1 and G-XS2.
Outputs cross_sectional_ranking_report.md and updates volatility_options_track_report.md.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from market.cross_section.ic_evaluator import CrossSectionalICEvaluator, ICSignalSummary
from market.cross_section.portfolio_sim import MonthlyPortfolioLog, PortfolioSimulator, PortfolioStats
from market.cross_section.signal_engine import (
    CrossSectionalSignalEngine,
    get_monthly_rebalance_dates,
    load_nifty_benchmark,
    load_universe_data,
)

SURVIVORSHIP_CAVEAT_VERBATIM = (
    "Backfilling current NIFTY 100 constituent tickers introduces survivorship bias, "
    "inflating long-side returns. Any passing verdict must be treated as upper-bound candidate "
    "edge pending historical constituent verification."
)


def run_cross_sectional_study(data_dir: Path = None):
    """
    Main study runner for Phase 2b Cross-Sectional Ranking Study.
    """
    print("=" * 80)
    print("CROSS-SECTIONAL RANKING STUDY (Phase 2b: Final Information-Value Experiment)")
    print("=" * 80)

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    universe_dfs = load_universe_data(data_dir)
    nifty_df = load_nifty_benchmark(data_dir)

    print(f"✓ Loaded {len(universe_dfs)} constituent ticker CSVs from {data_dir / 'universe'}")
    print(f"✓ Loaded NIFTY benchmark ({len(nifty_df)} rows)")

    rebalance_dates = get_monthly_rebalance_dates(universe_dfs, start_date="2023-01-01")
    print(f"✓ Extracted {len(rebalance_dates)} monthly rebalance dates ({rebalance_dates[0].strftime('%Y-%m-%d')} to {rebalance_dates[-1].strftime('%Y-%m-%d')})")

    engine = CrossSectionalSignalEngine(universe_dfs, nifty_df, history_cutoff_months=15)
    monthly_ranks: Dict[pd.Timestamp, pd.DataFrame] = {}

    for dt in rebalance_dates:
        df_ranks = engine.compute_cross_section_ranks(dt)
        if not df_ranks.empty:
            # Merge NIFTY forward 1-month return for benchmark
            nifty_hist = nifty_df[nifty_df["date"] <= dt]
            nifty_fwd = nifty_df[nifty_df["date"] > dt]
            if len(nifty_hist) > 0 and len(nifty_fwd) >= 15:
                nifty_entry = nifty_hist["close"].iloc[-1]
                nifty_exit = nifty_fwd["close"].iloc[min(21, len(nifty_fwd) - 1)]
                nifty_ret = (nifty_exit / nifty_entry) - 1.0
                df_ranks["nifty_fwd_return"] = nifty_ret
            monthly_ranks[dt] = df_ranks

    print(f"✓ Generated monthly ranks across {len(monthly_ranks)} valid rebalance months")

    # Evaluate IC & Permutation Nulls
    evaluator = CrossSectionalICEvaluator(n_bootstrap=2000, n_permutations=1000, seed=42)
    signal_cols = ["mom_6m1m", "rev_1m", "rs_nifty_3m", "vol_3m_inv", "vol_trend", "composite"]

    ic_summaries: Dict[str, ICSignalSummary] = {}
    print("\n--- Signal Rank IC & Significance Evaluation ---")
    for sig in signal_cols:
        _, summary = evaluator.evaluate_signal_ic(monthly_ranks, sig)
        ic_summaries[sig] = summary
        sig_str = "SIGNIFICANT" if summary.is_statistically_significant else "NOT SIG"
        print(f"  • {sig:12s}: Mean IC={summary.mean_ic:+.4f} | t-stat={summary.t_stat:+.2f} | 95% CI=[{summary.bootstrap_ci_lower:+.4f}, {summary.bootstrap_ci_upper:+.4f}] | Null P97.5={summary.null_p975:+.4f} [{sig_str}]")

    comp_summary = ic_summaries["composite"]

    # LEAKAGE STOP CONDITION CHECK
    if comp_summary.mean_ic > 0.15:
        raise RuntimeError(
            f"LEAKAGE AUDIT TRIGGERED: Composite mean IC = {comp_summary.mean_ic:.4f} exceeds +0.15 threshold! "
            f"Impossibly high for daily/monthly technical signals. Aborting study for audit."
        )

    # Portfolio Simulation & Cost Deduction (0.25% one-way cost)
    simulator = PortfolioSimulator(one_way_cost_pct=0.0025)
    portfolio_results: Dict[str, Tuple[List[MonthlyPortfolioLog], PortfolioStats, PortfolioStats]] = {}

    print("\n--- Monthly Quintile Portfolio Simulation (0.25% Transaction Costs) ---")
    for sig in signal_cols:
        logs, q5_stats, q1_stats = simulator.simulate_signal_portfolios(monthly_ranks, sig)
        portfolio_results[sig] = (logs, q5_stats, q1_stats)
        print(f"  • {sig:12s} Q5 Long Net: CAGR={q5_stats.cagr_pct:+.2f}% | Sharpe={q5_stats.sharpe_ratio:.2f} | MaxDD={q5_stats.max_drawdown_pct:.2f}% | Active={q5_stats.active_return_pct:+.2f}% | Turnover={q5_stats.mean_monthly_turnover:.2f}")

    # Compute NIFTY Benchmark Stats over the same period
    nifty_rets = [l.nifty_return / 100.0 for l in portfolio_results["composite"][0]]
    nifty_stats = simulator.compute_portfolio_stats(nifty_rets, [0.0] * len(nifty_rets))

    # Pre-Registered Gate Evaluation
    gate_g_xs1_passed = comp_summary.is_statistically_significant
    comp_q5_stats = portfolio_results["composite"][1]
    gate_g_xs2_passed = comp_q5_stats.sharpe_ratio > nifty_stats.sharpe_ratio

    if gate_g_xs1_passed and gate_g_xs2_passed:
        final_verdict = "EDGE-CANDIDATE"
    elif gate_g_xs1_passed and not gate_g_xs2_passed:
        final_verdict = "UNIMPLEMENTABLE SIGNAL"
    else:
        final_verdict = "CROSS-SECTIONAL NULL"

    print("\n" + "=" * 80)
    print(f"PRE-REGISTERED GATE G-XS1 (Signal Existence): {'PASSED' if gate_g_xs1_passed else 'FAILED'}")
    print(f"PRE-REGISTERED GATE G-XS2 (Implementable Sharpe): {'PASSED' if gate_g_xs2_passed else 'FAILED'}")
    print(f"FINAL TRACK VERDICT: {final_verdict}")
    print("=" * 80 + "\n")

    generate_markdown_reports(
        ic_summaries,
        portfolio_results,
        nifty_stats,
        gate_g_xs1_passed,
        gate_g_xs2_passed,
        final_verdict,
        len(monthly_ranks),
    )


def generate_markdown_reports(
    ic_summaries: Dict[str, ICSignalSummary],
    portfolio_results: Dict[str, Tuple[List[MonthlyPortfolioLog], PortfolioStats, PortfolioStats]],
    nifty_stats: PortfolioStats,
    gate_g_xs1_passed: bool,
    gate_g_xs2_passed: bool,
    final_verdict: str,
    total_months: int,
):
    report_lines = []
    report_lines.append("# Cross-Sectional Ranking Study Report (Phase 2b)")
    report_lines.append("\n**Date**: 2026-08-05  ")
    report_lines.append("**Substrate**: `dp-market-predictability-audit`  ")
    report_lines.append(f"**Evaluation Window**: {total_months} Rebalance Months  ")
    report_lines.append(f"**Final Track Verdict**: **`{final_verdict}`**\n")

    report_lines.append("## Pre-Registered Gate Summary\n")
    report_lines.append("| Gate / Metric | Pre-Registered Condition | Result | Verdict |")
    report_lines.append("| :--- | :--- | :---: | :---: |")

    comp_ic = ic_summaries["composite"]
    comp_q5 = portfolio_results["composite"][1]

    g1_res = f"Mean IC={comp_ic.mean_ic:+.4f}, 95% CI=[{comp_ic.bootstrap_ci_lower:+.4f}, {comp_ic.bootstrap_ci_upper:+.4f}], Null P97.5={comp_ic.null_p975:+.4f}"
    g1_v = "🟢 **PASSED**" if gate_g_xs1_passed else "🔴 **FAILED**"
    report_lines.append(f"| **G-XS1 (Signal Existence)** | Composite Mean IC > 0 & CI > 0 & > Null P97.5 | {g1_res} | {g1_v} |")

    g2_res = f"Top-Quintile Sharpe={comp_q5.sharpe_ratio:.2f} vs NIFTY Sharpe={nifty_stats.sharpe_ratio:.2f}"
    g2_v = "🟢 **PASSED**" if gate_g_xs2_passed else "🔴 **FAILED**"
    report_lines.append(f"| **G-XS2 (Retail Implementable)** | Top Quintile Sharpe > NIFTY Sharpe (after 0.25% cost) | {g2_res} | {g2_v} |")

    report_lines.append(f"\n### Final Track Decision: **`{final_verdict}`**\n")
    if final_verdict == "EDGE-CANDIDATE":
        report_lines.append("Composite cross-sectional ranking is statistically real AND retail-implementable after transaction costs. Proceeding to 6-month paper-trading protocol.")
    elif final_verdict == "UNIMPLEMENTABLE SIGNAL":
        report_lines.append("Composite cross-sectional signal is statistically real (G-XS1 passed), but net of 0.25% transaction costs on monthly turnover, it fails to beat NIFTY buy-and-hold on Sharpe ratio (G-XS2 failed). Signal is un-implementable for retail accounts.")
    else:
        report_lines.append("Composite cross-sectional ranking contains no statistically significant edge over random permutations. The information-value research program's empirical phase closes with a definitive CROSS-SECTIONAL NULL.")

    report_lines.append("\n## Signal Rank Information Coefficient (IC) Summary Table\n")
    report_lines.append("| Signal | Mean IC | Std IC | t-statistic | Bootstrap 95% CI | Null P97.5 | G-XS1 Status | Q5-Q1 Spread (Ann %) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sig, s in ic_summaries.items():
        sig_label = f"**{sig}**" if sig == "composite" else sig
        st_str = "PASS" if s.is_statistically_significant else "FAIL"
        report_lines.append(
            f"| {sig_label} | {s.mean_ic:+.4f} | {s.std_ic:.4f} | {s.t_stat:+.2f} | "
            f"[{s.bootstrap_ci_lower:+.4f}, {s.bootstrap_ci_upper:+.4f}] | {s.null_p975:+.4f} | {st_str} | {s.q5_q1_spread_annualized_pct:+.2f}% |"
        )

    report_lines.append("\n## Long-Only Equal-Weighted Top Quintile (Q5) Portfolio Performance (0.25% Costs)\n")
    report_lines.append("| Portfolio Signal | CAGR (%) | Sharpe Ratio | Max Drawdown (%) | Mean Monthly Turnover | Total Costs (%) | Active Return vs NIFTY (%) | Win Rate (%) |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sig, (logs, q5_st, q1_st) in portfolio_results.items():
        sig_label = f"**Q5 Top Quintile ({sig})**" if sig == "composite" else f"Q5 ({sig})"
        report_lines.append(
            f"| {sig_label} | {q5_st.cagr_pct:+.2f}% | {q5_st.sharpe_ratio:.2f} | {q5_st.max_drawdown_pct:.2f}% | "
            f"{q5_st.mean_monthly_turnover:.2f} | {q5_st.total_transaction_costs_pct:.2f}% | {q5_st.active_return_pct:+.2f}% | {q5_st.win_rate_pct:.2f}% |"
        )

    report_lines.append(
        f"| **NIFTY Benchmark (B&H)** | {nifty_stats.cagr_pct:+.2f}% | {nifty_stats.sharpe_ratio:.2f} | {nifty_stats.max_drawdown_pct:.2f}% | "
        f"0.00 | 0.00% | 0.00% | {nifty_stats.win_rate_pct:.2f}% |"
    )

    report_lines.append("\n---\n")
    report_lines.append("## Mandatory Survivorship Bias Caveat\n")
    report_lines.append(f"> {SURVIVORSHIP_CAVEAT_VERBATIM}\n")

    report_text = "\n".join(report_lines)
    out_path = Path(__file__).parent.parent / "cross_sectional_ranking_report.md"
    out_path.write_text(report_text)
    print(f"✓ Saved Cross-Sectional Ranking Study Report to {out_path}")

    update_volatility_options_track_report(final_verdict, comp_ic, comp_q5, nifty_stats)


def update_volatility_options_track_report(
    final_verdict: str,
    comp_ic: ICSignalSummary,
    comp_q5: PortfolioStats,
    nifty_stats: PortfolioStats,
):
    track_report_path = Path(__file__).parent.parent / "volatility_options_track_report.md"
    if not track_report_path.exists():
        return

    content = track_report_path.read_text()

    # Append Phase 2b Summary Section
    summary_addition = f"""

---

## 6. Phase 2b: Cross-Sectional Ranking Study & Final Program Status

**Status**: Completed  
**Final Track Verdict**: **`{final_verdict}`**

| Gate / Hypothesis | Subject | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **G-XS1** | Cross-Sectional Signal Existence | Composite Mean IC > 0 & 95% CI > 0 & > Null P97.5 | Mean IC={comp_ic.mean_ic:+.4f}, CI=[{comp_ic.bootstrap_ci_lower:+.4f}, {comp_ic.bootstrap_ci_upper:+.4f}] | {"🟢 **PASSED**" if comp_ic.is_statistically_significant else "🔴 **FAILED**"} |
| **G-XS2** | Retail Implementability | Top Quintile Sharpe > NIFTY Sharpe (after 0.25% cost) | Top Quintile Sharpe={comp_q5.sharpe_ratio:.2f} vs NIFTY Sharpe={nifty_stats.sharpe_ratio:.2f} | {"🟢 **PASSED**" if comp_q5.sharpe_ratio > nifty_stats.sharpe_ratio else "🔴 **FAILED**"} |

> {SURVIVORSHIP_CAVEAT_VERBATIM}
"""

    if "Phase 2b: Cross-Sectional Ranking Study" not in content:
        content += summary_addition
        track_report_path.write_text(content)
        print(f"✓ Appended Phase 2b summary section to {track_report_path}")


if __name__ == "__main__":
    run_cross_sectional_study()
