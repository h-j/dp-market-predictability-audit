"""
PROMPT E4 — 20-Seed Synthetic Battery & Gate E4 Evaluator.

Executes full protocol:
- 4 Scenarios at DEFAULT step lengths (S1=3000, S2=3000, S3=4000, S4=4000)
- 20 Seeds (0..19)
- 7 Learners:
  1. TrueModel
  2. FlatBayesian
  3. WindowedFrequency
  4. ContextualBayesian
  5. DP/EkamNet-E4a (Fix A only: s_hat prediction)
  6. DP/EkamNet-E4b (Fix B only: scope keying)
  7. DP/EkamNet-E4  (Combined: Fix A + Fix B)

Outputs:
- bench/results/e4_raw_metrics.jsonl
- bench/results/e4_reliability_curve.csv
- bench/results/e4_results.md
"""
import json
import math
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.synthworld.scenarios import s1_clean, s2_spurious, s3_regime, s4_scope
from bench.synthworld.world import World, Scenario
from bench.synthworld.learners import (
    TrueModel,
    FlatBayesian,
    WindowedFrequency,
    ContextualBayesian,
)
from experiments.e4_adapter import E4Adapter
from bench.synthworld import metrics
from bench.synthworld.harness import run as run_scenario


def mean(vals: List[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / max(1, len(vals))


def std_dev(vals: List[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var)


def iqr(vals: List[float]) -> Tuple[float, float]:
    vals = sorted([v for v in vals if not math.isnan(v)])
    if not vals:
        return (float("nan"), float("nan"))
    n = len(vals)
    q25 = vals[int(0.25 * n)]
    q75 = vals[min(int(0.75 * n), n - 1)]
    return (q25, q75)


def compute_ece(calibration_pairs: List[Tuple[float, int]], num_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    if not calibration_pairs:
        return 0.0, []

    bins = [[] for _ in range(num_bins)]
    for conf, outcome in calibration_pairs:
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append((conf, outcome))

    total_count = len(calibration_pairs)
    ece = 0.0
    curve = []

    for b in range(num_bins):
        bin_samples = bins[b]
        bin_lower = b / num_bins
        bin_upper = (b + 1) / num_bins
        bin_mid = (bin_lower + bin_upper) / 2.0

        if bin_samples:
            avg_conf = sum(c for c, o in bin_samples) / len(bin_samples)
            avg_acc = sum(o for c, o in bin_samples) / len(bin_samples)
            weight = len(bin_samples) / total_count
            ece += weight * abs(avg_acc - avg_conf)
            curve.append({
                "bin": b,
                "bin_midpoint": bin_mid,
                "count": len(bin_samples),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "calibration_gap": abs(avg_acc - avg_conf),
            })
        else:
            curve.append({
                "bin": b,
                "bin_midpoint": bin_mid,
                "count": 0,
                "avg_confidence": bin_mid,
                "avg_accuracy": bin_mid,
                "calibration_gap": 0.0,
            })

    return ece, curve


def run_single_seed_battery(seed: int) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Tuple[float, int]]]:
    scenario_factories = [
        ("S1", s1_clean),
        ("S2", s2_spurious),
        ("S3", s3_regime),
        ("S4", s4_scope),
    ]

    seed_results = {}
    e4_cal_pairs = []

    for sc_id, factory in scenario_factories:
        sc = factory()
        sc.seed = seed

        learners = [
            TrueModel(sc),
            FlatBayesian(sc),
            WindowedFrequency(sc),
            ContextualBayesian(sc),
            E4Adapter(sc, arm="E4a"),
            E4Adapter(sc, arm="E4b"),
            E4Adapter(sc, arm="E4"),
        ]

        r = run_scenario(sc, learners)
        scores = r["scores"]
        beliefs = r["beliefs"]
        timeline = r["timeline"]

        oracle_name = TrueModel(sc).name
        oracle_brier = metrics.phase_brier(scores[oracle_name], 300, sc.T)

        sc_metrics = {}

        for ln in learners:
            ln_brier = metrics.phase_brier(scores[ln.name], 300, sc.T)
            brier_regret = ln_brier - oracle_brier

            disc = metrics.discovery(beliefs[ln.name], sc)
            precision = disc["precision"]
            recall = disc["recall"]
            decoy_claims = disc["decoy_claims"]

            rec_steps = float("nan")
            collateral_val = float("nan")
            if sc_id == "S3":
                rec = metrics.recovery(scores[ln.name], flip_t=2000, affected=["E1", "E3"], window=100)
                rec_steps = float(rec["recovery_steps"]) if rec["recovery_steps"] is not None else 2000.0
                collateral_val = metrics.collateral(scores[ln.name], flip_t=2000, unaffected=["E2"], window=300)

            scoped_regret = float("nan")
            if sc_id == "S4":
                ctx_indices = [t for t in range(300, min(sc.T - 1, len(scores[ln.name]))) if timeline[t].get("C", 0) == 1]
                if ctx_indices:
                    ln_scoped_brier = metrics.mean([scores[ln.name][t][e] for t in ctx_indices for e in sc.effects])
                    oracle_scoped_brier = metrics.mean([scores[oracle_name][t][e] for t in ctx_indices for e in sc.effects])
                    scoped_regret = ln_scoped_brier - oracle_scoped_brier

            sc_metrics[ln.name] = {
                "brier_score": ln_brier,
                "brier_regret": brier_regret,
                "precision": precision,
                "recall": recall,
                "decoy_claims": decoy_claims,
                "recovery_steps": rec_steps,
                "collateral": collateral_val,
                "scoped_regret": scoped_regret,
            }

            if ln.name == "DP/EkamNet-E4":
                for t in range(sc.T - 1):
                    ev = timeline[t]
                    nxt = timeline[t + 1]
                    preds = ln.predict(t, ev)
                    for e in sc.effects:
                        e4_cal_pairs.append((preds[e], nxt[e]))

        seed_results[sc_id] = sc_metrics

    return seed_results, e4_cal_pairs


def run_e4_battery(num_seeds: int = 20) -> Dict[str, Any]:
    print("======================================================================")
    print(f"STARTING PROMPT E4 BATTERY ({num_seeds} Seeds x 4 Scenarios x 7 Learners)")
    print("======================================================================")

    all_raw_metrics = []
    e4_calibration_pairs = []

    scenarios = ["S1", "S2", "S3", "S4"]
    learner_names = [
        "TrueModel (oracle floor)",
        "FlatBayesian",
        "WindowedFrequency(w=200)",
        "ContextualBayesian",
        "DP/EkamNet-E4a",
        "DP/EkamNet-E4b",
        "DP/EkamNet-E4",
    ]

    metrics_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        s: {
            l: {
                "brier_score": [],
                "brier_regret": [],
                "precision": [],
                "recall": [],
                "decoy_claims": [],
                "recovery_steps": [],
                "collateral": [],
                "scoped_regret": [],
            }
            for l in learner_names
        }
        for s in scenarios
    }

    for seed in range(num_seeds):
        seed_res, cal_pairs = run_single_seed_battery(seed)
        e4_calibration_pairs.extend(cal_pairs)

        for sc_id, l_dict in seed_res.items():
            for l_name, m_dict in l_dict.items():
                record = {"seed": seed, "scenario": sc_id, "learner": l_name}
                record.update(m_dict)
                all_raw_metrics.append(record)

                for m_name, val in m_dict.items():
                    metrics_data[sc_id][l_name][m_name].append(val)

        print(f"✓ Completed Seed {seed + 1}/{num_seeds}")

    # Save raw metrics
    results_dir = PROJECT_ROOT / "bench" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_path = results_dir / "e4_raw_metrics.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in all_raw_metrics:
            f.write(json.dumps(r) + "\n")

    # Compute ECE
    ece, rel_curve = compute_ece(e4_calibration_pairs, num_bins=10)
    curve_path = results_dir / "e4_reliability_curve.csv"
    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("bin,bin_midpoint,count,avg_confidence,avg_accuracy,calibration_gap\n")
        for row in rel_curve:
            f.write(f"{row['bin']},{row['bin_midpoint']:.4f},{row['count']},{row['avg_confidence']:.4f},{row['avg_accuracy']:.4f},{row['calibration_gap']:.4f}\n")

    # Evaluate Gate E4 Criteria on Combined E4 Arm
    s1_regret = mean(metrics_data["S1"]["DP/EkamNet-E4"]["brier_regret"])
    s4_scoped_regret = mean(metrics_data["S4"]["DP/EkamNet-E4"]["scoped_regret"])
    s4_prec = mean(metrics_data["S4"]["DP/EkamNet-E4"]["precision"])
    s4_rec = mean(metrics_data["S4"]["DP/EkamNet-E4"]["recall"])
    s2_decoys = mean(metrics_data["S2"]["DP/EkamNet-E4"]["decoy_claims"])
    s2_prec = mean(metrics_data["S2"]["DP/EkamNet-E4"]["precision"])
    s1_rec = mean(metrics_data["S1"]["DP/EkamNet-E4"]["recall"])

    cond_s1_regret = s1_regret <= 0.0050
    cond_s4_regret = s4_scoped_regret <= 0.0100
    cond_s4_prec = s4_prec == 1.00
    cond_s4_rec = s4_rec == 1.00
    cond_s2_decoys = s2_decoys == 0.00
    cond_s2_prec = s2_prec == 1.00
    cond_s1_rec = s1_rec >= 0.45

    gate_pass = (
        cond_s1_regret
        and cond_s4_regret
        and cond_s4_prec
        and cond_s4_rec
        and cond_s2_decoys
        and cond_s2_prec
        and cond_s1_rec
    )

    if gate_pass:
        branch = "PASS"
    elif cond_s1_regret and not cond_s4_rec:
        branch = "PARTIAL_FIX_A"
    elif cond_s4_rec and not cond_s1_regret:
        branch = "PARTIAL_FIX_B"
    else:
        branch = "NULL"

    print("\n======================================================================")
    print(f"GATE E4 BRANCH EVALUATION VERDICT: [{branch}]")
    print("======================================================================")
    print("CRITERIA EVALUATION BREAKDOWN (DP/EkamNet-E4 Combined Arm):")
    print(f"  - S1 Brier Regret <= 0.0050:   {s1_regret:.4f} (Met: {cond_s1_regret})")
    print(f"  - S4 Scoped Regret <= 0.0100:  {s4_scoped_regret:.4f} (Met: {cond_s4_regret})")
    print(f"  - S4 Discovery Precision == 1: {s4_prec:.2f} (Met: {cond_s4_prec})")
    print(f"  - S4 Discovery Recall == 1:    {s4_rec:.2f} (Met: {cond_s4_rec})")
    print(f"  - S2 Decoy Claims == 0.00:    {s2_decoys:.2f} (Met: {cond_s2_decoys})")
    print(f"  - S2 Discovery Precision == 1: {s2_prec:.2f} (Met: {cond_s2_prec})")
    print(f"  - S1 Discovery Recall >= 0.45: {s1_rec:.2f} (Met: {cond_s1_rec})")
    print("----------------------------------------------------------------------")

    md_path = results_dir / "e4_results.md"
    generate_markdown_report(md_path, metrics_data, branch, ece)
    print(f"✓ Saved markdown report to {md_path}")

    return {
        "branch": branch,
        "ece": ece,
        "metrics": metrics_data,
        "raw_path": str(raw_path),
        "md_path": str(md_path),
    }


def generate_markdown_report(
    md_path: Path,
    metrics_data: Dict[str, Dict[str, Dict[str, List[float]]]],
    branch: str,
    ece: float,
):
    scenarios = ["S1", "S2", "S3", "S4"]
    learners = [
        "TrueModel (oracle floor)",
        "FlatBayesian",
        "WindowedFrequency(w=200)",
        "ContextualBayesian",
        "DP/EkamNet-E4a",
        "DP/EkamNet-E4b",
        "DP/EkamNet-E4",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# E4 — Design-Change Experiment Results (20 Seeds)\n\n")
        f.write("Authoritative 20-seed synthetic battery evaluation for Milestone E4.\n\n")
        f.write(f"### Registered Gate E4 Branch Verdict: **[{branch}]**\n\n")
        f.write(f"**Expected Calibration Error (ECE - Combined Arm)**: `{ece:.4f}`\n\n")
        f.write("---\n\n")

        for sc_id in scenarios:
            f.write(f"## Scenario {sc_id} Results (20 Seeds)\n\n")
            f.write("| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

            for l_name in learners:
                d = metrics_data[sc_id][l_name]
                b_reg = f"{mean(d['brier_regret']):.4f} ± {std_dev(d['brier_regret']):.4f}"
                prec = f"{mean(d['precision']):.4f} ± {std_dev(d['precision']):.4f}"
                rec = f"{mean(d['recall']):.4f} ± {std_dev(d['recall']):.4f}"
                dec = f"{mean(d['decoy_claims']):.2f} ± {std_dev(d['decoy_claims']):.2f}"
                rec_st = f"{mean(d['recovery_steps']):.1f} ± {std_dev(d['recovery_steps']):.1f}" if sc_id == "S3" else "N/A"
                col = f"{mean(d['collateral']):.4f} ± {std_dev(d['collateral']):.4f}" if sc_id == "S3" else "N/A"

                f.write(f"| **{l_name}** | {b_reg} | {prec} | {rec} | {dec} | {rec_st} | {col} |\n")

            f.write("\n")


if __name__ == "__main__":
    run_e4_battery(num_seeds=20)
