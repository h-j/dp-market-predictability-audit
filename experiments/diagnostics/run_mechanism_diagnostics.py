"""
PROMPT SD-008 Mechanism Diagnostic Study Script.

Empirically investigates the 3 root causes identified in SD-008:
1. Calibration & Brier Regret Gap (S1): Traces prediction gaps P(E) vs P*(E) over time.
2. Recall Gap (S1 & S2): Traces posterior trajectory for D1->E1 (strength 0.80) vs D2->E2 (strength 0.60), proving 0.75 strength boundary under k_falsify=3.0.
3. Scoped Reasoning Gap (S4): Traces unconditioned D1->E1 posterior under 50% context gating C=1.
"""
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench.synthworld.scenarios import s1_clean, s2_spurious, s3_regime, s4_scope
from bench.synthworld.world import World
from bench.synthworld.dp_adapter import DPAdapter
from bench.synthworld.learners import TrueModel, FlatBayesian, ContextualBayesian


def run_study_1_calibration_breakdown() -> Dict[str, Any]:
    print("======================================================================")
    print("STUDY 1: CALIBRATION & BRIER REGRET DIAGNOSTIC (S1 CLEAN)")
    print("======================================================================")

    sc = s1_clean(T=3000)
    sc.seed = 42
    world = World(sc)
    timeline = world.generate()

    adapter = DPAdapter(sc)
    oracle = TrueModel(sc)
    flat = FlatBayesian(sc)

    step_data = []

    for t in range(sc.T - 1):
        ev = timeline[t]
        nxt = timeline[t + 1]

        adapter.observe(t, ev)
        flat.observe(t, ev)

        p_dp = adapter.predict(t, ev)
        p_oracle = oracle.predict(t, ev)
        p_flat = flat.predict(t, ev)

        d1_e1_state = adapter.confidence_states.get(("D1", "E1"))
        d2_e2_state = adapter.confidence_states.get(("D2", "E2"))

        step_data.append({
            "t": t,
            "d1_active": ev.get("D1", 0),
            "e1_outcome": nxt.get("E1", 0),
            "dp_p_e1": p_dp["E1"],
            "oracle_p_e1": p_oracle["E1"],
            "flat_p_e1": p_flat["E1"],
            "d1_e1_conf": d1_e1_state.confidence if d1_e1_state else 0.0,
            "d2_e2_conf": d2_e2_state.confidence if d2_e2_state else 0.0,
        })

    # Segment into Phase A (0..300), Phase B (300..1500), Phase C (1500..3000)
    phase_a = step_data[0:300]
    phase_b = step_data[300:1500]
    phase_c = step_data[1500:3000]

    def phase_stats(phase_steps):
        dp_errs = [(s["dp_p_e1"] - s["e1_outcome"]) ** 2 for s in phase_steps]
        oracle_errs = [(s["oracle_p_e1"] - s["e1_outcome"]) ** 2 for s in phase_steps]
        flat_errs = [(s["flat_p_e1"] - s["e1_outcome"]) ** 2 for s in phase_steps]
        dp_brier = sum(dp_errs) / len(dp_errs)
        oracle_brier = sum(oracle_errs) / len(oracle_errs)
        flat_brier = sum(flat_errs) / len(flat_errs)
        return {
            "dp_brier_regret": dp_brier - oracle_brier,
            "flat_brier_regret": flat_brier - oracle_brier,
            "avg_dp_stating_gap": sum(abs(s["dp_p_e1"] - s["oracle_p_e1"]) for s in phase_steps) / len(phase_steps),
        }

    res_a = phase_stats(phase_a)
    res_b = phase_stats(phase_b)
    res_c = phase_stats(phase_c)

    print(f"Phase A (t=0..300 Warmup):   DP Regret = {res_a['dp_brier_regret']:.4f} | Flat Regret = {res_a['flat_brier_regret']:.4f} | Stating Gap = {res_a['avg_dp_stating_gap']:.4f}")
    print(f"Phase B (t=300..1500 Mid):   DP Regret = {res_b['dp_brier_regret']:.4f} | Flat Regret = {res_b['flat_brier_regret']:.4f} | Stating Gap = {res_b['avg_dp_stating_gap']:.4f}")
    print(f"Phase C (t=1500..3000 End):  DP Regret = {res_c['dp_brier_regret']:.4f} | Flat Regret = {res_c['flat_brier_regret']:.4f} | Stating Gap = {res_c['avg_dp_stating_gap']:.4f}")

    return {"phase_a": res_a, "phase_b": res_b, "phase_c": res_c}


def run_study_2_recall_breakdown() -> Dict[str, Any]:
    print("\n======================================================================")
    print("STUDY 2: RECALL & PROMOTION GATE DIAGNOSTIC (STRENGTH BOUNDARY)")
    print("======================================================================")

    sc = s1_clean(T=3000)
    sc.seed = 42
    world = World(sc)
    timeline = world.generate()

    adapter = DPAdapter(sc)

    for t in range(sc.T - 1):
        ev = timeline[t]
        adapter.observe(t, ev)

    d1_e1 = adapter.confidence_states[("D1", "E1")]
    d2_e2 = adapter.confidence_states[("D2", "E2")]

    print(f"Rule 1: D1 -> E1 (Ground Truth Strength = 0.80 > 0.75 Boundary):")
    print(f"  - Alpha: {d1_e1.alpha:.1f} | Beta: {d1_e1.beta:.1f}")
    print(f"  - Posterior Confidence E[Beta]: {d1_e1.confidence:.4f} (Threshold 0.50 -> Promoted: {d1_e1.confidence >= 0.50})")

    print(f"\nRule 2: D2 -> E2 (Ground Truth Strength = 0.60 < 0.75 Boundary):")
    print(f"  - Alpha: {d2_e2.alpha:.1f} | Beta: {d2_e2.beta:.1f}")
    print(f"  - Posterior Confidence E[Beta]: {d2_e2.confidence:.4f} (Threshold 0.50 -> Promoted: {d2_e2.confidence >= 0.50})")

    # Theoretical Equilibrium Proof
    print("\nMathematical Proof of 0.75 Strength Boundary under k_falsify=3.0:")
    print("For rule strength s, Expected Alpha gain per trigger = s, Expected Beta gain = 3.0*(1-s)")
    print("Equilibrium Posterior Ratio E[Beta] = s / (s + 3*(1-s)) = s / (3 - 2s)")
    for test_s in [0.90, 0.80, 0.75, 0.70, 0.60, 0.50]:
        eq_conf = test_s / (3.0 - 2.0 * test_s)
        print(f"  - Strength s={test_s:.2f} -> Equilibrium E[Beta] = {eq_conf:.4f} (>= 0.50: {eq_conf >= 0.50})")

    return {
        "d1_e1_conf": d1_e1.confidence,
        "d2_e2_conf": d2_e2.confidence,
        "d1_e1_promoted": d1_e1.confidence >= 0.50,
        "d2_e2_promoted": d2_e2.confidence >= 0.50,
    }


def run_study_3_scoped_reasoning_breakdown() -> Dict[str, Any]:
    print("\n======================================================================")
    print("STUDY 3: SCOPED REASONING DIAGNOSTIC (S4 SCOPED RULE)")
    print("======================================================================")

    sc = s4_scope(T=4000)
    sc.seed = 42
    world = World(sc)
    timeline = world.generate()

    adapter = DPAdapter(sc)
    contextual = ContextualBayesian(sc)

    for t in range(sc.T - 1):
        ev = timeline[t]
        adapter.observe(t, ev)
        contextual.observe(t, ev)

    dp_beliefs = adapter.beliefs()
    ctx_beliefs = contextual.beliefs()

    print(f"DP/EkamNet Claimed Persistent Beliefs on S4: {dp_beliefs}")
    print(f"ContextualBayesian Claimed Persistent Beliefs on S4: {ctx_beliefs}")

    d1_e1_state = adapter.confidence_states[("D1", "E1")]
    c_e1_state = adapter.confidence_states[("C", "E1")]

    print(f"\nDP Unconditioned Pair Analysis on S4:")
    print(f"  - (D1, E1) State -> Alpha: {d1_e1_state.alpha:.1f}, Beta: {d1_e1_state.beta:.1f}, Conf: {d1_e1_state.confidence:.4f}")
    print(f"  - (C, E1) State  -> Alpha: {c_e1_state.alpha:.1f}, Beta: {c_e1_state.beta:.1f}, Conf: {c_e1_state.confidence:.4f}")

    print("\nMathematical Proof of Scoped Rule Suppression:")
    print("Rule D1 -> E1 holds with strength 0.85 ONLY when C=1 (active 50% of time).")
    print("When C=0 (50% of time), D1=1 does NOT cause E1 (E1=0 with 95% probability).")
    print("Every step with D1=1 and C=0 adds k_falsify=3.0 to Beta!")
    print("Effective Strength s_eff = 0.85 * 0.50 = 0.425 < 0.75 boundary.")
    print("Unconditioned Equilibrium E[Beta] = 0.425 / (3 - 2*0.425) = 0.425 / 2.15 = 0.1977 << 0.50.")

    return {
        "dp_beliefs_count": len(dp_beliefs),
        "ctx_beliefs_count": len(ctx_beliefs),
        "d1_e1_conf": d1_e1_state.confidence,
    }


def main():
    s1_res = run_study_1_calibration_breakdown()
    s2_res = run_study_2_recall_breakdown()
    s3_res = run_study_3_scoped_reasoning_breakdown()

    summary = {
        "study_1_calibration": s1_res,
        "study_2_recall": s2_res,
        "study_3_scoped": s3_res,
    }

    out_file = PROJECT_ROOT / "bench" / "results" / "mechanism_diagnostics_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved full mechanism diagnostics report to {out_file}")


if __name__ == "__main__":
    main()
