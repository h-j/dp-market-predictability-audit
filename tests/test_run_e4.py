"""
Unit & Integration Tests for Milestone E4 Design-Change Experiment (PROMPT E4 / C6).

Verifies:
1. Per-seed determinism across 20-seed E4 battery runs.
2. Fix A vs Fix B isolation in E4a / E4b arms.
3. Frozen runtime parameters and pre-registered gate_e4_v2.yaml.
"""
from pathlib import Path
import pytest
import yaml
from bench.run_e4 import run_single_seed_battery, assert_frozen_constants
from experiments.e4_adapter import E4Adapter
from bench.synthworld.scenarios import s1_clean, s4_scope


def is_equal_or_nan(v1: float, v2: float) -> bool:
    import math
    if math.isnan(v1) and math.isnan(v2):
        return True
    return v1 == v2


def test_e4_seed_determinism():
    """Verify that running the same seed twice produces 100% identical metric outputs."""
    r1, _ = run_single_seed_battery(seed=42)
    r2, _ = run_single_seed_battery(seed=42)

    for sc_id in ["S1", "S2", "S3", "S4"]:
        for l_name in r1[sc_id]:
            assert is_equal_or_nan(r1[sc_id][l_name]["brier_score"], r2[sc_id][l_name]["brier_score"])
            assert is_equal_or_nan(r1[sc_id][l_name]["brier_regret"], r2[sc_id][l_name]["brier_regret"])
            assert is_equal_or_nan(r1[sc_id][l_name]["precision"], r2[sc_id][l_name]["precision"])
            assert is_equal_or_nan(r1[sc_id][l_name]["recall"], r2[sc_id][l_name]["recall"])


def test_e4_arm_isolation():
    """Verify E4a (Fix A only), E4b (Fix B only), and E4 (Combined) behavior differences."""
    sc4 = s4_scope()
    ad_e4a = E4Adapter(sc4, arm="E4a")
    ad_e4b = E4Adapter(sc4, arm="E4b")
    ad_e4 = E4Adapter(sc4, arm="E4")

    # E4a uses unconditioned hypotheses (no scope vars)
    assert not any(h.scope_var is not None for h in ad_e4a.hypotheses)
    # E4b and E4 use scope-conditioned hypotheses
    assert any(h.scope_var == "C" for h in ad_e4b.hypotheses)
    assert any(h.scope_var == "C" for h in ad_e4.hypotheses)


def test_frozen_constants_and_gate_v2():
    """Verify runtime assertions and gate_e4_v2.yaml existence."""
    assert_frozen_constants()
    
    gate_v2_path = Path(__file__).parent.parent / "experiments" / "preregistration" / "gate_e4_v2.yaml"
    assert gate_v2_path.exists()
    
    with open(gate_v2_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["gate_e4_v2"]["version"] == "2.0.0"
    assert data["gate_e4_v2"]["frozen_parameters"]["k_falsify"] == 3.0
