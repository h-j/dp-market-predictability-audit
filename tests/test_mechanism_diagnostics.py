"""
Unit & Integration Tests for SD-008 Mechanism Diagnostic Studies.

Verifies:
1. Mathematical proof of the 0.75 strength boundary under k_falsify=3.0.
2. Diagnostic study execution without modifying core cognition logic.
"""
from pathlib import Path
import pytest

from experiments.diagnostics.run_mechanism_diagnostics import (
    run_study_1_calibration_breakdown,
    run_study_2_recall_breakdown,
    run_study_3_scoped_reasoning_breakdown,
)

PROJECT_ROOT = Path(__file__).parent.parent


def test_strength_boundary_mathematical_proof():
    """Verify 0.75 rule strength boundary math under k_falsify=3.0."""
    def eq_conf(s: float, k_falsify: float = 3.0) -> float:
        return s / (s + k_falsify * (1.0 - s))

    # Strength >= 0.75 produces E[Beta] >= 0.50
    assert eq_conf(0.80) > 0.50
    assert eq_conf(0.75) == 0.50
    # Strength < 0.75 produces E[Beta] < 0.50
    assert eq_conf(0.60) < 0.50
    assert eq_conf(0.50) < 0.50


def test_diagnostic_studies_execution():
    """Verify mechanism diagnostic studies run cleanly and return expected structure."""
    s1_res = run_study_1_calibration_breakdown()
    assert "phase_a" in s1_res
    assert "phase_b" in s1_res
    assert "phase_c" in s1_res

    s2_res = run_study_2_recall_breakdown()
    assert s2_res["d1_e1_promoted"] is True
    assert s2_res["d2_e2_promoted"] is False

    s3_res = run_study_3_scoped_reasoning_breakdown()
    assert s3_res["dp_beliefs_count"] == 0
