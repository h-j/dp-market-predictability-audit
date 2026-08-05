"""
Regression baseline tests for prediction probe LLM phrasing variants (Item 9).

Documents current behavior of _infer_direction and _infer_confidence when presented with
canonical vs paraphrased/synonymous text variants in theory and reflection summaries.
"""

import pytest

from market.replay.prediction_probe import PredictionDirection, PredictionProbeGenerator


class MockObject:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_item9_canonical_phrasing_triggers_uncertainty():
    """Verify canonical keyword 'uncertain' in sentiment/reflection triggers PredictionDirection.uncertain."""
    probe_gen = PredictionProbeGenerator()

    obs = MockObject(
        trend_state="neutral",
        breadth_state="mixed",
        macro_sentiment="uncertain",
        candle_type="normal",
        descriptors=[],
    )
    theory = MockObject(summary="Market direction is unclear.")
    reflection = MockObject(reflection_summary="Reflecting on uncertain conditions.")

    dir_result = probe_gen._infer_direction(obs, theory, reflection)
    assert dir_result == PredictionDirection.uncertain


def test_item9_paraphrased_uncertainty_phrasing_baseline():
    """Document current behavior for paraphrased uncertainty (e.g. 'ambiguous outlook', 'high instability')."""
    probe_gen = PredictionProbeGenerator()

    obs = MockObject(
        trend_state="neutral",
        breadth_state="mixed",
        macro_sentiment="neutral",  # Missing 'uncertain' keyword
        candle_type="normal",
        descriptors=[],
    )

    # Variant A: 'ambiguous outlook' (lacks exact 'uncertain' substring)
    theory_a = MockObject(summary="Market structure indicates an ambiguous outlook.")
    reflection_a = MockObject(reflection_summary="Observations exhibit high instability.")

    dir_a = probe_gen._infer_direction(obs, theory_a, reflection_a)
    # Current behavior: falls back to range_bound because 'uncertain' is not matched
    assert dir_a == PredictionDirection.range_bound, "Current substring matching misses paraphrased uncertainty"

    # Variant B: 'exercise prudence' vs 'caution'
    conf_b1 = probe_gen._infer_confidence(
        PredictionDirection.higher,
        contradictions={},
        theory=MockObject(summary="Upward momentum."),
        reflection=MockObject(reflection_summary="Exercise prudence in positioning."),
    )
    conf_b2 = probe_gen._infer_confidence(
        PredictionDirection.higher,
        contradictions={},
        theory=MockObject(summary="Upward momentum."),
        reflection=MockObject(reflection_summary="Caution advised in positioning."),
    )
    # 'caution' reduces confidence, whereas 'exercise prudence' does not match substring
    assert conf_b2 < conf_b1, "Canonical 'caution' reduces confidence while paraphrase does not"


def test_item9_canonical_strength_weakness_confidence_adjustments():
    """Verify current confidence adjustments for canonical 'strength' and 'weak' terms."""
    probe_gen = PredictionProbeGenerator()
    usefulness = {"score": 0.8}

    base_conf = probe_gen._infer_confidence(
        PredictionDirection.higher,
        contradictions={},
        theory=MockObject(summary="Trend continuation."),
        reflection=MockObject(reflection_summary="Conditions stable."),
        theory_usefulness=usefulness,
    )

    strength_conf = probe_gen._infer_confidence(
        PredictionDirection.higher,
        contradictions={},
        theory=MockObject(summary="Trend continuation shows strength."),
        reflection=MockObject(reflection_summary="Conditions stable."),
        theory_usefulness=usefulness,
    )

    weak_conf = probe_gen._infer_confidence(
        PredictionDirection.higher,
        contradictions={},
        theory=MockObject(summary="Trend continuation looks weak."),
        reflection=MockObject(reflection_summary="Conditions stable."),
        theory_usefulness=usefulness,
    )

    # Document current behavior: 'strength' (+0.03) and 'weak' (-0.04) produce distinct confidence outputs
    assert strength_conf != weak_conf, "Strength and weak keywords should produce distinct confidence scores"
    assert strength_conf != base_conf, "'strength' keyword should alter base confidence"
    assert weak_conf != base_conf, "'weak' keyword should alter base confidence"
