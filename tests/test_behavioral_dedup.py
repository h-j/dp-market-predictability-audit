"""
Unit tests for behavioral & structural hypothesis deduplication engine.
"""

import pytest
from cognition.grammar.behavioral_dedup import (
    compute_jaccard_similarity,
    get_ast_canonical_key,
    cluster_hypotheses_behaviorally,
)


def test_jaccard_similarity_calculation():
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7, 8}
    # Intersection = {4, 5} (size 2), Union = {1, 2, 3, 4, 5, 6, 7, 8} (size 8)
    assert compute_jaccard_similarity(set_a, set_b) == 0.25

    set_c = {1, 2, 3, 4}
    set_d = {1, 2, 3, 4, 5}
    # Intersection = 4, Union = 5 -> Jaccard = 0.80
    assert compute_jaccard_similarity(set_c, set_d) == 0.80

    # Empty sets
    assert compute_jaccard_similarity(set(), set()) == 0.0


def test_ast_canonical_key():
    h1 = {
        "hypothesis_id": "H1",
        "target_mode": "volatility_5d",
        "logical_op": "AND",
        "clauses": [
            {"feature": "delivery_pct", "operator": "GREATER_THAN", "threshold_q1": 0.60},
            {"feature": "vix_close", "operator": "GREATER_THAN", "threshold_q1": 0.50},
        ],
    }
    h2 = {
        "hypothesis_id": "H2",
        "target_mode": "volatility_5d",
        "logical_op": "AND",
        "clauses": [
            {"feature": "vix_close", "operator": "GREATER_THAN", "threshold_q1": 0.50},
            {"feature": "delivery_pct", "operator": "GREATER_THAN", "threshold_q1": 0.60},
        ],
    }
    # Clauses in different order should yield identical canonical key
    assert get_ast_canonical_key(h1) == get_ast_canonical_key(h2)


def test_behavioral_clustering_planted_overlap():
    hypotheses = [
        {
            "hypothesis_id": "H_001",
            "target_mode": "volatility_5d",
            "logical_op": "AND",
            "clauses": [{"feature": "delivery_pct", "operator": "GREATER_THAN", "threshold_q1": 0.60}],
        },
        {
            "hypothesis_id": "H_002",
            "target_mode": "volatility_5d",
            "logical_op": "AND",
            "clauses": [{"feature": "delivery_pct", "operator": "GREATER_THAN", "threshold_q1": 0.65}],
        },
        {
            "hypothesis_id": "H_003",
            "target_mode": "volatility_5d",
            "logical_op": "AND",
            "clauses": [{"feature": "vix_close", "operator": "LESS_THAN", "threshold_q1": 0.20}],
        },
    ]

    # H_001 triggers on days {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    # H_002 triggers on days {10, 11, 12, 13, 14, 15, 16, 17, 18} (9 / 10 overlap -> Jaccard = 0.90 > 0.80)
    # H_003 triggers on days {1, 2, 3} (0 overlap -> Jaccard = 0.0)
    trigger_sets = {
        "H_001": {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        "H_002": {10, 11, 12, 13, 14, 15, 16, 17, 18},
        "H_003": {1, 2, 3},
    }

    result = cluster_hypotheses_behaviorally(hypotheses, trigger_sets, jaccard_threshold=0.80)

    assert result["total_input_hypotheses"] == 3
    assert result["struct_unique_count"] == 3
    assert result["behavioral_clusters_count"] == 2

    # H_001 and H_002 must belong to the same cluster
    assert result["hypothesis_to_cluster"]["H_001"] == result["hypothesis_to_cluster"]["H_002"]
    # H_003 must belong to a different cluster
    assert result["hypothesis_to_cluster"]["H_001"] != result["hypothesis_to_cluster"]["H_003"]
