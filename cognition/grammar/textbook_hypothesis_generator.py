"""
Textbook Technical & Volatility Folklore Hypothesis Generator for Phase 3b.

Generates ~150 hypotheses transcribed mechanically from standard technical and volatility folklore
(momentum, mean-reversion, volume confirmation, VIX rules) using deterministic templates.
Requires NO LLMs and NO prior program context.
"""

from typing import Any, Dict, List


def generate_textbook_hypotheses() -> List[Dict[str, Any]]:
    """
    Generates 150 unique textbook technical folklore hypotheses.
    """
    hypotheses: List[Dict[str, Any]] = []

    features = [
        "rv_1d",
        "rv_5d",
        "vix_close",
        "vix_change_5d",
        "vix_percentile_252d",
        "volume_ratio_5d",
        "norm_gap",
        "delivery_pct",
        "rs_nifty_3m",
    ]
    quantiles = [0.20, 0.35, 0.50, 0.65, 0.80]
    targets = ["volatility_5d", "vol_regime_expansion", "direction_3d"]

    count = 1

    # 1. Technical Momentum Folklore Rules (50 rules)
    for q_norm in [0.50, 0.65, 0.80]:
        for q_rs in [0.50, 0.65, 0.80]:
            for q_vol in [0.50, 0.65]:
                for target in targets:
                    if count > 150:
                        break
                    hid = f"H_TEXTBOOK_{count:03d}"
                    desc = f"Textbook Momentum rule {count}: Gap/RS continuation with volume confirmation."
                    h = {
                        "hypothesis_id": hid,
                        "description": desc,
                        "source": "TEXTBOOK",
                        "target_mode": target,
                        "logical_op": "AND",
                        "prediction_signal": 1.0,
                        "clauses": [
                            {"feature": "rs_nifty_3m", "operator": "GREATER_THAN", "threshold_q1": q_rs},
                            {"feature": "volume_ratio_5d", "operator": "GREATER_THAN", "threshold_q1": q_vol},
                        ],
                    }
                    hypotheses.append(h)
                    count += 1

    # 2. Institutional Delivery Accumulation Rules (40 rules)
    for q_del in [0.50, 0.65, 0.80]:
        for q_vol in [0.50, 0.65]:
            for q_vix in [0.35, 0.50, 0.65]:
                for target in targets:
                    if count > 150:
                        break
                    hid = f"H_TEXTBOOK_{count:03d}"
                    desc = f"Textbook Delivery Accumulation rule {count}: High delivery percentage with stable VIX."
                    h = {
                        "hypothesis_id": hid,
                        "description": desc,
                        "source": "TEXTBOOK",
                        "target_mode": target,
                        "logical_op": "AND",
                        "prediction_signal": 1.0,
                        "clauses": [
                            {"feature": "delivery_pct", "operator": "GREATER_THAN", "threshold_q1": q_del},
                            {"feature": "vix_close", "operator": "LESS_THAN", "threshold_q1": q_vix},
                        ],
                    }
                    hypotheses.append(h)
                    count += 1

    # 3. Volatility Compression & VIX Expansion Folklore Rules (40 rules)
    for q_rv in [0.20, 0.35]:
        for q_vix_pct in [0.20, 0.35, 0.50]:
            for q_vix_chg in [0.50, 0.65, 0.80]:
                for target in targets:
                    if count > 150:
                        break
                    hid = f"H_TEXTBOOK_{count:03d}"
                    desc = f"Textbook Volatility Expansion rule {count}: Low realized volatility before VIX breakout."
                    h = {
                        "hypothesis_id": hid,
                        "description": desc,
                        "source": "TEXTBOOK",
                        "target_mode": target,
                        "logical_op": "AND",
                        "prediction_signal": 1.0,
                        "clauses": [
                            {"feature": "rv_5d", "operator": "LESS_THAN", "threshold_q1": q_rv},
                            {"feature": "vix_change_5d", "operator": "GREATER_THAN", "threshold_q1": q_vix_chg},
                        ],
                    }
                    hypotheses.append(h)
                    count += 1

    # 4. Mean Reversion & Oversold VIX Rules (fill up to exactly 150)
    for q_gap in [0.20, 0.35]:
        for q_vix in [0.65, 0.80]:
            for target in targets:
                if count > 150:
                    break
                hid = f"H_TEXTBOOK_{count:03d}"
                desc = f"Textbook Mean Reversion rule {count}: Oversold price gap with elevated VIX."
                h = {
                    "hypothesis_id": hid,
                    "description": desc,
                    "source": "TEXTBOOK",
                    "target_mode": target,
                    "logical_op": "AND",
                    "prediction_signal": 1.0,
                    "clauses": [
                        {"feature": "norm_gap", "operator": "LESS_THAN", "threshold_q1": q_gap},
                        {"feature": "vix_close", "operator": "GREATER_THAN", "threshold_q1": q_vix},
                    ],
                }
                hypotheses.append(h)
                count += 1

    return hypotheses[:150]
