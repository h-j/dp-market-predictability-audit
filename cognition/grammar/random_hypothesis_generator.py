"""
Random Hypothesis Generator (Control Arm).

Generates N valid unique hypotheses uniformly sampled from grammar G under fixed random seed (seed=42).
Collapses exact duplicates within the arm and formats sequential IDs H_RAND_001 to H_RAND_N.
"""

from typing import List, Set, Tuple

import numpy as np

from cognition.grammar.hypothesis_grammar import (
    FEATURES,
    OPERATORS,
    QUANTILES,
    TARGET_MODES,
    ClauseAST,
    HypothesisAST,
)


class RandomHypothesisGenerator:
    """
    Control arm generator producing 150 unique random hypotheses uniformly sampled from grammar G.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_random_clause(self) -> ClauseAST:
        feat = str(self.rng.choice(FEATURES))
        op = str(self.rng.choice(OPERATORS))
        q1 = float(self.rng.choice(QUANTILES))
        q2 = None
        if op == "BETWEEN":
            higher_qs = [q for q in QUANTILES if q > q1]
            q2 = float(self.rng.choice(higher_qs)) if higher_qs else min(1.0, q1 + 0.2)
        return ClauseAST(feature=feat, operator=op, threshold_q1=q1, threshold_q2=q2)

    def generate_hypotheses(self, count: int = 150) -> List[HypothesisAST]:
        """
        Generate N unique valid hypotheses without exact duplicates.
        """
        hypotheses: List[HypothesisAST] = []
        seen_fingerprints: Set[Tuple] = set()

        attempts = 0
        max_attempts = count * 20

        while len(hypotheses) < count and attempts < max_attempts:
            attempts += 1
            n_clauses = int(self.rng.choice([1, 2, 3]))
            clauses: List[ClauseAST] = []

            # Sample distinct features for clauses
            sampled_feats = self.rng.choice(FEATURES, size=n_clauses, replace=False)
            for feat in sampled_feats:
                op = str(self.rng.choice(OPERATORS))
                q1 = float(self.rng.choice(QUANTILES))
                q2 = None
                if op == "BETWEEN":
                    higher_qs = [q for q in QUANTILES if q > q1]
                    q2 = float(self.rng.choice(higher_qs)) if higher_qs else min(1.0, q1 + 0.2)

                clauses.append(ClauseAST(feature=str(feat), operator=op, threshold_q1=q1, threshold_q2=q2))

            target_mode = str(self.rng.choice(TARGET_MODES))
            logical_op = str(self.rng.choice(["AND", "OR"]))
            prediction_signal = float(self.rng.choice([1.0, -1.0]))

            fingerprint = (
                target_mode,
                logical_op,
                prediction_signal,
                tuple(sorted([(c.feature, c.operator, c.threshold_q1, c.threshold_q2) for c in clauses])),
            )

            if fingerprint in seen_fingerprints:
                continue

            seen_fingerprints.add(fingerprint)
            h_id = f"H_RAND_{len(hypotheses)+1:03d}"
            desc = f"Random Hypothesis {len(hypotheses)+1}: {' '.join([c.feature for c in clauses])} -> {target_mode}"

            hypotheses.append(
                HypothesisAST(
                    hypothesis_id=h_id,
                    description=desc,
                    source="RANDOM",
                    target_mode=target_mode,
                    clauses=clauses,
                    logical_op=logical_op,
                    prediction_signal=prediction_signal,
                )
            )

        return hypotheses
