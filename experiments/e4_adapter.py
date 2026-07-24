"""
Isolated E4 Experimental Adapter (PROMPT E4).

Evaluates isolated design changes for SD-008:
- Fix A: Empirical rule-strength probability estimator separation (s_hat) for predictions.
- Fix B: Scope-keyed belief representation (cause, effect, scope_key).

Supports 3 experimental arms:
- "E4a": Fix A only (s_hat prediction on unconditioned pairs)
- "E4b": Fix B only (scoped tuples (c, e, x) with E[Beta] prediction)
- "E4": Combined (scoped tuples (c, e, x) with s_hat prediction)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
from bench.synthworld.world import Scenario


@dataclass
class E4Hypothesis:
    cause: str
    effect: str
    scope_var: Optional[str] = None
    scope_val: Optional[int] = None

    @property
    def key(self) -> Tuple[str, str, Optional[Tuple[str, int]]]:
        if self.scope_var is not None:
            return (self.cause, self.effect, (self.scope_var, self.scope_val))
        return (self.cause, self.effect, None)

    @property
    def name(self) -> str:
        if self.scope_var is not None:
            return f"[{self.scope_var}={self.scope_val}] {self.cause} -> {self.effect}"
        return f"{self.cause} -> {self.effect}"


@dataclass
class E4ConfidenceState:
    alpha: float = 1.0
    beta: float = 1.0
    k_falsify: float = 3.0
    decay_lambda: float = 0.01
    triggers: int = 0
    supported_triggers: int = 0

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength_hat(self) -> float:
        """Laplace-smoothed empirical rule strength computed strictly from past trigger history."""
        if self.triggers == 0:
            return 0.50
        return (self.supported_triggers + 1.0) / (self.triggers + 2.0)

    def evolve_supported(self):
        self.alpha += 1.0
        self.triggers += 1
        self.supported_triggers += 1

    def evolve_contradicted(self):
        self.beta += self.k_falsify
        self.triggers += 1

    def decay(self):
        self.alpha = 1.0 + (self.alpha - 1.0) * (1.0 - self.decay_lambda)
        self.beta = 1.0 + (self.beta - 1.0) * (1.0 - self.decay_lambda)


class E4Adapter:
    """
    Substrate Adapter for E4 Design-Change Experiment.
    Implements Learner protocol required by bench.synthworld.harness.
    """

    def __init__(self, scenario: Scenario, arm: str = "E4", promotion_threshold: float = 0.50):
        self.scenario = scenario
        self.arm = arm
        self.name = f"DP/EkamNet-{arm}"
        self.promotion_threshold = promotion_threshold

        self.causes = list(scenario.drivers)
        self.effects = list(scenario.effects)

        # Build systematic candidate hypothesis space
        self.hypotheses: List[E4Hypothesis] = []

        # 1. Unconditioned pairs
        for c in self.causes:
            for e in self.effects:
                self.hypotheses.append(E4Hypothesis(cause=c, effect=e))

        # 2. Context-conditioned tuples (Fix B) if arm in ("E4b", "E4")
        if self.arm in ("E4b", "E4"):
            # Detect context variables from scenario (e.g. 'C')
            ctx_vars = [k for k in ("C", "ctx", "context") if k not in self.causes and k not in self.effects]
            if not ctx_vars:
                # Fallback: check scenario properties
                ctx_vars = ["C"]

            for c in self.causes:
                for e in self.effects:
                    for ctx in ctx_vars:
                        for val in (0, 1):
                            self.hypotheses.append(E4Hypothesis(cause=c, effect=e, scope_var=ctx, scope_val=val))

        # Initialize confidence states per hypothesis key
        self.states: Dict[Tuple[str, str, Optional[Tuple[str, int]]], E4ConfidenceState] = {
            h.key: E4ConfidenceState() for h in self.hypotheses
        }

        # Track active triggers from t to resolve outcome at t+1
        self.pending_triggers: Dict[Tuple[str, str, Optional[Tuple[str, int]]], bool] = {}

    def _is_scope_active(self, h: E4Hypothesis, events: Dict[str, Any]) -> bool:
        if h.scope_var is None:
            return True
        return events.get(h.scope_var) == h.scope_val

    def observe(self, t: int, events: Dict[str, Any]) -> None:
        """
        Observe event slice at step t.
        1. Resolves terminal outcomes for hypotheses triggered at step t-1.
        2. Evaluates step t activation for all hypotheses.
        """
        # Step 1: Resolve terminal outcomes from step t-1
        for key, was_triggered in self.pending_triggers.items():
            state = self.states[key]
            # key is (cause, effect, scope)
            effect_name = key[1]
            effect_occurred = bool(events.get(effect_name, 0))

            if was_triggered:
                if effect_occurred:
                    state.evolve_supported()
                else:
                    state.evolve_contradicted()
            else:
                state.decay()

        self.pending_triggers.clear()

        # Step 2: Evaluate triggers for step t (to be resolved at step t+1)
        for h in self.hypotheses:
            cause_active = bool(events.get(h.cause, 0))
            scope_active = self._is_scope_active(h, events)

            is_triggered = cause_active and scope_active
            self.pending_triggers[h.key] = is_triggered

    def predict(self, t: int, events: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict probability for each effect at step t given current events.
        Enforces strict t-1 historical boundary for s_hat calculations.
        """
        predictions = {}

        for e in self.effects:
            prob_neg = 1.0

            for h in self.hypotheses:
                if h.effect != e:
                    continue

                cause_active = bool(events.get(h.cause, 0))
                scope_active = self._is_scope_active(h, events)

                if not (cause_active and scope_active):
                    continue

                state = self.states[h.key]

                # Check promotion threshold E[Beta] >= 0.50
                if state.confidence < self.promotion_threshold:
                    continue

                # Determine predictive contribution p_h
                if self.arm in ("E4a", "E4"):
                    # Fix A: Use empirical rule strength estimator s_hat
                    p_h = state.strength_hat
                else:
                    # Fix B only (E4b): Use belief confidence E[Beta]
                    p_h = state.confidence

                prob_neg *= (1.0 - p_h)

            predictions[e] = 1.0 - prob_neg

        return predictions

    def beliefs(self) -> Dict[Tuple[str, str], float]:
        """
        Extract established beliefs (E[Beta] >= promotion_threshold).
        Returns dictionary mapping (cause, effect) -> max confidence across scopes.
        """
        result = {}
        for h in self.hypotheses:
            state = self.states[h.key]
            if state.confidence >= self.promotion_threshold:
                # Map to standard (cause, effect) taking max confidence across scope keys
                pair = (h.cause, h.effect)
                result[pair] = max(result.get(pair, 0.0), state.confidence)
        return result

