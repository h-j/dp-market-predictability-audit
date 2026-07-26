# Cognition Parameter & Gate Registration Tuning Changelog

All hyperparameter changes, threshold updates, and gate registrations must be logged in this file prior to execution.

---

## [1.0.0] - 2026-07-24

### Registered
- **Gate A Pre-Registration (`experiments/preregistration/gate_a.yaml`)**:
  - Registered formal E1 counterfactual influence gate criteria:
    - Preconditions: Ablation target Beta confidence > 0.65 with `evidence_count >= 5`; `substitution_count` and `reinvocation_count` must be reported.
    - PASS: `unpredicted_divergence` empty AND `verified_influence` non-empty.
    - INSTRUMENTATION_FAIL: `unpredicted_divergence` non-empty.
    - NULL: `observed_divergence` empty.
  - Registered formal E2 synthetic benchmark battery criteria:
    - Simultaneously requires: S2 `decoy_claims` rate < `FlatBayesian`'s AND S2 `discovery_precision` >= `FlatBayesian`'s; AND S3 `recovery_steps` < `FlatBayesian`'s AND S3 `collateral` <= `WindowedFrequency`'s; AND S1 Brier regret <= `FlatBayesian`'s.
    - Three-branch interpretation table registered verbatim (`PASS` / `FAIL` / `AMBIGUOUS`).

---

## [1.1.0] - 2026-07-26

### Voided & Re-Registered
- **Gate E4 Re-Registration (`experiments/preregistration/gate_e4_v2.yaml`)**:
  - Voided prior `gate_e4.yaml` verdict `PARTIAL_FIX_B` due to process defects (registration committed with results; non-failable H2 threshold `0.00`). Old file marked `VOID` in place.
  - Re-registered clean verification gate `experiments/preregistration/gate_e4_v2.yaml` prior to execution:
    - Frozen constants: $k_{\text{falsify}}=3.0$, $\lambda=0.01$, promotion threshold $=0.50$ (asserted at runtime).
    - H1 (Fix A calibration): PASS if DP S1 Brier regret $\le 0.010$ AND S3 Brier regret $\le 0.010$ vs oracle.
    - H2 (Fix B scoped discovery): PASS if DP S4 recall $\ge 0.90$ AND DP S4 precision $\ge 0.90$.
    - Precision Guard: Credited arm must maintain S1 precision $\ge 0.90$ AND S3 precision $\ge 0.90$.
    - Decoy Guard (S2): Decoy claims rate $\le 0.05$.
    - Four-branch interpretation table registered verbatim (`FIX_CONFIRMED` / `STRUCTURAL_CALIBRATION_BOUND` / `PARTIAL` / `STRUCTURAL` / `REGRESSION`).

