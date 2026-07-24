# Pre-Registration Document: Milestone E4 Design-Change Experiment

**Registration Date**: 2026-07-24  
**Target Milestone**: Milestone E4 (SD-008 Design-Change Experiment)  
**Governance Scope**: Immutable Benchmark, Baselines, and Frozen Constants ($k_{\text{falsify}}=3.0, \lambda=0.01$, Promotion Threshold $=0.50$)

---

## 1. Background & Scientific Rationale

Following the execution of the 20-seed reference synthetic benchmark battery (`E2_v2`, `commit dc5502d`), the pre-registered `Gate A` evaluation returned **`AMBIGUOUS`**. Scientific debt entry **`SD-008`** recorded that the DP lifecycle underperforms simple Bayesian baselines on prediction quality across static calibration (S1), discovery recall (S1/S2), and scoped reasoning (S4).

Subsequent empirical mechanism diagnostic studies (`experiments/diagnostics/run_mechanism_diagnostics.py`) proved two distinct structural root causes:
1. **Calibration Gap (Fix A)**: The prediction engine consumed posterior belief confidence $E[\text{Beta}] = \frac{\alpha}{\alpha + \beta}$ directly in Noisy-OR prediction as a proxy for conditional outcome probability $P(E=1 \mid C=1)$, creating a persistent stating gap.
2. **Scoped Reasoning Gap (Fix B)**: The adapter enumerated unconditioned cause-effect pairs $(c, e)$, causing context-gated rules ($D1 \to E1$ when $C=1$) to suffer $k_{\text{falsify}}=3.0$ penalties on non-active context steps ($C=0$), suppressing their posteriors below the $0.50$ promotion threshold.

---

## 2. Pre-Registered Design Changes

The E4 experiment evaluates two isolated structural modifications:

### Fix A: Empirical Rule Strength Estimator ($\hat{s}$)
- **Belief Confidence Evolution**: Remains locked to the `ScoredConfidenceEngine` ($k_{\text{falsify}}=3.0, \lambda=0.01$, threshold $=0.50$). Belief confidence $E[\text{Beta}]$ governs belief promotion/demotion and lifecycle state only.
- **Probability Estimation**: For promoted beliefs ($E[\text{Beta}] \ge 0.50$), prediction intensity uses an empirical rule strength estimator $\hat{s}$:
  $$\hat{s}_i = \frac{\text{supported\_triggers}_i + 1}{\text{total\_triggers}_i + 2} \quad \text{(Laplace-smoothed from historical } t-1 \text{ validation records)}.$$
- **Predictive Consumption**:
  $$P(E = 1) = 1.0 - \prod_{i \in \text{promoted}} (1.0 - \hat{s}_i).$$

### Fix B: Scope-Keyed Belief Representation ($(c, e, x)$)
- Candidate belief keys are expanded from unconditioned pairs $(c, e)$ to context-conditioned tuples $(c, e, x)$, where $x \in \{\text{unconditioned}, C=0, C=1\}$.
- **Conditional Evaluation**: Rule triggers and validation updates are evaluated strictly when context predicate $x$ holds. On timesteps where context predicate $x$ does NOT hold, the rule is not triggered (zero $\alpha$/$\beta$ updates and zero $k_{\text{falsify}}$ penalties).

---

## 3. Experimental Protocol & Factorial Arms

The experiment evaluates 20 seeds (0..19) across all 4 benchmark scenarios ($T=3000, 3000, 4000, 4000$) for three experimental arms:
1. **`E4a` (Fix A Only)**: Evaluates estimator separation $\hat{s}$ on unconditioned pairs $(c, e)$.
2. **`E4b` (Fix B Only)**: Evaluates scope keying $(c, e, x)$ using belief confidence $E[\text{Beta}]$ for prediction.
3. **`E4` (Combined)**: Evaluates both Fix A ($\hat{s}$) and Fix B (scope keying) simultaneously.

---

## 4. Pre-Registered Gate E4 Pass Criteria

To achieve a **`PASS`** verdict, the combined **`E4`** arm must simultaneously satisfy:
1. **S1 Brier Regret**: $\le 0.0050$ vs Oracle (matching `FlatBayesian`).
2. **S4 Scoped Regret**: $\le 0.0100$ vs Oracle (matching `ContextualBayesian`).
3. **S4 Discovery Precision**: $= 1.00$.
4. **S4 Discovery Recall**: $= 1.00$.
5. **S2 Decoy Claims**: $= 0.00$ (No decoy regression guard).
6. **S2 Discovery Precision**: $= 1.00$ (No decoy regression guard).
7. **S1/S2 Discovery Recall**: $\ge 0.45$ (No recall regression guard).

---

## 5. Pre-Registered Three-Branch Interpretation Table

```yaml
three_branch_interpretation_table:
  PASS:
    consequence: "Separating probability estimation (Fix A) and keying beliefs by scope (Fix B) resolves S1 calibration regret and S4 scoped reasoning without decoy or recall regression."
    next_milestone: "Phase 2 scope abstraction and multi-symbol generalization."
  PARTIAL_FIX_A:
    consequence: "Fix A eliminates S1 calibration regret, but Fix B is insufficient for S4 scoped reasoning."
    action: "Register scope representation extension for Phase 2."
  PARTIAL_FIX_B:
    consequence: "Fix B resolves S4 scoped reasoning, but Fix A probability estimation requires prior smoothing."
    action: "Register probability estimator prior smoothing experiment."
  NULL:
    consequence: "Neither Fix A nor Fix B resolves target metrics."
    action: "Maintain SD-008 OPEN and disproven hypotheses."
```
