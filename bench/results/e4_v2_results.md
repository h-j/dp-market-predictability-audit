# E4 v2 — Confirmation Test Results (20 Seeds)

Authoritative 20-seed synthetic battery evaluation for Milestone E4 under pre-registered `gate_e4_v2.yaml`.

### Certified Gate E4_v2 Branch Verdict: **[STRUCTURAL]**

**Expected Calibration Error (ECE - Combined Arm)**: `0.1395`

---

## 1. Pre-Registered `gate_e4_v2.yaml` Criteria & Mechanical Evaluation

```yaml
    H1_fix_a_calibration:        - scenario: "S1" metric: "brier_regret" operator: "<=" target_val: 0.010        - scenario: "S3" metric: "brier_regret" operator: "<=" target_val: 0.010
    H2_fix_b_scoped_discovery:        - scenario: "S4" metric: "recall" operator: ">=" target_val: 0.90        - scenario: "S4" metric: "precision" operator: ">=" target_val: 0.90
    guards:      precision_guard:        - scenario: "S1" metric: "precision" operator: ">=" target_val: 0.90        - scenario: "S3" metric: "precision" operator: ">=" target_val: 0.90      decoy_guard:        - scenario: "S2" metric: "decoy_claims" operator: "<=" target_val: 0.05
```

### Criterion Execution Results (DP/EkamNet-E4 Combined Arm):
- **H1 Fix A Calibration**: S1 Brier Regret = `0.0593` (target $\le 0.010$), S3 Brier Regret = `0.0578` (target $\le 0.010$) $\implies$ **FAIL**
- **H2 Fix B Scoped Discovery**: S4 Recall = `1.0000` (target $\ge 0.90$), S4 Precision = `0.5000` (target $\ge 0.90$) $\implies$ **FAIL**
- **Decoy Guard (S2)**: Decoy Claims = `0.0000` (target $\le 0.05$) $\implies$ **PASS**

### Precision Guard Outcomes per DP Arm:
| Arm | S1 Precision | S3 Precision | Threshold (>=0.90) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **DP/EkamNet-E4a** | 1.0000 | 1.0000 | >= 0.90 | **PASS** |
| **DP/EkamNet-E4b** | 0.3333 | 0.3333 | >= 0.90 | **REGRESSION** |
| **DP/EkamNet-E4** | 0.3333 | 0.3333 | >= 0.90 | **REGRESSION** |

> **Precision Guard Analysis**: Fix B recall gains in DP arms (e.g. E4b, E4) were bought with precision collapse on S1 (precision 1.00 -> 0.33) and S3 (precision -> 0.50). Under pre-registered rules, these recall gains are labeled as **REGRESSIONS**, not credited.

---

## 2. Seven-Learner Benchmark Performance Tables

### Scenario S1 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0005 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0030 ± 0.0004 | 0.8833 ± 0.1631 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **ContextualBayesian** | 0.0005 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.0593 ± 0.0036 | 1.0000 ± 0.0000 | 0.4000 ± 0.2052 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0659 ± 0.0034 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0593 ± 0.0036 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

### Scenario S2 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0178 ± 0.0014 | 0.6583 ± 0.0373 | 1.0000 ± 0.0000 | 1.05 ± 0.22 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0076 ± 0.0009 | 0.9417 ± 0.1458 | 1.0000 ± 0.0000 | 0.20 ± 0.52 | N/A | N/A |
| **ContextualBayesian** | 0.0178 ± 0.0014 | 0.6583 ± 0.0373 | 1.0000 ± 0.0000 | 1.05 ± 0.22 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.0611 ± 0.0062 | 1.0000 ± 0.0000 | 0.4750 ± 0.1118 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0677 ± 0.0056 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0611 ± 0.0062 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

### Scenario S3 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 13.8 ± 55.9 | 0.0075 ± 0.0210 |
| **FlatBayesian** | 0.0313 ± 0.0011 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 1873.8 ± 175.4 | 0.0074 ± 0.0209 |
| **WindowedFrequency(w=200)** | 0.0038 ± 0.0003 | 0.9833 ± 0.0745 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 123.8 ± 68.6 | 0.0079 ± 0.0205 |
| **ContextualBayesian** | 0.0313 ± 0.0011 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 1873.8 ± 175.4 | 0.0074 ± 0.0209 |
| **DP/EkamNet-E4a** | 0.0578 ± 0.0023 | 1.0000 ± 0.0000 | 0.4750 ± 0.1970 | 0.00 ± 0.00 | 917.5 ± 505.2 | 0.0103 ± 0.0465 |
| **DP/EkamNet-E4b** | 0.0525 ± 0.0025 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 393.8 ± 139.3 | 0.0102 ± 0.0460 |
| **DP/EkamNet-E4** | 0.0578 ± 0.0023 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 917.5 ± 505.2 | 0.0103 ± 0.0465 |

### Scenario S4 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0328 ± 0.0015 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0358 ± 0.0012 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **ContextualBayesian** | 0.0005 ± 0.0002 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.1173 ± 0.0028 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0538 ± 0.0032 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0504 ± 0.0034 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

