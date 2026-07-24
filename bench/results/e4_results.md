# E4 — Design-Change Experiment Results (20 Seeds)

Authoritative 20-seed synthetic battery evaluation for Milestone E4.

### Registered Gate E4 Branch Verdict: **[PARTIAL_FIX_B]**

**Expected Calibration Error (ECE - Combined Arm)**: `0.1395`

---

## Scenario S1 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0005 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0030 ± 0.0004 | 0.8833 ± 0.1631 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **ContextualBayesian** | 0.0005 ± 0.0002 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.0593 ± 0.0036 | 1.0000 ± 0.0000 | 0.4000 ± 0.2052 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0659 ± 0.0034 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0593 ± 0.0036 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

## Scenario S2 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0178 ± 0.0014 | 0.6583 ± 0.0373 | 1.0000 ± 0.0000 | 1.05 ± 0.22 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0076 ± 0.0009 | 0.9417 ± 0.1458 | 1.0000 ± 0.0000 | 0.20 ± 0.52 | N/A | N/A |
| **ContextualBayesian** | 0.0178 ± 0.0014 | 0.6583 ± 0.0373 | 1.0000 ± 0.0000 | 1.05 ± 0.22 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.0611 ± 0.0062 | 1.0000 ± 0.0000 | 0.4750 ± 0.1118 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0677 ± 0.0056 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0611 ± 0.0062 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

## Scenario S3 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 13.8 ± 55.9 | 0.0075 ± 0.0210 |
| **FlatBayesian** | 0.0313 ± 0.0011 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 1873.8 ± 175.4 | 0.0074 ± 0.0209 |
| **WindowedFrequency(w=200)** | 0.0038 ± 0.0003 | 0.9833 ± 0.0745 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 123.8 ± 68.6 | 0.0079 ± 0.0205 |
| **ContextualBayesian** | 0.0313 ± 0.0011 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 1873.8 ± 175.4 | 0.0074 ± 0.0209 |
| **DP/EkamNet-E4a** | 0.0578 ± 0.0023 | 1.0000 ± 0.0000 | 0.4750 ± 0.1970 | 0.00 ± 0.00 | 917.5 ± 505.2 | 0.0103 ± 0.0465 |
| **DP/EkamNet-E4b** | 0.0525 ± 0.0025 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 393.8 ± 139.3 | 0.0102 ± 0.0460 |
| **DP/EkamNet-E4** | 0.0578 ± 0.0023 | 0.3333 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | 917.5 ± 505.2 | 0.0103 ± 0.0465 |

## Scenario S4 Results (20 Seeds)

| Learner | Brier Regret (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Decoy Claims (mean ± std) | Recovery Steps (mean ± std) | Collateral (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TrueModel (oracle floor)** | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **FlatBayesian** | 0.0328 ± 0.0015 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **WindowedFrequency(w=200)** | 0.0358 ± 0.0012 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **ContextualBayesian** | 0.0005 ± 0.0002 | 0.6667 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4a** | 0.1173 ± 0.0028 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4b** | 0.0538 ± 0.0032 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |
| **DP/EkamNet-E4** | 0.0504 ± 0.0034 | 0.5000 ± 0.0000 | 1.0000 ± 0.0000 | 0.00 ± 0.00 | N/A | N/A |

