"""
Substrate LLM Hypothesis Generator (Phase 3 Rebuild).

Queries OllamaClient with domain grammar definition & feature glossary.
Parses strict-JSON output into HypothesisAST.
Enforces hard stop if Ollama client errors or is unreachable (fallbacks/mock generation prohibited).
Logs every attempt (malformed rate, duplicates) to data/hypotheses/llm_generation_log.json.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognition.grammar.hypothesis_grammar import (
    FEATURES,
    OPERATORS,
    QUANTILES,
    TARGET_MODES,
    ClauseAST,
    HypothesisAST,
)
from interfaces.ollama_client import OllamaClient

LLM_SYSTEM_PROMPT = """
You are an expert quantitative research scientist for a reflective cognition substrate.
Your task is to generate valid, testable trading hypotheses formatted STRICTLY as a single JSON object.

### Structural Grammar Rules:
1. "target_mode": Must be one of ["volatility_5d", "vol_regime_expansion", "direction_3d"].
2. "logical_op": Must be "AND" or "OR".
3. "prediction_signal": Must be 1.0 (bullish/expansion) or -1.0 (bearish/compression).
4. "clauses": Array of 1 to 3 clause objects. Each clause MUST contain:
   - "feature": Must be one of ["rv_1d", "rv_5d", "vix_close", "vix_change_5d", "vix_percentile_252d", "volume_ratio_5d", "norm_gap", "delivery_pct", "rs_nifty_3m"].
   - "operator": Must be one of ["GREATER_THAN", "LESS_THAN", "BETWEEN"].
   - "threshold_q1": Must be a quantile float from [0.20, 0.35, 0.50, 0.65, 0.80].
   - "threshold_q2": Optional quantile float (required if operator is BETWEEN, must be > threshold_q1).

### Feature Glossary:
- rv_1d / rv_5d: 1-day & 5-day daily return volatility magnitude.
- vix_close / vix_change_5d / vix_percentile_252d: India VIX level, 5d momentum, and 1-year rank.
- volume_ratio_5d: 5-day volume / 20-day volume average.
- norm_gap: Normalized open-to-prev-close gap relative to ATR.
- delivery_pct: Institutional delivery volume ratio.
- rs_nifty_3m: 3-month return relative to NIFTY benchmark.

### Strict JSON Output Example:
{
  "description": "High VIX Rank and low Volume Ratio implies short-term Volatility Compression",
  "target_mode": "volatility_5d",
  "logical_op": "AND",
  "prediction_signal": -1.0,
  "clauses": [
    {
      "feature": "vix_percentile_252d",
      "operator": "GREATER_THAN",
      "threshold_q1": 0.80,
      "threshold_q2": null
    },
    {
      "feature": "volume_ratio_5d",
      "operator": "LESS_THAN",
      "threshold_q1": 0.35,
      "threshold_q2": null
    }
  ]
}

Return ONLY the raw JSON object. Do not include markdown codeblocks or conversational text.
"""


class OllamaUnreachableError(Exception):
    """Raised when Ollama client connection fails or returns error."""
    pass


class SubstrateLLMHypothesisGenerator:
    """
    Treatment group generator calling Ollama per Phase 3 Rebuild spec.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.client = ollama_client if ollama_client is not None else OllamaClient(mode="auto", temperature=0.7, seed=seed)

    def generate_single_llm_hypothesis(self, attempt_idx: int) -> Tuple[Optional[HypothesisAST], Dict[str, Any]]:
        """
        Query Ollama for a single hypothesis, parse strict JSON, and validate against grammar G.
        Log record captures attempt index, raw response, status, and error details.
        """
        log_entry: Dict[str, Any] = {
            "attempt": attempt_idx + 1,
            "raw_response": None,
            "status": "FAILED",
            "error_reason": None,
            "hypothesis": None,
        }

        # Varied prompts across feature combinations to encourage unique hypothesis synthesis
        feature_combos = [
            ("vix_close", "rv_5d", "volatility_5d"),
            ("vix_percentile_252d", "volume_ratio_5d", "vol_regime_expansion"),
            ("delivery_pct", "norm_gap", "direction_3d"),
            ("rs_nifty_3m", "vix_change_5d", "volatility_5d"),
            ("rv_1d", "norm_gap", "vol_regime_expansion"),
            ("vix_close", "delivery_pct", "direction_3d"),
            ("volume_ratio_5d", "rs_nifty_3m", "volatility_5d"),
        ]
        c1, c2, target_m = feature_combos[attempt_idx % len(feature_combos)]

        user_prompt = (
            f"Generate unique hypothesis proposal #{attempt_idx + 1}. Primary features: {c1} and {c2} "
            f"targeting mode: {target_m}. Iteration seed offset: {attempt_idx * 31 + 7}."
        )

        full_prompt = f"{LLM_SYSTEM_PROMPT}\n\n{user_prompt}"

        try:
            client = OllamaClient(mode="auto", temperature=0.7, seed=(self.seed + attempt_idx * 13) % 10000)
            raw_resp = client.generate(full_prompt, json_format=True)
            log_entry["raw_response"] = raw_resp
        except Exception as e:
            log_entry["error_reason"] = f"Ollama client error: {e}"
            raise OllamaUnreachableError(f"HARD STOP: Ollama client error or unreachable: {e}") from e

        if not raw_resp or not isinstance(raw_resp, str) or len(raw_resp.strip()) == 0:
            log_entry["error_reason"] = "Empty response from Ollama"
            return None, log_entry

        # Clean JSON from response (remove markdown codeblocks if present)
        cleaned_json = raw_resp.strip()
        if "```json" in cleaned_json:
            cleaned_json = re.sub(r"```json\s*", "", cleaned_json)
            cleaned_json = re.sub(r"```\s*$", "", cleaned_json)
        elif "```" in cleaned_json:
            cleaned_json = re.sub(r"```\s*", "", cleaned_json)

        cleaned_json = cleaned_json.strip()

        try:
            data = json.loads(cleaned_json)
        except Exception as parse_err:
            log_entry["error_reason"] = f"JSON parse error: {parse_err}"
            return None, log_entry

        # Validate JSON against Grammar G
        val_success, err_msg, ast = self._validate_and_build_ast(data, attempt_idx)
        if not val_success:
            log_entry["error_reason"] = f"Grammar validation error: {err_msg}"
            return None, log_entry

        log_entry["status"] = "SUCCESS"
        log_entry["hypothesis"] = ast.to_dict()
        return ast, log_entry

    def _validate_and_build_ast(self, data: Dict[str, Any], attempt_idx: int) -> Tuple[bool, str, Optional[HypothesisAST]]:
        if not isinstance(data, dict):
            return False, "Output must be a JSON object", None

        target_mode = data.get("target_mode")
        if target_mode not in TARGET_MODES:
            return False, f"Invalid target_mode: {target_mode}", None

        logical_op = str(data.get("logical_op", "AND")).upper()
        if logical_op not in ["AND", "OR"]:
            return False, f"Invalid logical_op: {logical_op}", None

        pred_sig = float(data.get("prediction_signal", 1.0))
        if pred_sig not in [1.0, -1.0]:
            return False, f"Invalid prediction_signal: {pred_sig}", None

        clauses_data = data.get("clauses", [])
        if not isinstance(clauses_data, list) or len(clauses_data) == 0:
            return False, "Clauses must be a non-empty array", None

        parsed_clauses: List[ClauseAST] = []
        for c_data in clauses_data[:3]:
            feat = c_data.get("feature")
            if feat not in FEATURES:
                return False, f"Invalid feature in clause: {feat}", None

            op = c_data.get("operator")
            if op not in OPERATORS:
                return False, f"Invalid operator in clause: {op}", None

            q1 = float(c_data.get("threshold_q1", 0.50))
            if q1 not in QUANTILES:
                q1 = float(min(QUANTILES, key=lambda x: abs(x - q1)))

            q2 = c_data.get("threshold_q2")
            if q2 is not None:
                q2 = float(q2)
                if q2 not in QUANTILES:
                    q2 = float(min(QUANTILES, key=lambda x: abs(x - q2)))

            parsed_clauses.append(ClauseAST(feature=feat, operator=op, threshold_q1=q1, threshold_q2=q2))

        desc = str(data.get("description", f"Ollama Hypothesis {attempt_idx+1}"))
        h_id = f"H_LLM_{attempt_idx+1:03d}"

        ast = HypothesisAST(
            hypothesis_id=h_id,
            description=desc,
            source="LLM",
            target_mode=target_mode,
            clauses=parsed_clauses,
            logical_op=logical_op,
            prediction_signal=pred_sig,
        )
        return True, "", ast

    def generate_llm_arm(
        self, target_count: int = 150, max_attempts: int = 400
    ) -> Tuple[List[HypothesisAST], List[Dict[str, Any]]]:
        """
        Generate 150 valid unique hypotheses by calling Ollama in a parse-or-reject loop up to 400 attempts.
        Aborts with error if Ollama is unreachable.
        """
        valid_hypotheses: List[HypothesisAST] = []
        seen_fingerprints = set()
        generation_log: List[Dict[str, Any]] = []

        attempt = 0
        while len(valid_hypotheses) < target_count and attempt < max_attempts:
            ast, log_entry = self.generate_single_llm_hypothesis(attempt)
            attempt += 1

            if ast is not None:
                # Check uniqueness by AST fingerprint
                fingerprint = (
                    ast.target_mode,
                    ast.logical_op,
                    ast.prediction_signal,
                    tuple(sorted([(c.feature, c.operator, c.threshold_q1, c.threshold_q2) for c in ast.clauses])),
                )

                if fingerprint in seen_fingerprints:
                    log_entry["status"] = "DUPLICATE"
                    log_entry["error_reason"] = "Exact duplicate of previously generated hypothesis"
                else:
                    seen_fingerprints.add(fingerprint)
                    ast.hypothesis_id = f"H_LLM_{len(valid_hypotheses)+1:03d}"
                    valid_hypotheses.append(ast)
                    log_entry["status"] = "SUCCESS_UNIQUE"

            generation_log.append(log_entry)

        if len(valid_hypotheses) < target_count:
            print(f"Warning: Generated {len(valid_hypotheses)} valid unique hypotheses after {attempt} attempts (target: {target_count}).")

        return valid_hypotheses, generation_log
