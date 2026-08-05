"""
Hypothesis Arm Generation & Freezing Script (Phase 3 Rebuild).

1. Queries OllamaClient for 150 valid unique LLM hypotheses (H_LLM). Aborts if Ollama is unreachable.
2. Generates 150 valid unique Random hypotheses (H_Random, seed 42).
3. Freezes data/hypotheses/llm_arm.json, data/hypotheses/random_arm.json, and data/hypotheses/llm_generation_log.json.
"""

import json
from pathlib import Path
from typing import Dict, List

from cognition.grammar.llm_hypothesis_generator import SubstrateLLMHypothesisGenerator
from cognition.grammar.random_hypothesis_generator import RandomHypothesisGenerator


def generate_and_freeze_arms(data_dir: Path = None):
    """
    Generate, format, and freeze treatment and control hypothesis arms for audit.
    """
    print("=" * 80)
    print("PHASE 3 REBUILD: HYPOTHESIS ARM GENERATION & FREEZING FOR AUDIT")
    print("=" * 80)

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    hypotheses_dir = data_dir / "hypotheses"
    hypotheses_dir.mkdir(parents=True, exist_ok=True)

    # 1. Treatment Arm Generation (LLM via Ollama)
    print("\n[1/2] Querying Ollama for Treatment Arm (H_LLM, target=150 unique)...")
    llm_gen = SubstrateLLMHypothesisGenerator(seed=42)
    llm_hyps, llm_log = llm_gen.generate_llm_arm(target_count=150, max_attempts=400)

    total_llm_attempts = len(llm_log)
    malformed_count = sum(1 for entry in llm_log if "error_reason" in entry and entry["error_reason"] and "duplicate" not in entry["error_reason"].lower())
    duplicate_count = sum(1 for entry in llm_log if entry.get("status") == "DUPLICATE")
    malformed_rate_pct = (malformed_count / total_llm_attempts * 100.0) if total_llm_attempts > 0 else 0.0

    print(f"✓ Ollama Generation Summary:")
    print(f"  • Total Attempts       : {total_llm_attempts}")
    print(f"  • Valid Unique Saved   : {len(llm_hyps)}")
    print(f"  • Duplicate Count      : {duplicate_count}")
    print(f"  • Malformed / Rejected : {malformed_count} ({malformed_rate_pct:.1f}%)")

    # Save LLM Arm & Raw Log
    llm_arm_json = [h.to_dict() for h in llm_hyps]
    llm_arm_path = hypotheses_dir / "llm_arm.json"
    llm_arm_path.write_text(json.dumps(llm_arm_json, indent=2))
    print(f"✓ Saved frozen LLM arm ({len(llm_hyps)} hypotheses) to {llm_arm_path}")

    llm_log_path = hypotheses_dir / "llm_generation_log.json"
    llm_log_path.write_text(json.dumps(llm_log, indent=2))
    print(f"✓ Saved raw LLM generation log ({total_llm_attempts} entries) to {llm_log_path}")

    # 2. Control Arm Generation (Random Grammar, seed=42)
    print("\n[2/2] Generating Control Arm (H_Random, target=150 unique, seed=42)...")
    rand_gen = RandomHypothesisGenerator(seed=42)
    rand_hyps = rand_gen.generate_hypotheses(count=150)

    print(f"✓ Random Generation Summary:")
    print(f"  • Valid Unique Saved   : {len(rand_hyps)}")

    # Save Random Arm
    rand_arm_json = [h.to_dict() for h in rand_hyps]
    rand_arm_path = hypotheses_dir / "random_arm.json"
    rand_arm_path.write_text(json.dumps(rand_arm_json, indent=2))
    print(f"✓ Saved frozen Random arm ({len(rand_hyps)} hypotheses) to {rand_arm_path}")

    print("\n" + "=" * 80)
    print("HYPOTHESIS ARMS SUCCESSFULLY GENERATED & FROZEN FOR EXTERNAL AUDIT")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generate_and_freeze_arms()
