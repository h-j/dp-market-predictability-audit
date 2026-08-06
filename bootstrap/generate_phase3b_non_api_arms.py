"""
Generates and freezes non-API arms for Phase 3b:
- R-FRESH (data/hypotheses/phase3b/r_fresh.json): seed=43, N=150
- TEXTBOOK (data/hypotheses/phase3b/textbook.json): N=150
- F-CONTEXT (data/hypotheses/phase3b/f_context.json): copied from claude_arm.json
"""

import json
import shutil
from pathlib import Path

from cognition.grammar.random_hypothesis_generator import RandomHypothesisGenerator
from cognition.grammar.textbook_hypothesis_generator import generate_textbook_hypotheses


def main():
    out_dir = Path("data/hypotheses/phase3b")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. R-FRESH (Control Arm, seed 43)
    rand_gen = RandomHypothesisGenerator(seed=43)
    r_fresh_asts = rand_gen.generate_hypotheses(count=150)
    r_fresh_data = [ast.to_dict() for ast in r_fresh_asts]

    r_fresh_path = out_dir / "r_fresh.json"
    with open(r_fresh_path, "w", encoding="utf-8") as f:
        json.dump(r_fresh_data, f, indent=2)
    print(f"✓ Saved R-FRESH arm ({len(r_fresh_data)} hypotheses) to {r_fresh_path}")

    # 2. TEXTBOOK (Exploratory Arm)
    textbook_data = generate_textbook_hypotheses()
    textbook_path = out_dir / "textbook.json"
    with open(textbook_path, "w", encoding="utf-8") as f:
        json.dump(textbook_data, f, indent=2)
    print(f"✓ Saved TEXTBOOK arm ({len(textbook_data)} hypotheses) to {textbook_path}")

    # 3. F-CONTEXT (Secondary Pilot Arm)
    pilot_src = Path("data/hypotheses/claude_arm.json")
    if not pilot_src.exists():
        pilot_src = Path("data/hypotheses/llm_arm.json")

    f_context_path = out_dir / "f_context.json"
    if pilot_src.exists():
        shutil.copy(pilot_src, f_context_path)
        print(f"✓ Copied pilot arm ({pilot_src}) to F-CONTEXT at {f_context_path}")
    else:
        print(f"⚠️ Pilot arm source not found!")


if __name__ == "__main__":
    main()
