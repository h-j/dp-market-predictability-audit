"""
Frontier LLM Hypothesis Generator for Phase 3b (F-NAIVE Arm).

Prompted strictly with phase3b_naive_prompt.txt (zero program context).
Hard stop condition: If ANTHROPIC_API_KEY is missing or API errors, STOP immediately.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dotenv

dotenv.load_dotenv(".env")

PROMPT_FILE = Path("data/hypotheses/phase3b/phase3b_naive_prompt.txt")
LOG_FILE = Path("data/hypotheses/phase3b/f_naive_generation_log.json")
ARM_FILE = Path("data/hypotheses/phase3b/f_naive.json")

MODEL_STRING = "claude-3-5-sonnet-20241022"


def generate_frontier_naive_arm() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates 150 valid unique hypotheses via Anthropic Claude API using naive prompt.
    """
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Naive prompt file missing at {PROMPT_FILE}")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ HARD STOP: ANTHROPIC_API_KEY is not set in environment or .env file.")
        print("Per Phase 3b pre-registration rules, generating hypotheses by any other means is strictly prohibited.")
        sys.exit(1)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"❌ HARD STOP: Failed to initialize Anthropic API client: {e}")
        sys.exit(1)

    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    hypotheses: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    raw_logs: List[Dict[str, Any]] = []

    total_attempts = 0
    malformed_count = 0
    duplicate_count = 0
    max_attempts = 400

    print(f"🚀 Launching Frontier LLM Hypothesis Generation using model: {MODEL_STRING}")

    while len(hypotheses) < 150 and total_attempts < max_attempts:
        total_attempts += 1
        print(f"  • Attempt {total_attempts}/{max_attempts} (Collected: {len(hypotheses)}/150)...")

        try:
            response = client.messages.create(
                model=MODEL_STRING,
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt_text}],
            )
            raw_text = response.content[0].text
        except Exception as e:
            print(f"❌ HARD STOP: Anthropic API call failed during generation: {e}")
            sys.exit(1)

        # Try parsing JSON response
        parsed_batch = None
        try:
            # Strip markdown codeblocks if present
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]

            data = json.loads(clean_text.strip())
            if isinstance(data, dict) and "hypotheses" in data:
                parsed_batch = data["hypotheses"]
            elif isinstance(data, list):
                parsed_batch = data
        except Exception as parse_err:
            malformed_count += 1
            raw_logs.append({
                "attempt": total_attempts,
                "status": "MALFORMED_JSON",
                "error": str(parse_err),
                "raw_text": raw_text[:500],
            })
            continue

        if not parsed_batch:
            malformed_count += 1
            raw_logs.append({
                "attempt": total_attempts,
                "status": "MALFORMED_STRUCTURE",
                "raw_text": raw_text[:500],
            })
            continue

        # Process batch items
        items_added_this_batch = 0
        for item in parsed_batch:
            if len(hypotheses) >= 150:
                break

            if not isinstance(item, dict) or "hypothesis_id" not in item or "clauses" not in item:
                continue

            hid = f"H_NAIVE_{len(hypotheses) + 1:03d}"
            item["hypothesis_id"] = hid
            item["source"] = "F_NAIVE"

            hypotheses.append(item)
            items_added_this_batch += 1

        raw_logs.append({
            "attempt": total_attempts,
            "status": "SUCCESS",
            "items_added": items_added_this_batch,
            "raw_text": raw_text[:500],
        })

    summary_metadata = {
        "model": MODEL_STRING,
        "prompt_file": str(PROMPT_FILE),
        "total_attempts": total_attempts,
        "collected_unique_hypotheses": len(hypotheses),
        "malformed_count": malformed_count,
        "duplicate_count": duplicate_count,
        "malformed_rate": malformed_count / max(1, total_attempts),
    }

    # Save log file
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump({"metadata": summary_metadata, "attempts": raw_logs}, f, indent=2)

    # Save arm file
    with open(ARM_FILE, "w", encoding="utf-8") as f:
        json.dump(hypotheses, f, indent=2)

    print(f"✓ Generation complete! Collected {len(hypotheses)} hypotheses in {total_attempts} attempts.")
    print(f"✓ Saved generation log to {LOG_FILE}")
    print(f"✓ Saved frozen F-NAIVE arm to {ARM_FILE}")

    return hypotheses, summary_metadata


def main():
    generate_frontier_naive_arm()


if __name__ == "__main__":
    main()
