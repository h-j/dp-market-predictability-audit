"""
Read-Side Unification of Per-Session Ledger Files.

Concatenates one or more per-session JSONL consultation ledgers into a single
unified record list consumable by `influence_trace.compute_influence_set()` and
`divergence_analyzer.analyze_divergence_and_influence()`.

Design Principles:
- `decision_id` is prefixed per session by default (`prefix_decision_ids=True`)
  to prevent collision of decision IDs across independent concurrent sessions.
- `object_structural_id` is left UNPREFIXED so that identical structural objects
  referenced across multiple sessions unify into a single object identity.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Union


def merge_session_ledgers(
    session_paths: List[Union[str, Path]],
    prefix_decision_ids: bool = True,
    session_ids: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Merge multiple session JSONL ledger files into a single unified record list.

    Args:
        session_paths: List of file paths to session ledger JSONL files.
        prefix_decision_ids: If True, prefixes decision_id in all records with a session identifier.
        session_ids: Optional custom session identifier strings corresponding to session_paths.
                     Defaults to "session_1", "session_2", etc. if None.

    Returns:
        List[Dict]: Ordered list of records merged in file-then-session order.

    Raises:
        FileNotFoundError: If any path in session_paths does not exist.
    """
    merged_records: List[Dict] = []

    for idx, path_input in enumerate(session_paths):
        path = Path(path_input)
        if not path.exists():
            raise FileNotFoundError(f"Session ledger file not found: '{path_input}'")

        if session_ids and idx < len(session_ids):
            prefix = session_ids[idx]
        else:
            prefix = f"session_{idx + 1}"

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if prefix_decision_ids and "decision_id" in record:
                    record["decision_id"] = f"{prefix}:{record['decision_id']}"

                merged_records.append(record)

    return merged_records
