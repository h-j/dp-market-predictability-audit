"""
Unit & Integration Tests for merge_session_ledgers.py (Step 3).

Verifies:
1. Merging two session ledger files produces a combined list in file-then-session order.
2. Default prefix_decision_ids=True prevents decision_id collisions across sessions.
3. object_structural_id remains unprefixed so shared objects unify across sessions.
4. Nonexistent file path raises FileNotFoundError containing the path.
5. Integration test: merged session records fed into influence_trace.compute_influence_set()
   correctly resolve influence spanning decisions across both sessions.
"""
import pytest
from pathlib import Path
import pytest

from dp.domain.dp_taxonomy import DP_OBJECT_KINDS, DP_ROLES
from dp.observability.consultation_ledger import ConsultationLedger
from dp.observability.merge_session_ledgers import merge_session_ledgers
from dp.observability.influence_trace import compute_influence_set


def test_merge_two_session_ledgers_order(tmp_path):
    """1. Merging two session ledgers produces combined list in file-then-session order."""
    file1 = tmp_path / "session1.jsonl"
    file2 = tmp_path / "session2.jsonl"

    ledger1 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file1,
    )
    ledger1.record_consultation("0:dec:0", "obj:1", "theory", "prompt_context")
    ledger1.record_decision("0:dec:0", "Output 1", day=0)

    ledger2 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file2,
    )
    ledger2.record_consultation("0:dec:1", "obj:2", "theory", "prompt_context")
    ledger2.record_decision("0:dec:1", "Output 2", day=0)

    records = merge_session_ledgers([file1, file2])

    assert len(records) == 4
    # File 1 records first, then File 2 records
    assert records[0]["kind"] == "consultation"
    assert records[0]["decision_id"] == "session_1:0:dec:0"
    assert records[1]["kind"] == "decision"
    assert records[1]["decision_id"] == "session_1:0:dec:0"

    assert records[2]["kind"] == "consultation"
    assert records[2]["decision_id"] == "session_2:0:dec:1"
    assert records[3]["kind"] == "decision"
    assert records[3]["decision_id"] == "session_2:0:dec:1"


def test_prefix_decision_ids_prevents_collision(tmp_path):
    """2. Default prefix_decision_ids=True prevents decision_id collisions when raw IDs match."""
    file1 = tmp_path / "s1.jsonl"
    file2 = tmp_path / "s2.jsonl"

    # Both sessions use raw decision_id "0:dec:0"
    l1 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file1,
    )
    l1.record_consultation("0:dec:0", "obj:1", "theory", "prompt_context")

    l2 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file2,
    )
    l2.record_consultation("0:dec:0", "obj:2", "theory", "prompt_context")

    merged = merge_session_ledgers([file1, file2])

    dec_ids = [r["decision_id"] for r in merged]
    assert dec_ids[0] == "session_1:0:dec:0"
    assert dec_ids[1] == "session_2:0:dec:0"
    assert dec_ids[0] != dec_ids[1]  # No collision


def test_object_structural_id_remains_unprefixed(tmp_path):
    """3. object_structural_id remains unprefixed so shared objects unify across sessions."""
    file1 = tmp_path / "s1.jsonl"
    file2 = tmp_path / "s2.jsonl"

    # Both sessions reference the exact same object_structural_id "shared_doc:123"
    l1 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file1,
    )
    l1.record_consultation("0:dec:0", "shared_doc:123", "theory", "prompt_context")

    l2 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file2,
    )
    l2.record_consultation("0:dec:1", "shared_doc:123", "theory", "prompt_context")

    merged = merge_session_ledgers([file1, file2])

    # object_structural_id is identical in both records (unprefixed)
    assert merged[0]["object_structural_id"] == "shared_doc:123"
    assert merged[1]["object_structural_id"] == "shared_doc:123"


def test_nonexistent_file_raises_filenotfounderror(tmp_path):
    """4. Passing a nonexistent path raises FileNotFoundError containing the path."""
    bad_path = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError, match="does_not_exist.jsonl"):
        merge_session_ledgers([bad_path])


def test_integration_influence_trace_across_merged_sessions(tmp_path):
    """5. Integration test: merged session records fed into influence_trace span decisions from both sessions."""
    file1 = tmp_path / "session_a.jsonl"
    file2 = tmp_path / "session_b.jsonl"

    # Session A: decision 0:dec:A consults target object "shared_knowledge:1"
    l1 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file1,
    )
    l1.record_consultation("0:dec:A", "shared_knowledge:1", "theory", "prompt_context")
    l1.record_decision("0:dec:A", "Session A output", day=0)

    # Session B: decision 0:dec:B also consults target object "shared_knowledge:1"
    l2 = ConsultationLedger(
        valid_object_kinds=DP_OBJECT_KINDS,
        valid_roles=DP_ROLES,
        output_path=file2,
    )
    l2.record_consultation("0:dec:B", "shared_knowledge:1", "theory", "gate")
    l2.record_decision("0:dec:B", "Session B output", day=0)

    merged_records = merge_session_ledgers([file1, file2], session_ids=["sessA", "sessB"])

    trace = compute_influence_set(merged_records, "shared_knowledge:1")

    assert trace["target_object_id"] == "shared_knowledge:1"
    assert trace["total_influenced"] == 2
    assert "sessA:0:dec:A" in trace["direct_consultations"]
    assert "sessB:0:dec:B" in trace["direct_consultations"]
