"""
Unit & Integration Tests for Consultation Ledger & Read-Side Provenance (PROMPT E0a).

Verifies:
1. Decision ID formatting and determinism.
2. Byte-stability of consultation_ledger.jsonl (NO wall-clock timestamps).
3. Transitive influence chain resolution in influence_trace.py on synthetic ledger.
4. 5-day replay produces consultation ledger with entries from theory generation, reflection, and gate sites.
5. Two identical replays produce 100% byte-identical consultation ledgers.
"""
import asyncio
import json
from pathlib import Path
import pytest

from cognition.schemas.identity import build_structural_id
from dp.observability.consultation_ledger import (
    ConsultationLedger,
    record_consultation,
    set_active_consultation_ledger,
)
from dp.observability.influence_trace import compute_influence_set, parse_consultation_ledger
from market.replay.replay_engine import ReplayExecutor


def test_decision_ids_and_byte_stability(tmp_path):
    """Verify structural ID formatting and byte-stability of consultation ledger output."""
    ledger_path1 = tmp_path / "run1" / "consultation_ledger.jsonl"
    ledger1 = ConsultationLedger(output_path=ledger_path1)

    dec_id_0 = build_structural_id(0, "theory", 0)
    assert dec_id_0 == "0:theory:0"

    ledger1.record_consultation(
        decision_id="0:theory:0",
        object_structural_id="0:regime_memory:0",
        object_kind="regime_memory",
        role="prompt_context",
    )
    ledger1.record_decision(
        decision_id="0:theory:0",
        output_content="Sample theory output text",
        day=0,
    )

    ledger_path2 = tmp_path / "run2" / "consultation_ledger.jsonl"
    ledger2 = ConsultationLedger(output_path=ledger_path2)

    ledger2.record_consultation(
        decision_id="0:theory:0",
        object_structural_id="0:regime_memory:0",
        object_kind="regime_memory",
        role="prompt_context",
    )
    ledger2.record_decision(
        decision_id="0:theory:0",
        output_content="Sample theory output text",
        day=0,
    )

    bytes1 = ledger_path1.read_bytes()
    bytes2 = ledger_path2.read_bytes()

    assert bytes1 == bytes2, "Consultation ledgers from identical inputs must be 100% byte-identical"


def test_influence_trace_transitive_chain(tmp_path):
    """Verify transitive chain resolution in influence_trace on synthetic ledger fixture."""
    ledger_path = tmp_path / "synthetic_ledger.jsonl"
    ledger = ConsultationLedger(output_path=ledger_path)

    # 0:theory:0 directly consults 0:regime_memory:0
    ledger.record_consultation(
        decision_id="0:theory:0",
        object_structural_id="0:regime_memory:0",
        object_kind="regime_memory",
        role="prompt_context",
    )
    ledger.record_decision(decision_id="0:theory:0", output_content="Theory 0", day=0)

    # 1:reflection:0 consults 0:theory:0 (1-hop transitive taint)
    ledger.record_consultation(
        decision_id="1:reflection:0",
        object_structural_id="0:theory:0",
        object_kind="theory",
        role="prompt_context",
    )
    ledger.record_decision(decision_id="1:reflection:0", output_content="Reflection 1", day=1)

    # 2:theory:0 consults 1:reflection:0 (2-hop transitive taint)
    ledger.record_consultation(
        decision_id="2:theory:0",
        object_structural_id="1:reflection:0",
        object_kind="theory",
        role="prompt_context",
    )
    ledger.record_decision(decision_id="2:theory:0", output_content="Theory 2", day=2)

    records = parse_consultation_ledger(ledger_path)
    res = compute_influence_set(records, "0:regime_memory:0")

    assert res["target_object_id"] == "0:regime_memory:0"
    assert "0:theory:0" in res["direct_consultations"]
    assert "1:reflection:0" in res["influenced_decisions"]
    assert "2:theory:0" in res["influenced_decisions"]
    assert res["total_influenced"] == 3


@pytest.mark.requires_ollama
def test_5day_replay_consultation_ledger_and_reproducibility():

    """
    Run 5-day replay, verifying:
    1. consultation_ledger.jsonl is created with entries from theory, reflection, and gate sites.
    2. Two identical 5-day replays produce 100% byte-identical consultation ledgers.
    """
    exec1 = ReplayExecutor(max_days=5, quiet=True)
    exec1.execute(emit_summary=False)

    ledger_path1 = exec1.run_dir / "consultation_ledger.jsonl"
    assert ledger_path1.exists(), f"Consultation ledger not created at {ledger_path1}"

    records1 = parse_consultation_ledger(ledger_path1)
    kinds = {r.get("kind") for r in records1}
    roles = {r.get("role") for r in records1 if r.get("kind") == "consultation"}
    obj_kinds = {r.get("object_kind") for r in records1 if r.get("kind") == "consultation"}

    assert "consultation" in kinds
    assert "decision" in kinds
    assert "prompt_context" in roles or "gate" in roles
    assert "theory" in obj_kinds or "regime_memory" in obj_kinds

    # Second run to test 100% byte stability across identical replays
    exec2 = ReplayExecutor(max_days=5, quiet=True)
    exec2.execute(emit_summary=False)
    ledger_path2 = exec2.run_dir / "consultation_ledger.jsonl"

    bytes1 = ledger_path1.read_bytes()
    bytes2 = ledger_path2.read_bytes()

    assert bytes1 == bytes2, "Two identical 5-day replays must produce 100% byte-identical consultation ledgers"

    # Execute influence trace on actual replay ledger
    influence_res = compute_influence_set(records1, "0:regime_memory:0")
    assert "target_object_id" in influence_res
    assert influence_res["total_influenced"] >= 1


def test_injectable_vocabulary_and_provenance_method(tmp_path):
    """
    Verify that ConsultationLedger accepts custom injectable object_kind/role sets
    and records the provenance_method field.
    """
    custom_ledger = ConsultationLedger(
        output_path=tmp_path / "custom.jsonl",
        valid_object_kinds={"prompt_segment"},
        valid_roles={"context_window"},
    )

    # Custom ledger accepts 'prompt_segment' and 'context_window'
    rec1 = custom_ledger.record_consultation(
        decision_id="0:dec:0",
        object_structural_id="seg:1",
        object_kind="prompt_segment",
        role="context_window",
        provenance_method="ablation_inferred",
    )
    assert rec1["object_kind"] == "prompt_segment"
    assert rec1["role"] == "context_window"
    assert rec1["provenance_method"] == "ablation_inferred"

    # Custom ledger rejects DP's default 'theory' kind
    with pytest.raises(ValueError, match="Invalid object_kind 'theory'"):
        custom_ledger.record_consultation(
            decision_id="0:dec:1",
            object_structural_id="0:theory:0",
            object_kind="theory",
            role="context_window",
        )

    # Default-constructed ledger accepts 'theory' and rejects 'prompt_segment'
    default_ledger = ConsultationLedger(output_path=tmp_path / "default.jsonl")
    rec2 = default_ledger.record_consultation(
        decision_id="0:dec:0",
        object_structural_id="0:theory:0",
        object_kind="theory",
        role="prompt_context",
    )
    assert rec2["object_kind"] == "theory"
    assert rec2["provenance_method"] == "observed"  # default provenance_method

    with pytest.raises(ValueError, match="Invalid object_kind 'prompt_segment'"):
        default_ledger.record_consultation(
            decision_id="0:dec:1",
            object_structural_id="seg:1",
            object_kind="prompt_segment",
            role="prompt_context",
        )


def test_async_contextvars_isolation(tmp_path):
    """
    Verify that set_active_consultation_ledger uses contextvars for per-async-task
    isolation without cross-contamination.
    """
    ledger_a = ConsultationLedger(output_path=tmp_path / "ledger_a.jsonl")
    ledger_b = ConsultationLedger(output_path=tmp_path / "ledger_b.jsonl")

    async def task_worker(ledger: ConsultationLedger, task_id_str: str):
        set_active_consultation_ledger(ledger)
        await asyncio.sleep(0.01)  # Yield control to encourage race condition if un-isolated
        record_consultation(
            decision_id=f"{task_id_str}:dec:0",
            object_structural_id=f"{task_id_str}:obj:0",
            object_kind="theory",
            role="prompt_context",
        )

    async def main():
        await asyncio.gather(
            task_worker(ledger_a, "task_a"),
            task_worker(ledger_b, "task_b"),
        )

    asyncio.run(main())

    recs_a = ledger_a.get_records()
    recs_b = ledger_b.get_records()

    assert len(recs_a) == 1
    assert len(recs_b) == 1
    assert recs_a[0]["decision_id"] == "task_a:dec:0"
    assert recs_b[0]["decision_id"] == "task_b:dec:0"


