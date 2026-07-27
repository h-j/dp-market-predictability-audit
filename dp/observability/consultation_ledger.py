"""
Consultation Ledger — Read-Side Provenance Substrate.

Boundary Rule:
A read is a CONSULTATION if it informed a decision's output or gated control flow;
reads for logging/metrics display are INSPECTIONS and are not recorded.

Records every cognitive read (prior theories, regime memory, lessons, principles, confidence states)
and decision output in an append-only, byte-stable JSONL ledger without wall-clock fields.
"""
import contextvars
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union


class ConsultationLedger:
    """
    Append-only consultation ledger for tracking read-side provenance.

    Produces byte-identical JSONL output across identical replay runs.
    Does NOT contain wall-clock fields (e.g. timestamp/created_at).
    """

    def __init__(
        self,
        valid_object_kinds: Set[str],
        valid_roles: Set[str],
        output_path: Optional[Union[str, Path]] = None,
    ):
        self.output_path = Path(output_path) if output_path else None
        self.valid_object_kinds = set(valid_object_kinds)
        self.valid_roles = set(valid_roles)
        self._decision_seq: Dict[str, int] = {}
        self._records: List[Dict] = []
        if self.output_path and self.output_path.exists():
            self.output_path.unlink()

    def set_output_path(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        if self.output_path.exists():
            self.output_path.unlink()
        # Flush existing in-memory records if any were recorded before path set
        for rec in self._records:
            self._write_record(rec)

    def _write_record(self, record: Dict):
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def record_consultation(
        self,
        decision_id: str,
        object_structural_id: str,
        object_kind: str,
        role: str,
        provenance_method: str = "observed",
    ) -> Dict:
        """
        Record a read consultation that informed a decision or gated control flow.
        """
        if self.valid_object_kinds and object_kind not in self.valid_object_kinds:
            raise ValueError(
                f"Invalid object_kind '{object_kind}'. Must be one of {self.valid_object_kinds}"
            )
        if self.valid_roles and role not in self.valid_roles:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of {self.valid_roles}"
            )

        seq = self._decision_seq.get(decision_id, 0) + 1
        self._decision_seq[decision_id] = seq

        record = {
            "kind": "consultation",
            "decision_id": decision_id,
            "object_structural_id": object_structural_id,
            "object_kind": object_kind,
            "role": role,
            "provenance_method": provenance_method,
            "seq": seq,
        }
        self._records.append(record)
        self._write_record(record)
        return record

    def record_decision(
        self,
        decision_id: str,
        output_content: str,
        day: int,
    ) -> Dict:
        """
        Record a decision output and its deterministic SHA-256 output hash.
        """
        output_hash = hashlib.sha256(
            (output_content or "").encode("utf-8")
        ).hexdigest()

        record = {
            "kind": "decision",
            "decision_id": decision_id,
            "output_hash": output_hash,
            "day": day,
        }
        self._records.append(record)
        self._write_record(record)
        return record

    def get_records(self) -> List[Dict]:
        return list(self._records)


_ledger_ctx: contextvars.ContextVar[Optional[ConsultationLedger]] = (
    contextvars.ContextVar("consultation_ledger", default=None)
)


def get_active_consultation_ledger() -> Optional[ConsultationLedger]:
    return _ledger_ctx.get()


def set_active_consultation_ledger(ledger: Optional[ConsultationLedger]):
    _ledger_ctx.set(ledger)


def record_consultation(
    decision_id: str,
    object_structural_id: str,
    object_kind: str,
    role: str,
    provenance_method: str = "observed",
) -> Optional[Dict]:
    ledger = get_active_consultation_ledger()
    if ledger:
        return ledger.record_consultation(
            decision_id,
            object_structural_id,
            object_kind,
            role,
            provenance_method=provenance_method,
        )
    return None


def record_decision(
    decision_id: str,
    output_content: str,
    day: int,
) -> Optional[Dict]:
    ledger = get_active_consultation_ledger()
    if ledger:
        return ledger.record_decision(decision_id, output_content, day)
    return None
