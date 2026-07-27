"""
DP Domain Taxonomy — Object Kinds and Roles.

Defines DP's domain-specific vocabulary for read-side consultation tracking.
"""
from typing import Set

DP_OBJECT_KINDS: Set[str] = {
    "theory",
    "lesson",
    "principle",
    "regime_memory",
    "confidence_state",
}

DP_ROLES: Set[str] = {
    "prompt_context",
    "gate",
    "prior",
}
