"""
Behavioral & Structural Hypothesis Deduplication Engine for Phase 3b.

1. Structural AST Deduplication: Identifies identical AST syntax keys.
2. Behavioral Clustering: Groups hypotheses whose in-sample trigger-day sets have
   Jaccard similarity > 0.80 using connected components.
"""

from typing import Any, Dict, List, Set, Tuple
import numpy as np


def get_ast_canonical_key(hypothesis: Dict[str, Any]) -> str:
    """
    Computes a canonical structural key for a hypothesis JSON dict.
    """
    target = hypothesis.get("target_mode", "").lower()
    op = hypothesis.get("logical_op", "AND").upper()
    clauses = hypothesis.get("clauses", [])

    norm_clauses = []
    for c in clauses:
        feat = c.get("feature", "").lower()
        cop = c.get("operator", "").upper()
        q1 = round(float(c.get("threshold_q1", 0.0)), 4)
        q2 = round(float(c.get("threshold_q2", 0.0)), 4) if c.get("threshold_q2") is not None else None
        norm_clauses.append((feat, cop, q1, q2))

    norm_clauses.sort()
    clause_str = ";".join([f"{f}:{cop}:{q1}:{q2}" for f, cop, q1, q2 in norm_clauses])
    return f"target={target}|op={op}|clauses={clause_str}"


def compute_jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Computes Jaccard index J(A, B) = |A ∩ B| / |A ∪ B|.
    Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / float(union)


class UnionFind:

    def __init__(self, elements: List[str]):
        self.parent = {e: e for e in elements}

    def find(self, i: str) -> str:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def cluster_hypotheses_behaviorally(
    hypotheses: List[Dict[str, Any]],
    trigger_day_sets: Dict[str, Set[int]],
    jaccard_threshold: float = 0.80,
) -> Dict[str, Any]:
    """
    Deduplicates hypotheses structurally and clusters them behaviorally.

    Parameters:
    - hypotheses: List of hypothesis dictionaries.
    - trigger_day_sets: Map of hypothesis_id -> Set of in-sample trigger day indices.
    - jaccard_threshold: Jaccard overlap threshold for behavioral clustering (default 0.80).

    Returns dictionary containing cluster metadata.
    """
    # 1. Structural AST deduplication
    ast_keys: Dict[str, str] = {}  # ast_key -> first hypothesis_id
    struct_unique_hypotheses: List[Dict[str, Any]] = []
    struct_duplicates_count = 0

    for h in hypotheses:
        hid = h["hypothesis_id"]
        key = get_ast_canonical_key(h)
        if key in ast_keys:
            struct_duplicates_count += 1
        else:
            ast_keys[key] = hid
            struct_unique_hypotheses.append(h)

    unique_ids = [h["hypothesis_id"] for h in struct_unique_hypotheses]
    uf = UnionFind(unique_ids)

    # 2. Pairwise Jaccard behavioral overlap on in-sample trigger sets
    n = len(struct_unique_hypotheses)
    for i in range(n):
        id_a = unique_ids[i]
        set_a = trigger_day_sets.get(id_a, set())
        for j in range(i + 1, n):
            id_b = unique_ids[j]
            set_b = trigger_day_sets.get(id_b, set())
            jaccard = compute_jaccard_similarity(set_a, set_b)
            if jaccard > jaccard_threshold:
                uf.union(id_a, id_b)

    # 3. Aggregate clusters
    clusters_map: Dict[str, List[str]] = {}
    for hid in unique_ids:
        root = uf.find(hid)
        clusters_map.setdefault(root, []).append(hid)

    # Rename cluster IDs to cluster_001, cluster_002, etc.
    formatted_clusters: Dict[str, List[str]] = {}
    hypothesis_to_cluster: Dict[str, str] = {}
    for idx, (root, members) in enumerate(sorted(clusters_map.items(), key=lambda x: x[0]), 1):
        cid = f"cluster_{idx:03d}"
        formatted_clusters[cid] = members
        for m in members:
            hypothesis_to_cluster[m] = cid

    return {
        "clusters": formatted_clusters,
        "hypothesis_to_cluster": hypothesis_to_cluster,
        "total_input_hypotheses": len(hypotheses),
        "struct_unique_count": len(struct_unique_hypotheses),
        "struct_duplicates_count": struct_duplicates_count,
        "behavioral_clusters_count": len(formatted_clusters),
    }
