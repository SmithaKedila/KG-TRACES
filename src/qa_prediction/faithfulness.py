
from __future__ import annotations
from typing import Any, Dict, List, Set, Tuple

def _norm(x: Any) -> str:
    return " ".join(str(x).strip().lower().split())

def build_evidence_set(sample: Dict[str, Any]) -> Set[Tuple[str, str, str]]:
    """
    Build a set of evidence triples from sample['graph'].
    graph items are lists: [head, relation, tail]
    """
    ev: Set[Tuple[str, str, str]] = set()
    for item in sample.get("graph", []):
        if isinstance(item, list) and len(item) == 3:
            h, r, t = item
            ev.add((_norm(h), _norm(r), _norm(t)))
    return ev

def verify_triple_path(
    pred_triples: List[Tuple[str, str, str]],
    evidence: Set[Tuple[str, str, str]],
) -> Tuple[bool, List[Tuple[str, str, str]]]:
    """
    Returns (all_supported, unsupported_triples)
    """
    unsupported = []
    for h, r, t in pred_triples:
        if (_norm(h), _norm(r), _norm(t)) not in evidence:
            unsupported.append((h, r, t))
    return (len(unsupported) == 0), unsupported
