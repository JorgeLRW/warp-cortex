"""
Continuous Epistemic & Causal State Machine for Warp Cortex.

Represents research projects, causal hypotheses, and complex truth-seeking
as an epistemic graph with continuous semantic coordinates and typed constraints:
- Nodes represent Axioms, Hypotheses, Empirical Observations, or Questions.
- Edges represent typed constraints (logically_requires, evidence_depends_on, supports, refutes, blocks).
- Empirical results inject localized impulses that cascade through dependencies.
- Contradiction Energy measures positive simultaneous commitment to mutually incompatible claims.
- Directed counterfactual reachability impact discovers keystone hypotheses that govern project topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


class EpistemicKind(str, Enum):
    AXIOM = "axiom"                     # Ground truth or foundational premise
    HYPOTHESIS = "hypothesis"           # Active theoretical claim under test
    OBSERVATION_POS = "evidence_pos"    # Empirical result confirming a claim
    OBSERVATION_NEG = "evidence_neg"    # Empirical result refuting a claim
    QUESTION = "question"               # Unresolved open frontier


class EpistemicStatus(str, Enum):
    PROVEN = "proven"                   # Axiom or established mathematical/empirical ground truth
    CONFIRMED = "confirmed"             # High confidence, backed by valid evidence
    UNVERIFIED = "unverified"           # Active hypothesis awaiting testing
    UNSUPPORTED = "unsupported"         # Evidence collapsed; claim lacks justification (uncertain)
    FALSIFIED = "falsified"             # Direct empirical refutation or logical requirement failure


class EpistemicRelation(str, Enum):
    LOGICALLY_REQUIRES = "logically_requires"  # Deductive invariant: ¬Parent => ¬Child (falsification)
    EVIDENCE_DEPENDS_ON = "evidence_depends_on" # Evidential support: ¬Parent => Child becomes unsupported (C -> 0)
    DEPENDS_ON = "depends_on"                  # Alias for LOGICALLY_REQUIRES (backward compatibility)
    SUPPORTS = "supports"                      # Mutual positive reinforcement (+w)
    REFUTES = "refutes"                        # Contradictory tension (-w); generates contradiction energy
    BLOCKS = "blocks"                          # Inhibitory gate
    RELATED_TO = "related_to"                  # Ambient associative link


@dataclass
class EpistemicNode:
    """A single claim or observation node in the epistemic state machine."""
    node_id: str
    statement: str
    kind: EpistemicKind
    embedding: torch.Tensor                    # Continuous semantic coordinate on S^{D-1}
    confidence: float = 0.0                    # Belief potential: -1.0 (refuted) to +1.0 (proven)
    status: EpistemicStatus = EpistemicStatus.UNVERIFIED
    strain: float = 0.0                        # Contradiction energy localized to this node
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def is_falsified(self) -> bool:
        return self.status == EpistemicStatus.FALSIFIED or self.confidence <= -0.5

    def is_confirmed(self) -> bool:
        return self.status in (EpistemicStatus.CONFIRMED, EpistemicStatus.PROVEN) or self.confidence >= 0.7

    def is_unsupported(self) -> bool:
        return self.status == EpistemicStatus.UNSUPPORTED


@dataclass
class EpistemicEdge:
    """A directed or bidirectional constraint between two nodes."""
    source_id: str
    target_id: str
    relation: EpistemicRelation
    weight: float = 1.0
    edge_strain: float = 0.0                   # Accumulated empirical tension against this invariant
    is_active: bool = True                     # False if suspended or severed under persistent strain


class EpistemicManifold:
    """
    Typed Epistemic & Causal State Machine.
    
    Decouples:
    - Continuous semantic coordinates: what is relevant/salient on S^{D-1}.
    - Hard causal logic: what state transitions and conclusions are permissible.
    """

    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim
        self.nodes: Dict[str, EpistemicNode] = {}
        self.edges: List[EpistemicEdge] = []
        self._adjacency: Dict[str, Dict[str, EpistemicEdge]] = {}
        self._reverse_adjacency: Dict[str, Dict[str, EpistemicEdge]] = {}

    def register_claim(
        self,
        node_id: str,
        statement: str,
        kind: EpistemicKind = EpistemicKind.HYPOTHESIS,
        embedding: Optional[torch.Tensor] = None,
        confidence: float = 0.0,
        status: Optional[EpistemicStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EpistemicNode:
        """Register a theoretical claim, hypothesis, or axiom."""
        if embedding is None:
            seed = hash(statement) % (2**31 - 1)
            torch.manual_seed(seed)
            emb = F.normalize(torch.randn(self.hidden_dim), dim=0)
        else:
            emb = F.normalize(embedding.detach().float().reshape(-1), dim=0)

        conf = max(-1.0, min(1.0, float(confidence)))
        if status is None:
            if kind == EpistemicKind.AXIOM:
                status = EpistemicStatus.PROVEN
            elif conf >= 0.7:
                status = EpistemicStatus.CONFIRMED
            elif conf <= -0.5:
                status = EpistemicStatus.FALSIFIED
            else:
                status = EpistemicStatus.UNVERIFIED

        node = EpistemicNode(
            node_id=node_id,
            statement=statement,
            kind=kind,
            embedding=emb,
            confidence=conf,
            status=status,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        if node_id not in self._adjacency:
            self._adjacency[node_id] = {}
        if node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node_id] = {}
        return node

    def link_claims(
        self,
        source_id: str,
        target_id: str,
        relation: EpistemicRelation = EpistemicRelation.LOGICALLY_REQUIRES,
        weight: float = 1.0,
    ) -> None:
        """Add a causal, supportive, or contradictory constraint between claims."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise KeyError(f"Nodes {source_id} and {target_id} must both exist in manifold")

        edge = EpistemicEdge(source_id=source_id, target_id=target_id, relation=relation, weight=weight)
        self.edges.append(edge)
        self._adjacency[source_id][target_id] = edge
        self._reverse_adjacency[target_id][source_id] = edge

    def inject_observation(
        self,
        target_id: str,
        observation_text: str,
        confidence_delta: float,
        obs_id: Optional[str] = None,
        embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Inject an empirical observation or experimental result into a target claim.
        Cascades changes through dependencies and recalculates contradiction energy.
        """
        if target_id not in self.nodes:
            raise KeyError(f"Target node {target_id} not found in manifold")

        target_node = self.nodes[target_id]
        obs_id = obs_id or f"obs_{int(time.time()*1000)}"
        kind = EpistemicKind.OBSERVATION_POS if confidence_delta >= 0 else EpistemicKind.OBSERVATION_NEG

        obs_node = self.register_claim(
            node_id=obs_id,
            statement=observation_text,
            kind=kind,
            embedding=embedding or target_node.embedding,
            confidence=1.0,
            status=EpistemicStatus.PROVEN,
        )

        relation = EpistemicRelation.SUPPORTS if confidence_delta >= 0 else EpistemicRelation.REFUTES
        self.link_claims(source_id=obs_id, target_id=target_id, relation=relation, weight=abs(confidence_delta))

        # Update target node confidence and status
        old_conf = target_node.confidence
        target_node.confidence = max(-1.0, min(1.0, old_conf + confidence_delta))
        if target_node.confidence <= -0.5:
            target_node.status = EpistemicStatus.FALSIFIED
        elif target_node.confidence >= 0.7:
            target_node.status = EpistemicStatus.CONFIRMED
        else:
            target_node.status = EpistemicStatus.UNVERIFIED
        target_node.updated_at = time.time()

        # Propagate cascade down the dependency graph
        cascade_events = self._cascade_updates(target_id)

        # Recalculate contradiction energy
        strain_summary = self.calculate_contradiction_energy()

        return {
            "observation_id": obs_id,
            "target_id": target_id,
            "target_new_confidence": target_node.confidence,
            "target_new_status": target_node.status.value,
            "cascade_events": cascade_events,
            "strain_summary": strain_summary,
        }

    def _cascade_updates(self, start_id: str) -> List[Dict[str, Any]]:
        """
        Propagate causal state changes down the graph.
        
        Rules:
        1. LOGICALLY_REQUIRES / DEPENDS_ON:
           Strict deductive necessity: ¬Parent => ¬Child.
           If parent is falsified (conf <= -0.5), child is clamped to falsified (C_child <= C_parent).
        
        2. EVIDENCE_DEPENDS_ON:
           Justificatory dependency: if parent evidence is invalidated (conf <= -0.5),
           the child does NOT become falsified; its justification collapses back to neutral
           prior (C_child -> 0.0, status -> UNSUPPORTED), unless other valid evidence paths exist.
        """
        events = []
        visited = set()
        queue = [start_id]

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            curr_node = self.nodes[curr_id]

            # Find all nodes that depend on curr_node (curr_node is target of the dependency edge)
            for child_id, edge in self._reverse_adjacency.get(curr_id, {}).items():
                if not getattr(edge, "is_active", True):
                    continue
                child_node = self.nodes[child_id]
                old_conf = child_node.confidence
                old_status = child_node.status

                # 1. Deductive Dependency: LOGICALLY_REQUIRES or legacy DEPENDS_ON
                if edge.relation in (EpistemicRelation.LOGICALLY_REQUIRES, EpistemicRelation.DEPENDS_ON):
                    parent_bound = curr_node.confidence
                    if child_node.confidence > parent_bound:
                        child_node.confidence = max(-1.0, min(child_node.confidence, parent_bound))
                        if child_node.confidence <= -0.5:
                            child_node.status = EpistemicStatus.FALSIFIED
                        child_node.updated_at = time.time()
                        events.append({
                            "child_id": child_id,
                            "parent_id": curr_id,
                            "relation": edge.relation.value,
                            "old_confidence": old_conf,
                            "new_confidence": child_node.confidence,
                            "status": child_node.status.value,
                            "action": "deductive_falsification_clamped",
                        })
                        queue.append(child_id)

                # 2. Justificatory Dependency: EVIDENCE_DEPENDS_ON
                elif edge.relation == EpistemicRelation.EVIDENCE_DEPENDS_ON:
                    if curr_node.is_falsified():
                        # Check if child has ANY other valid supporting evidence
                        has_other_support = False
                        for other_parent_id, other_edge in self._adjacency.get(child_id, {}).items():
                            if other_parent_id != curr_id and other_edge.relation == EpistemicRelation.EVIDENCE_DEPENDS_ON:
                                if self.nodes[other_parent_id].is_confirmed():
                                    has_other_support = True
                                    break

                        if not has_other_support:
                            # Justification collapses: revert to neutral unsupported prior
                            child_node.confidence = 0.0
                            child_node.status = EpistemicStatus.UNSUPPORTED
                            child_node.updated_at = time.time()
                            events.append({
                                "child_id": child_id,
                                "parent_id": curr_id,
                                "relation": edge.relation.value,
                                "old_confidence": old_conf,
                                "new_confidence": 0.0,
                                "status": EpistemicStatus.UNSUPPORTED.value,
                                "action": "justification_invalidated_to_unsupported",
                            })
                            queue.append(child_id)

        return events

    def calculate_contradiction_energy(self) -> Dict[str, float]:
        """
        Calculate contradiction energy across the epistemic network.
        Contradiction occurs ONLY when two claims linked by REFUTES are simultaneously believed (> 0).
        If both are rejected/falsified (C_i <= 0, C_j <= 0), there is NO contradiction.
        
        E_contradiction = sum_{(i,j) in E_refutes} w_{ij} * max(0, C_i) * max(0, C_j)
        """
        total_energy = 0.0
        for node in self.nodes.values():
            node.strain = 0.0

        for edge in self.edges:
            if not getattr(edge, "is_active", True):
                continue
            if edge.relation == EpistemicRelation.REFUTES:
                n1 = self.nodes[edge.source_id]
                n2 = self.nodes[edge.target_id]
                # True contradiction energy: positive simultaneous commitment
                if n1.confidence > 0.0 and n2.confidence > 0.0:
                    tension = n1.confidence * n2.confidence * edge.weight
                    n1.strain += tension
                    n2.strain += tension
                    total_energy += tension

        return {"total_strain": total_energy, "contradiction_energy": total_energy}

    def check_topology_revisions(self, threshold: float = 2.5) -> List[Dict[str, Any]]:
        """
        Level 2 Topology Revision:
        Identifies structural invariant edges where persistent empirical strain exceeds threshold.
        Suspends the invariant so the edge itself becomes the falsified hypothesis,
        preventing ideological dogmatism when reality contradicts the prior DAG.
        """
        revisions = []
        for edge in self.edges:
            if getattr(edge, "is_active", True) and edge.edge_strain >= threshold:
                edge.is_active = False
                revisions.append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation.value,
                    "accumulated_strain": edge.edge_strain,
                    "action": "suspended_under_persistent_strain",
                })
        return revisions

    def _recalculate_strain(self) -> Dict[str, float]:
        """Backward compatibility alias for calculate_contradiction_energy."""
        return self.calculate_contradiction_energy()

    def find_keystone_hypotheses(self) -> List[str]:
        """
        Find keystone hypotheses using Directed Counterfactual Reachability Impact:
        
        Impact(v) = number of downstream dependent claims that lose their connection
        to foundational axioms or are invalidated when hypothesis v is removed.
        
        Replaces classic undirected Tarjan articulation with mathematically sound directed reachability.
        """
        hypothesis_nodes = [
            nid for nid, node in self.nodes.items()
            if node.kind == EpistemicKind.HYPOTHESIS
        ]
        if not hypothesis_nodes:
            return []

        def count_downstream_reach(root_id: str) -> int:
            visited = set()
            queue = [root_id]
            while queue:
                curr = queue.pop(0)
                for child_id, edge in self._reverse_adjacency.get(curr, {}).items():
                    if edge.relation in (
                        EpistemicRelation.LOGICALLY_REQUIRES,
                        EpistemicRelation.DEPENDS_ON,
                        EpistemicRelation.EVIDENCE_DEPENDS_ON,
                        EpistemicRelation.SUPPORTS,
                    ):
                        if child_id not in visited and child_id != root_id:
                            visited.add(child_id)
                            queue.append(child_id)
            return len(visited)

        # Rank hypotheses by downstream reach descending
        ranked = sorted(hypothesis_nodes, key=lambda nid: count_downstream_reach(nid), reverse=True)
        return [nid for nid in ranked if count_downstream_reach(nid) > 0]

    def get_active_frontier(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the highest-priority claims for autonomous research agents:
        Ranks by highest contradiction energy to resolve and highest uncertainty.
        """
        scored = []
        for nid, node in self.nodes.items():
            if node.kind in (EpistemicKind.HYPOTHESIS, EpistemicKind.QUESTION):
                uncertainty = 1.0 - abs(node.confidence)
                priority_score = node.strain * 2.0 + uncertainty
                scored.append({
                    "node_id": nid,
                    "statement": node.statement,
                    "kind": node.kind.value,
                    "confidence": node.confidence,
                    "status": node.status.value,
                    "strain": node.strain,
                    "priority_score": round(priority_score, 4),
                })

        scored.sort(key=lambda x: x["priority_score"], reverse=True)
        return scored[:top_k]

    def get_summary(self) -> Dict[str, Any]:
        """Overview of the research epistemic network health and topology."""
        confirmed = sum(1 for n in self.nodes.values() if n.is_confirmed())
        falsified = sum(1 for n in self.nodes.values() if n.is_falsified())
        unsupported = sum(1 for n in self.nodes.values() if n.is_unsupported())
        strain_summary = self.calculate_contradiction_energy()
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "confirmed_claims": confirmed,
            "falsified_claims": falsified,
            "unsupported_claims": unsupported,
            "keystones": self.find_keystone_hypotheses(),
            "contradiction_energy": round(strain_summary["total_strain"], 4),
            "active_frontier": self.get_active_frontier(3),
        }
