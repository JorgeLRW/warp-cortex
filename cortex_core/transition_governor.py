"""
Proof-Carrying State Transition Governor for Warp Cortex.

Formalizes machine-verifiable transition certificates Pi = (S_t, Delta S, E, P, R):
    Admit(Delta S) = 1[ InvariantOK and PathExists and EvidenceSufficient ]

Key Conceptual Separations:
1. Truth Gate != Blast-Radius Gate:
   The epistemic admissibility of an update depends solely on evidence quality and logical invariants:
       TransitionCost = |Delta C| / (EvidenceReliability + epsilon) <= max_cost_threshold
   Central hypotheses are NOT penalized for being central. A decisive experiment (u_e >= 0.90)
   can overturn a foundational assumption immediately.
2. Structural Blast Radius:
   Once an update is admitted, the governor independently calculates the consequence blast radius:
       BlastRadius = |Downstream Dependents Governed by Claim|
   and marks dependent conclusions for formal revalidation / invalidation cascade.
3. Formal Certificate Pi = (S_t, Delta S, E, P, R):
   - E: Authenticated evidence reference with parameterized reliability.
   - P: Verified typed edge sequence through trusted substrate.
   - R: Transition rule (EMPIRICAL_UPDATE, DEDUCTIVE_FALSIFICATION, EVIDENTIAL_COLLAPSE).
   - Verifier V(S_t, Pi) in {0, 1}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from cortex_core.epistemic_manifold import (
    EpistemicManifold,
    EpistemicNode,
    EpistemicRelation,
    EpistemicStatus,
    EpistemicKind,
)


class EvidenceSourceTier(str, Enum):
    LAB_ASSAY = "lab_assay"                         # Direct empirical measurement / assay
    REPLICATED_STUDY = "replicated_study"           # Multi-center / independent replication
    SYSTEM_DEDUCTION = "system_deduction"           # Deterministic inference from prior axioms
    UNVERIFIED_CLAIM = "unverified_claim"           # Unverified user prompt or external rumor


class TransitionRule(str, Enum):
    DIRECT_EMPIRICAL_UPDATE = "direct_empirical_update"
    DEDUCTIVE_INVARIANT_CLAMP = "deductive_invariant_clamp"
    EVIDENTIAL_JUSTIFICATION_REVERSION = "evidential_justification_reversion"


@dataclass
class EvidenceRecord:
    """An independently verified evidence record with parameterized reliability."""
    evidence_id: str
    source_type: str
    tier: EvidenceSourceTier
    reliability: float                        # Parameterized score in [0.0, 1.0]
    description: str
    sample_size: int = 1
    measurement_uncertainty: float = 0.05     # Standard error / noise
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceRegistry:
    """External trusted evidence store; decouples evidence strength from proposing models."""

    def __init__(self):
        self._records: Dict[str, EvidenceRecord] = {}

    def register_evidence(
        self,
        evidence_id: str,
        tier: EvidenceSourceTier,
        source_type: str,
        description: str,
        sample_size: int = 1,
        measurement_uncertainty: float = 0.05,
        custom_reliability: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        if custom_reliability is not None:
            rel = custom_reliability
        else:
            # Parameterized reliability function: base prior modulated by uncertainty & sample size
            base_prior = {
                EvidenceSourceTier.LAB_ASSAY: 0.95,
                EvidenceSourceTier.REPLICATED_STUDY: 0.90,
                EvidenceSourceTier.SYSTEM_DEDUCTION: 0.50,
                EvidenceSourceTier.UNVERIFIED_CLAIM: 0.15,
            }[tier]
            # Uncertainty penalty and sample size bonus
            rel = base_prior * (1.0 - min(0.5, measurement_uncertainty)) * min(1.1, 1.0 + 0.02 * math.log(max(1, sample_size)))
            rel = max(0.05, min(0.99, rel))

        record = EvidenceRecord(
            evidence_id=evidence_id,
            source_type=source_type,
            tier=tier,
            reliability=round(rel, 4),
            description=description,
            sample_size=sample_size,
            measurement_uncertainty=measurement_uncertainty,
            metadata=metadata or {},
        )
        self._records[evidence_id] = record
        return record

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._records.get(evidence_id)


@dataclass
class TransitionCertificate:
    """A formal machine-verifiable transition certificate Pi = (S_t, Delta S, E, P, R)."""
    evidence_id: str                           # E: Reference in EvidenceRegistry
    target_node_id: str                        # Target claim being updated
    proposed_confidence_delta: float           # Delta C
    rule: TransitionRule = TransitionRule.DIRECT_EMPIRICAL_UPDATE # R: Transition rule
    causal_path: List[Tuple[str, str, str]] = field(default_factory=list) # P: Typed edge path
    semantic_path: List[str] = field(default_factory=list)
    rationale: str = ""


# Alias for backward compatibility
TransitionWitness = TransitionCertificate


@dataclass
class TransitionDecision:
    """The binary gate outcome with separated truth admissibility and blast radius."""
    admitted: bool
    reason: str
    transition_cost: float
    evidence_reliability: float
    blast_radius: int                          # Number of downstream conclusions affected
    affected_dependents: List[str] = field(default_factory=list)
    violated_invariants: List[str] = field(default_factory=list)


class TransitionGovernor:
    """
    Deterministic Governor implementing machine-verifiable transition validation.
    Separates the Epistemic Truth Gate from the Structural Blast-Radius Gate.
    """

    def __init__(
        self,
        evidence_registry: Optional[EvidenceRegistry] = None,
        max_cost_threshold: float = 4.0,
        epsilon: float = 0.05,
        topology_revision_threshold: float = 2.0,
    ):
        self.registry = evidence_registry or EvidenceRegistry()
        self.max_cost_threshold = max_cost_threshold
        self.epsilon = epsilon
        self.topology_revision_threshold = topology_revision_threshold
        self.decision_history: List[Dict[str, Any]] = []

    def evaluate_transition(
        self,
        manifold: EpistemicManifold,
        certificate: TransitionCertificate,
    ) -> TransitionDecision:
        """
        Machine verification: V(S_t, Pi) in {0, 1}.
        Admit(Delta S) = 1[ InvariantOK and PathExists and EvidenceSufficient ].
        """
        target_id = certificate.target_node_id
        if target_id not in manifold.nodes:
            return TransitionDecision(
                admitted=False,
                reason=f"Target claim '{target_id}' does not exist in epistemic state.",
                transition_cost=float("inf"),
                evidence_reliability=0.0,
                blast_radius=0,
            )

        # -------------------------------------------------------------------------------
        # 1. Independent Evidence Provenance Verification
        # -------------------------------------------------------------------------------
        evidence_rec = self.registry.get(certificate.evidence_id)
        if evidence_rec is None:
            return TransitionDecision(
                admitted=False,
                reason=(
                    f"Provenance Failure: Evidence ID '{certificate.evidence_id}' is not registered in the "
                    "trusted EvidenceRegistry. Self-declared or fabricated evidence cannot justify state changes."
                ),
                transition_cost=float("inf"),
                evidence_reliability=0.0,
                blast_radius=0,
            )

        evidence_strength = evidence_rec.reliability
        target_node = manifold.nodes[target_id]
        new_conf = max(-1.0, min(1.0, target_node.confidence + certificate.proposed_confidence_delta))

        # -------------------------------------------------------------------------------
        # 2. Hard Invariant Check (Admissible Region Omega)
        # -------------------------------------------------------------------------------
        violated_invariants = []

        # Deductive prerequisite invariant
        for parent_id, edge in manifold._adjacency.get(target_id, {}).items():
            if not getattr(edge, "is_active", True):
                continue
            if edge.relation in (EpistemicRelation.LOGICALLY_REQUIRES, EpistemicRelation.DEPENDS_ON):
                parent_node = manifold.nodes.get(parent_id)
                if parent_node and new_conf > parent_node.confidence:
                    # Level 2 Strain Accumulation: empirical evidence on hypotheses creates pressure against the invariant
                    if certificate.rule == TransitionRule.DIRECT_EMPIRICAL_UPDATE and target_node.kind == EpistemicKind.HYPOTHESIS:
                        edge.edge_strain += evidence_strength * max(0.5, new_conf - parent_node.confidence)
                        if edge.edge_strain >= self.topology_revision_threshold:
                            # Level 2 Topology Revision: persistent empirical strain suspends the false invariant!
                            edge.is_active = False
                    
                    if getattr(edge, "is_active", True):
                        violated_invariants.append(
                            f"Invariant Violation: '{target_id}' LOGICALLY_REQUIRES parent '{parent_id}' "
                            f"(parent conf: {parent_node.confidence:.2f}); child cannot be increased to {new_conf:.2f}. "
                            f"[Edge Strain: {edge.edge_strain:.2f}/{self.topology_revision_threshold:.2f}]"
                        )

        # Mutual contradiction invariant
        for other_id, edge in manifold._adjacency.get(target_id, {}).items():
            if edge.relation == EpistemicRelation.REFUTES:
                other_node = manifold.nodes.get(other_id)
                if other_node and other_node.confidence > 0.70 and new_conf > 0.70:
                    violated_invariants.append(
                        f"Invariant Violation: Simultaneous high belief in mutually refuting claims "
                        f"'{target_id}' (conf {new_conf:.2f}) and '{other_id}' (conf {other_node.confidence:.2f})."
                    )

        if violated_invariants:
            decision = TransitionDecision(
                admitted=False,
                reason="Hard epistemic invariant violated in Omega.",
                transition_cost=float("inf"),
                evidence_reliability=evidence_strength,
                blast_radius=0,
                violated_invariants=violated_invariants,
            )
            self._record(certificate, decision)
            return decision

        # -------------------------------------------------------------------------------
        # 3. Verified Causal Path Continuity Check
        # -------------------------------------------------------------------------------
        if certificate.causal_path:
            for src, tgt, rel in certificate.causal_path:
                edge = manifold._adjacency.get(src, {}).get(tgt)
                if not edge or edge.relation.value != rel:
                    decision = TransitionDecision(
                        admitted=False,
                        reason=f"Path Continuity Violation: Edge ({src} -[{rel}]-> {tgt}) does not exist in verified substrate.",
                        transition_cost=float("inf"),
                        evidence_reliability=evidence_strength,
                        blast_radius=0,
                    )
                    self._record(certificate, decision)
                    return decision

        # -------------------------------------------------------------------------------
        # 4. Epistemic Truth Gate (NO Keystones Penalty)
        # -------------------------------------------------------------------------------
        # Admissibility depends purely on whether evidence quality justifies the confidence displacement:
        # TransitionCost = |Delta C| / (EvidenceStrength + epsilon)
        delta_c = abs(certificate.proposed_confidence_delta)
        transition_cost = delta_c / (evidence_strength + self.epsilon)

        if transition_cost > self.max_cost_threshold:
            decision = TransitionDecision(
                admitted=False,
                reason=(
                    f"Justificatory Consistency Failure: Confidence displacement ({delta_c:.2f}) "
                    f"is disproportionate to evidence reliability ({evidence_strength:.2f}). "
                    f"TransitionCost {transition_cost:.2f} > {self.max_cost_threshold:.2f}."
                ),
                transition_cost=round(transition_cost, 4),
                evidence_reliability=evidence_strength,
                blast_radius=0,
            )
            self._record(certificate, decision)
            return decision

        # -------------------------------------------------------------------------------
        # 5. Blast-Radius Gate (Consequence Quantification)
        # -------------------------------------------------------------------------------
        # Truth gate passed! Now compute consequence blast radius for downstream invalidation
        downstream_dependents = self._get_downstream_dependents(manifold, target_id)
        blast_radius = len(downstream_dependents)

        decision = TransitionDecision(
            admitted=True,
            reason=(
                f"Warrant verified. Admitted with blast radius of {blast_radius} dependent conclusions."
                if blast_radius > 0 else "Warrant verified. Leaf claim admitted."
            ),
            transition_cost=round(transition_cost, 4),
            evidence_reliability=evidence_strength,
            blast_radius=blast_radius,
            affected_dependents=downstream_dependents,
        )
        self._record(certificate, decision)
        return decision

    def commit_if_admitted(
        self,
        manifold: EpistemicManifold,
        certificate: TransitionCertificate,
    ) -> Tuple[bool, TransitionDecision]:
        """Evaluates certificate and commits the update to the persistent manifold if admitted."""
        decision = self.evaluate_transition(manifold, certificate)
        if decision.admitted:
            target_node = manifold.nodes[certificate.target_node_id]
            target_node.confidence = max(-1.0, min(1.0, target_node.confidence + certificate.proposed_confidence_delta))
            if target_node.confidence <= -0.5:
                target_node.status = EpistemicStatus.FALSIFIED
            elif target_node.confidence >= 0.7:
                target_node.status = EpistemicStatus.CONFIRMED
            else:
                target_node.status = EpistemicStatus.UNVERIFIED

            # If node was falsified, trigger cascading invalidations across its blast radius
            if target_node.is_falsified():
                manifold._cascade_updates(certificate.target_node_id)

            manifold.calculate_contradiction_energy()

        return decision.admitted, decision

    def _get_downstream_dependents(self, manifold: EpistemicManifold, root_id: str) -> List[str]:
        visited = set()
        queue = [root_id]
        while queue:
            curr = queue.pop(0)
            for child_id, edge in manifold._reverse_adjacency.get(curr, {}).items():
                if edge.relation in (
                    EpistemicRelation.LOGICALLY_REQUIRES,
                    EpistemicRelation.DEPENDS_ON,
                    EpistemicRelation.EVIDENCE_DEPENDS_ON,
                    EpistemicRelation.SUPPORTS,
                ):
                    if child_id not in visited and child_id != root_id:
                        visited.add(child_id)
                        queue.append(child_id)
        return list(visited)

    def _record(self, cert: TransitionCertificate, decision: TransitionDecision) -> None:
        self.decision_history.append({
            "target_node": cert.target_node_id,
            "delta": cert.proposed_confidence_delta,
            "evidence_id": cert.evidence_id,
            "reliability": decision.evidence_reliability,
            "blast_radius": decision.blast_radius,
            "cost": decision.transition_cost,
            "admitted": decision.admitted,
            "reason": decision.reason,
            "invariants": decision.violated_invariants,
        })
