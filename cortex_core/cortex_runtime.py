"""
Warp Cortex Runtime: The Persistent Coordination Layer.
======================================================
Cortex is a persistent runtime substrate for state, relevance, and warranted change.
It sits underneath LLMs, databases, graphs, and multi-agent systems to make AI
event-driven rather than prompt-driven:

Core Runtime Operations:
1. observe(event) -> PropagationSummary:
   Injects an event into the persistent reaction field, diffusing potential across
   stationary continuous semantic aspects in < 1 millisecond. Supports idempotency
   and backpressure coalescing.
2. wake() -> List[AwakenedAgent]:
   Returns only the small subset of agents/tools that currently need expensive cognition (k << N).
3. commit(proposal) -> CommitResult:
   Validates proposed state changes and downstream actions using Optimistic Concurrency Control (OCC),
   verified provenance, and hard causal invariants before modifying the shared world state.
4. assemble_context(query) / assemble_innate_context(event) -> RetrievedContext:
   Hierarchical multi-aspect context assembly powered by the Semantic Context Fabric.
"""

from __future__ import annotations

import copy
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from cortex_core.reaction_harness import (
    ContinuousReactionManifold,
    ManifoldEntity,
    ManifoldImpulse,
)
from cortex_core.epistemic_manifold import (
    EpistemicManifold,
    EpistemicNode,
    EpistemicEdge,
    EpistemicRelation,
    EpistemicKind,
    EpistemicStatus,
)
from cortex_core.transition_governor import (
    EvidenceRegistry,
    EvidenceSourceTier,
    TransitionGovernor,
    TransitionCertificate,
    TransitionDecision,
    TransitionRule,
)
from cortex_core.semantic_fabric import (
    SemanticContextFabric,
    FabricItem,
    RetrievedContext,
    SemanticBand,
)


@dataclass
class RuntimeEvent:
    event_id: str
    text: str
    embedding: Optional[torch.Tensor]
    magnitude: float
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AwakenedAgent:
    agent_id: str
    name: str
    role: str
    energy: float
    activation_threshold: float
    trigger_aspect: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropagationSummary:
    event_id: str
    elapsed_ms: float
    awakened_count: int
    direct_hits_count: int
    highest_energy_entity: str
    highest_energy_value: float
    is_idempotent_skip: bool = False


@dataclass
class ProposedCommit:
    commit_id: str
    action_type: str                         # "STATE_UPDATE" or "ACTION_EXECUTION"
    target_node_id: str
    proposed_confidence_delta: float
    evidence_id: str
    rule: TransitionRule = TransitionRule.DIRECT_EMPIRICAL_UPDATE
    causal_path: List[Tuple[str, str, str]] = field(default_factory=list)
    proposing_agent_id: str = "system"
    payload: Dict[str, Any] = field(default_factory=dict)
    # Production Concurrency Semantics (OCC)
    base_version: int = 0                    # Substrate state version observed during agent reasoning
    read_set: List[str] = field(default_factory=list)  # Nodes read by the agent
    write_set: List[str] = field(default_factory=list) # Nodes targeted for state mutation


@dataclass
class CommitResult:
    admitted: bool
    commit_id: str
    target_node_id: str
    reason: str
    transition_cost: float
    blast_radius: int
    affected_dependents: List[str]
    violated_invariants: List[str]
    secondary_event_emitted: bool = False
    resulting_state_version: int = 0
    stale_detected: bool = False


class CortexRuntime:
    """
    The Persistent Coordination Runtime.
    
    Unifies:
    - Continuous Semantic Reaction Field (tracks dynamic relevance & waking).
    - Semantic Context Fabric (multi-aspect innate context & hierarchical retrieval).
    - Typed Epistemic & Causal Manifold (tracks truth, dependencies, and state).
    - Proof-Carrying Transition Governor (validates warrants & guards consistency).
    - Monotonic State Versioning & Optimistic Concurrency Control (guards distributed transactions).
    - Append-Only Event Log & Deterministic Replay (provides auditability, replay, and provenance).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        decay_rate: float = 0.20,
        diffusion_rate: float = 0.15,
        kernel_sigma: float = 0.35,
        topology_revision_threshold: float = 2.0,
    ):
        self.hidden_dim = hidden_dim
        
        # 1. Soft Layer: Continuous Reaction Field
        self.reaction_field = ContinuousReactionManifold(
            hidden_dim=hidden_dim,
            decay_rate=decay_rate,
            diffusion_rate=diffusion_rate,
            kernel_sigma=kernel_sigma,
        )

        # 2. Semantic Context Fabric: Frequency-like Multi-Aspect Context Space
        self.context_fabric = SemanticContextFabric(
            hidden_dim=hidden_dim,
        )

        # 3. Hard Layer: Typed Epistemic & Causal Graph
        self.epistemic_manifold = EpistemicManifold(hidden_dim=hidden_dim)

        # 4. Provenance & Evidence Registry
        self.evidence_registry = EvidenceRegistry()

        # 5. Gatekeeper: Transition Governor
        self.governor = TransitionGovernor(
            evidence_registry=self.evidence_registry,
            topology_revision_threshold=topology_revision_threshold,
        )

        # 6. Monotonic State Versioning & OCC Transaction Tracking
        self.state_version: int = 0
        self.node_last_modified_version: Dict[str, int] = {}

        # 7. Idempotency Tracking
        self.seen_event_ids: Set[str] = set()
        self.seen_commit_ids: Set[str] = set()

        # 8. Persistent Append-Only Logs
        self.event_log: List[RuntimeEvent] = []
        self.commit_log: List[Tuple[ProposedCommit, CommitResult]] = []

        # 9. Snapshots & Checkpoints
        self.checkpoints: Dict[str, Dict[str, Any]] = {}

        # 10. Observability & Causal Audit Lineage
        self.causal_audit_traces: Dict[str, Dict[str, Any]] = {}

        # Operational Metrics
        self.total_observe_calls = 0
        self.total_observe_time_ms = 0.0
        self.total_wake_events = 0
        self.total_commits_requested = 0
        self.total_commits_admitted = 0
        self.total_commits_blocked = 0
        self.total_stale_proposals_rejected = 0
        self.total_idempotent_skips = 0

    # -------------------------------------------------------------------------
    # Setup & Registration APIs
    # -------------------------------------------------------------------------

    def register_agent_entity(
        self,
        agent_id: str,
        name: str,
        role: str,
        prototypes: Dict[str, torch.Tensor],
        activation_threshold: float = 0.40,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent or system component in the continuous reaction substrate."""
        self.reaction_field.register_entity(
            entity_id=agent_id,
            name=name,
            role=role,
            prototypes=prototypes,
            activation_threshold=activation_threshold,
            state_metadata=metadata or {},
            rebuild_topology=True,
        )

    def register_claim(
        self,
        claim_id: str,
        statement: str,
        kind: EpistemicKind = EpistemicKind.HYPOTHESIS,
        confidence: float = 0.0,
    ) -> EpistemicNode:
        """Register a theoretical claim, proposition, or action goal in the causal state."""
        node = self.epistemic_manifold.register_claim(
            node_id=claim_id,
            statement=statement,
            kind=kind,
            confidence=confidence,
        )
        self.node_last_modified_version[claim_id] = self.state_version
        return node

    def link_causal_dependency(
        self,
        source_id: str,
        target_id: str,
        relation: EpistemicRelation = EpistemicRelation.LOGICALLY_REQUIRES,
    ) -> None:
        """Define a hard causal or supportive invariant between claims or actions."""
        self.epistemic_manifold.link_claims(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
        )

    def register_evidence(
        self,
        evidence_id: str,
        source_type: str,
        tier: EvidenceSourceTier = EvidenceSourceTier.LAB_ASSAY,
        description: str = "",
        reliability: Optional[float] = None,
    ) -> None:
        """Register verified empirical evidence in the trusted registry."""
        self.evidence_registry.register_evidence(
            evidence_id=evidence_id,
            source_type=source_type,
            tier=tier,
            description=description,
            custom_reliability=reliability,
        )

    def register_fabric_item(
        self,
        item_id: str,
        title: str,
        content: str,
        aspect_vectors: Dict[str, torch.Tensor],
        primary_aspect: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        causal_node_id: Optional[str] = None,
        validity_status: str = "VALID",
    ) -> FabricItem:
        """Register a knowledge item in the Semantic Context Fabric."""
        return self.context_fabric.register_item(
            item_id=item_id,
            title=title,
            content=content,
            aspect_vectors=aspect_vectors,
            primary_aspect=primary_aspect,
            metadata=metadata,
            causal_node_id=causal_node_id,
            validity_status=validity_status,
        )

    # -------------------------------------------------------------------------
    # Core Runtime Operation 1: observe(event)
    # -------------------------------------------------------------------------

    def observe(
        self,
        text: str,
        embedding: Optional[torch.Tensor] = None,
        magnitude: float = 1.0,
        source: str = "world",
        diffusion_steps: int = 2,
        event_id: Optional[str] = None,
    ) -> PropagationSummary:
        """
        Observe an external or internal event.
        Propagates cheap continuous reaction potential across multi-prototype aspects
        in sub-millisecond tensor operations.
        Guarantees idempotency via event deduplication.
        """
        t0 = time.perf_counter()

        ev_id = event_id or f"evt_{uuid.uuid4().hex[:8]}"

        # Idempotency check: duplicate event returns immediately without side-effects
        if ev_id in self.seen_event_ids:
            self.total_idempotent_skips += 1
            return PropagationSummary(
                event_id=ev_id,
                elapsed_ms=0.001,
                awakened_count=0,
                direct_hits_count=0,
                highest_energy_entity="",
                highest_energy_value=0.0,
                is_idempotent_skip=True,
            )

        self.seen_event_ids.add(ev_id)

        if embedding is None:
            seed = hash(text) % (2**31 - 1)
            torch.manual_seed(seed)
            emb = F.normalize(torch.randn(self.hidden_dim), dim=0)
        else:
            emb = F.normalize(embedding.detach().float().reshape(-1), dim=0)

        runtime_event = RuntimeEvent(
            event_id=ev_id,
            text=text,
            embedding=emb,
            magnitude=magnitude,
            source=source,
            timestamp=time.time(),
        )
        self.event_log.append(runtime_event)

        # 1. Inject impulse via radial Gaussian kernel on S^{D-1}
        direct_hits = self.reaction_field.inject_impulse(
            text=text,
            embedding=emb,
            magnitude=magnitude,
            source=source,
            event_id=ev_id,
        )

        # 2. Diffuse energy across coupled topological aspects
        triggered = self.reaction_field.step_diffusion(steps=diffusion_steps)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.total_observe_calls += 1
        self.total_observe_time_ms += elapsed_ms

        # Summary telemetry
        if self.reaction_field.entities:
            highest_id = max(self.reaction_field.entities, key=lambda k: self.reaction_field.entities[k].current_energy)
            highest_val = self.reaction_field.entities[highest_id].current_energy
            awakened_cnt = sum(1 for e in self.reaction_field.entities.values() if e.is_triggered())
        else:
            highest_id = ""
            highest_val = 0.0
            awakened_cnt = 0

        return PropagationSummary(
            event_id=ev_id,
            elapsed_ms=elapsed_ms,
            awakened_count=awakened_cnt,
            direct_hits_count=len(direct_hits),
            highest_energy_entity=highest_id,
            highest_energy_value=highest_val,
            is_idempotent_skip=False,
        )

    # -------------------------------------------------------------------------
    # Core Runtime Operation 2: wake()
    # -------------------------------------------------------------------------

    def wake(self, auto_cool: bool = True, cool_factor: float = 0.2) -> List[AwakenedAgent]:
        """
        Return only the small subset of agents/tools that currently crossed their
        activation threshold (k << N).
        Cools down potential after waking to avoid infinite busy-looping.
        """
        awakened: List[AwakenedAgent] = []
        for eid, ent in self.reaction_field.entities.items():
            if ent.is_triggered():
                best_aspect = "core"
                if ent.prototypes:
                    best_aspect = next(iter(ent.prototypes.keys()))

                awakened.append(AwakenedAgent(
                    agent_id=eid,
                    name=ent.name,
                    role=ent.role,
                    energy=ent.current_energy,
                    activation_threshold=ent.activation_threshold,
                    trigger_aspect=best_aspect,
                    metadata=ent.state_metadata,
                ))

                if auto_cool:
                    self.reaction_field.cool_down_entity(eid, factor=cool_factor)

        self.total_wake_events += len(awakened)
        return awakened

    # -------------------------------------------------------------------------
    # Core Runtime Operation 3: commit(proposal) with OCC Transaction Semantics
    # -------------------------------------------------------------------------

    def commit(self, proposal: ProposedCommit) -> CommitResult:
        """
        Validate and apply a proposed state modification or downstream action.
        
        Optimistic Concurrency Control (OCC) Semantics:
        CommitAllowed = [version_valid] and [warrant_valid] and [no_conflicting_update]
        
        If an agent reads state at base_version V_0, but concurrent commits mutate
        nodes in read_set before this proposal reaches commit(), the proposal is
        rejected with STALE_PROPOSAL_REVALIDATE.
        """
        self.total_commits_requested += 1

        # Idempotency check: duplicated commit proposals are rejected safely
        if proposal.commit_id in self.seen_commit_ids:
            self.total_idempotent_skips += 1
            res = CommitResult(
                admitted=False,
                commit_id=proposal.commit_id,
                target_node_id=proposal.target_node_id,
                reason="DUPLICATE_COMMIT_IDEMPOTENT: Proposal has already been processed.",
                transition_cost=0.0,
                blast_radius=0,
                affected_dependents=[],
                violated_invariants=["IDEMPOTENT_DUPLICATE"],
                secondary_event_emitted=False,
                resulting_state_version=self.state_version,
                stale_detected=False,
            )
            return res

        # OCC Conflict & Staleness Detection:
        # Check if any node in read_set or target_node_id was updated after base_version
        conflicted_nodes = []
        nodes_to_check = set(proposal.read_set)
        nodes_to_check.add(proposal.target_node_id)
        
        for n_id in nodes_to_check:
            last_mod = self.node_last_modified_version.get(n_id, 0)
            if last_mod > proposal.base_version:
                conflicted_nodes.append((n_id, last_mod))

        if conflicted_nodes and proposal.base_version < self.state_version:
            self.total_commits_blocked += 1
            self.total_stale_proposals_rejected += 1
            res = CommitResult(
                admitted=False,
                commit_id=proposal.commit_id,
                target_node_id=proposal.target_node_id,
                reason=(
                    f"STALE_PROPOSAL_REVALIDATE: base_version {proposal.base_version} < current_version {self.state_version}. "
                    f"Conflicted read_set modifications: {conflicted_nodes}."
                ),
                transition_cost=0.0,
                blast_radius=0,
                affected_dependents=[],
                violated_invariants=["OCC_VERSION_CONFLICT"],
                secondary_event_emitted=False,
                resulting_state_version=self.state_version,
                stale_detected=True,
            )
            self.commit_log.append((proposal, res))
            return res

        # Governor Warrant & Invariant Verification
        certificate = TransitionCertificate(
            evidence_id=proposal.evidence_id,
            target_node_id=proposal.target_node_id,
            proposed_confidence_delta=proposal.proposed_confidence_delta,
            rule=proposal.rule,
            causal_path=proposal.causal_path,
            rationale=f"Proposed by {proposal.proposing_agent_id}",
        )

        admitted, decision = self.governor.commit_if_admitted(
            manifold=self.epistemic_manifold,
            certificate=certificate,
        )

        secondary_emitted = False
        if admitted:
            self.total_commits_admitted += 1
            # Advance monotonic state version
            self.state_version += 1
            self.node_last_modified_version[proposal.target_node_id] = self.state_version
            for w in proposal.write_set:
                self.node_last_modified_version[w] = self.state_version
            
            self.seen_commit_ids.add(proposal.commit_id)

            # Record Observability Audit Trace
            self.causal_audit_traces[proposal.commit_id] = {
                "commit_id": proposal.commit_id,
                "state_version": self.state_version,
                "agent_id": proposal.proposing_agent_id,
                "target": proposal.target_node_id,
                "evidence_id": proposal.evidence_id,
                "delta": proposal.proposed_confidence_delta,
                "read_set": proposal.read_set,
                "write_set": proposal.write_set,
                "timestamp": time.time(),
            }

            # Propagate secondary perturbation into continuous field
            target_node = self.epistemic_manifold.nodes.get(proposal.target_node_id)
            if target_node:
                self.reaction_field.inject_impulse(
                    text=f"Committed change to {proposal.target_node_id}: delta={proposal.proposed_confidence_delta:.2f}",
                    embedding=target_node.embedding,
                    magnitude=0.60,
                    source=f"commit:{proposal.target_node_id}",
                )
                # Update Context Fabric item status if linked
                for fb in self.context_fabric.items.values():
                    if fb.causal_node_id == proposal.target_node_id:
                        if target_node.status in (EpistemicStatus.FALSIFIED, EpistemicStatus.UNSUPPORTED):
                            self.context_fabric.update_dynamic_state(fb.item_id, energy_delta=0.8, validity_status="TAINTED")
                        else:
                            self.context_fabric.update_dynamic_state(fb.item_id, energy_delta=0.2)
                secondary_emitted = True
        else:
            self.total_commits_blocked += 1

        res = CommitResult(
            admitted=admitted,
            commit_id=proposal.commit_id,
            target_node_id=proposal.target_node_id,
            reason=decision.reason,
            transition_cost=decision.transition_cost,
            blast_radius=decision.blast_radius,
            affected_dependents=decision.affected_dependents,
            violated_invariants=decision.violated_invariants,
            secondary_event_emitted=secondary_emitted,
            resulting_state_version=self.state_version,
            stale_detected=False,
        )

        self.commit_log.append((proposal, res))
        return res

    # -------------------------------------------------------------------------
    # Context Assembly & Innate Context APIs
    # -------------------------------------------------------------------------

    def query_context(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor] = None,
        target_aspects: Optional[List[str]] = None,
        token_budget: int = 1024,
        state_weight: float = 0.40,
        include_structural_neighbors: bool = True,
    ) -> RetrievedContext:
        """Query context hierarchically through the Semantic Context Fabric."""
        return self.context_fabric.assemble_context(
            query=query,
            query_embedding=query_embedding,
            target_aspects=target_aspects,
            token_budget=token_budget,
            state_weight=state_weight,
            epistemic_manifold=self.epistemic_manifold,
            include_structural_neighbors=include_structural_neighbors,
        )

    def get_innate_context(
        self,
        trigger_entity_id: str,
        token_budget: int = 1024,
    ) -> RetrievedContext:
        """Assemble innate unprompted context directly from entity coordinates and causal graph."""
        return self.context_fabric.assemble_innate_context(
            trigger_entity_id=trigger_entity_id,
            epistemic_manifold=self.epistemic_manifold,
            token_budget=token_budget,
        )

    # -------------------------------------------------------------------------
    # Checkpointing, Snapshots & Deterministic Replay
    # -------------------------------------------------------------------------

    def create_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Create an immutable snapshot of the current substrate state."""
        snapshot = {
            "checkpoint_id": checkpoint_id,
            "state_version": self.state_version,
            "nodes": {
                k: {"confidence": v.confidence, "status": v.status.value, "kind": v.kind.value}
                for k, v in self.epistemic_manifold.nodes.items()
            },
            "node_versions": dict(self.node_last_modified_version),
            "event_log_length": len(self.event_log),
            "commit_log_length": len(self.commit_log),
            "timestamp": time.time(),
        }
        self.checkpoints[checkpoint_id] = snapshot
        return snapshot

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore substrate state from a previously saved checkpoint."""
        if checkpoint_id not in self.checkpoints:
            return False
        snap = self.checkpoints[checkpoint_id]
        self.state_version = snap["state_version"]
        self.node_last_modified_version = dict(snap["node_versions"])
        for nid, data in snap["nodes"].items():
            if nid in self.epistemic_manifold.nodes:
                node = self.epistemic_manifold.nodes[nid]
                node.confidence = data["confidence"]
                node.status = EpistemicStatus(data["status"])
        return True

    def replay_from_log(
        self,
        events: List[RuntimeEvent],
        commits: Optional[List[ProposedCommit]] = None,
    ) -> int:
        """
        Deterministically reconstruct current state by replaying an append-only event & commit log.
        Guarantees that replaying E_{1:t} produces identical S_t.
        """
        replayed_count = 0
        for ev in events:
            self.observe(
                text=ev.text,
                embedding=ev.embedding,
                magnitude=ev.magnitude,
                source=ev.source,
                event_id=ev.event_id,
            )
            replayed_count += 1
            
        if commits:
            for prop in commits:
                self.commit(prop)
                replayed_count += 1
                
        return replayed_count

    def get_audit_trace(self, commit_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve causal audit and lineage information for a specific commit."""
        return self.causal_audit_traces.get(commit_id)

    # -------------------------------------------------------------------------
    # State Inspection & Telemetry
    # -------------------------------------------------------------------------

    def get_substrate_state(self) -> Dict[str, Any]:
        """Return the current consistent state snapshot."""
        claims_state = {
            cid: {
                "statement": node.statement,
                "confidence": round(node.confidence, 4),
                "status": node.status.value,
                "kind": node.kind.value,
                "version": self.node_last_modified_version.get(cid, 0),
            }
            for cid, node in self.epistemic_manifold.nodes.items()
        }
        active_invariants = [
            f"({e.source_id} -[{e.relation.value}]-> {e.target_id})"
            for e in self.epistemic_manifold.edges if getattr(e, "is_active", True)
        ]
        dormant_count = sum(1 for e in self.reaction_field.entities.values() if not e.is_triggered())
        total_entities = len(self.reaction_field.entities)
        dormant_pct = (dormant_count / max(1, total_entities)) * 100.0

        avg_obs_ms = self.total_observe_time_ms / max(1, self.total_observe_calls)

        return {
            "state_version": self.state_version,
            "entity_count": total_entities,
            "dormant_percentage": round(dormant_pct, 1),
            "claims_count": len(claims_state),
            "active_invariants_count": len(active_invariants),
            "claims": claims_state,
            "active_invariants": active_invariants,
            "total_events_observed": self.total_observe_calls,
            "average_observe_latency_ms": round(avg_obs_ms, 3),
            "total_commits_requested": self.total_commits_requested,
            "total_commits_admitted": self.total_commits_admitted,
            "total_commits_blocked": self.total_commits_blocked,
            "total_stale_proposals_rejected": self.total_stale_proposals_rejected,
            "total_idempotent_skips": self.total_idempotent_skips,
        }
