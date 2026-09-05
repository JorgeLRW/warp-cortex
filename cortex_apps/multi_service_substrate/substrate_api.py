"""
Substrate API & Data Definitions.
=================================
Formal specifications for the Context Substrate interface, snapshots, events,
actions, verification results, and operational instrumentation metrics.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from cortex_apps.research_agent_system.world_state import ResearchDocument, ResearchWorldCatalog


class EntityStatus(str, Enum):
    NOMINAL = "NOMINAL"
    TAINTED = "TAINTED"
    RECALIBRATING = "RECALIBRATING"
    OFFLINE = "OFFLINE"
    VALID = "VALID"
    INVALID = "INVALID"


@dataclass(frozen=True)
class TelemetryEvent:
    event_id: str
    timestamp: float
    event_type: str  # "SENSOR_SHOCK", "REMEDIATION", "AMBIENT_NOISE", "GRAPH_MUTATION", "DECISION_PROBE"
    entity_id: str
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposedAction:
    action_id: str
    action_name: str
    target_node: str
    required_prerequisites: List[str]  # List of causal node IDs that must be NOMINAL/VALID
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    permit: bool
    reason: str
    version: int
    violated_prerequisites: List[str] = field(default_factory=list)


@dataclass
class ContextPack:
    documents: List[ResearchDocument]
    token_budget: int
    tokens_used: int
    version: int
    doc_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubstrateSnapshot:
    version: int
    timestamp: float
    entity_states: Dict[str, EntityStatus]
    graph_edges: List[Tuple[str, str, str]]
    event_log_length: int


@dataclass
class OperationMetrics:
    writes: int = 0
    index_mutations: int = 0
    serialization_ops: int = 0
    invalidation_ops: int = 0
    cpu_time_ms: float = 0.0
    bytes_allocated: int = 0
    version_mismatches: int = 0
    service_calls: Dict[str, int] = field(default_factory=lambda: {
        "ingest": 0,
        "context": 0,
        "route": 0,
        "affected": 0,
        "search": 0,
        "verify": 0,
        "subscribe": 0,
    })

    def record_call(self, service_name: str):
        self.service_calls[service_name] = self.service_calls.get(service_name, 0) + 1


class ContextSubstrate(ABC):
    """
    Abstract Base Class for Context Substrates.
    Every service method takes an optional version watermark. If version is None,
    it reads the latest available snapshot version.
    """

    @abstractmethod
    def ingest(self, event: TelemetryEvent) -> int:
        """
        Ingest an event, update internal state/topology, advance version v -> v+1,
        and notify any matching changefeed subscribers.
        Returns the new version watermark.
        """
        pass

    @abstractmethod
    def context(self, query: str, token_budget: int, version: Optional[int] = None) -> ContextPack:
        """
        Service 1: Tiered context selection & packing (Tier 0: Status, Tier 1: Graph, Tier 2: Static Z).
        """
        pass

    @abstractmethod
    def route(self, event: TelemetryEvent, version: Optional[int] = None) -> Tuple[List[str], int]:
        """
        Service 2: Agent wake router.
        Identifies agent IDs whose operational scope overlaps with the event's entity,
        its active anomalies, graph neighbors, or semantic frontier.
        Returns (list_of_agent_ids, version_observed).
        """
        pass

    @abstractmethod
    def affected(self, entity_id: str, version: Optional[int] = None) -> Tuple[List[str], int]:
        """
        Service 3: Affected downstream frontier candidate generation.
        Returns (list_of_entity_ids, version_observed).
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, version: Optional[int] = None) -> Tuple[List[ResearchDocument], int]:
        """
        Service 4: Hybrid BM25 + dense multi-aspect search conditioned on operational status.
        Returns (ranked_documents, version_observed).
        """
        pass

    @abstractmethod
    def verify(self, action: ProposedAction, version: Optional[int] = None) -> VerificationResult:
        """
        Service 5: Invariant & prerequisite verification.
        Evaluates: 1[all prerequisites of action a hold in snapshot v].
        Returns VerificationResult.
        """
        pass

    @abstractmethod
    def subscribe(self, predicate: Callable[[TelemetryEvent], bool], callback: Callable[[TelemetryEvent, int], None]) -> str:
        """
        Service 6: Event-driven changefeed/subscription.
        Dispatches callback(event, version) whenever an ingested event satisfies predicate.
        Returns subscription_id.
        """
        pass

    @abstractmethod
    def get_snapshot(self, version: Optional[int] = None) -> SubstrateSnapshot:
        """Return the immutable substrate snapshot for version v."""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset operational metrics counters."""
        pass

    @abstractmethod
    def get_metrics(self) -> OperationMetrics:
        """Return tracked operational metrics."""
        pass
