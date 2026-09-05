"""
Semantic Context Fabric: Self-Organizing Context Space with Frequency-like Multi-Aspect Decomposition.
=====================================================================================================

Thesis:
Semantics != causality, but Semantics is an innate coordinate system for context.
Rather than treating all knowledge as an undifferentiated vector soup searched globally via flat ANN,
information decomposes into multi-aspect frequency bands:
    x -> {aspect_1, aspect_2, ..., aspect_m}
e.g. {instrumentation, data_validity, mechanism, manufacturing, safety, unit_economics}.

Context assembly is hierarchical:
    Semantic Routing -> State-Conditioned Activation -> Local Search -> Structural Completion -> Context Packing

State-Conditioned Invariant:
    Identical queries yield different, correct contexts when the underlying world state changes:
    Context(q, S_{t1}) != Context(q, S_{t2}).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


class SemanticBand(str, Enum):
    INSTRUMENTATION = "instrumentation"
    DATA_VALIDITY = "data_validity"
    MECHANISM = "mechanism"
    MANUFACTURING = "manufacturing"
    SAFETY = "safety"
    UNIT_ECONOMICS = "unit_economics"
    GENERAL = "general"


@dataclass
class FabricItem:
    """
    An information item residing in the Semantic Context Fabric.
    Carries multi-aspect coordinates (frequency bands), structural linkage, and dynamic state.
    """
    item_id: str
    title: str
    content: str
    aspect_vectors: Dict[str, torch.Tensor]  # normalized D-dimensional vectors per aspect band
    primary_aspect: str = SemanticBand.GENERAL.value
    metadata: Dict[str, Any] = field(default_factory=dict)
    causal_node_id: Optional[str] = None     # linkage to EpistemicManifold node
    dynamic_energy: float = 0.0              # dynamic strain / activation h_i(t)
    validity_status: str = "VALID"           # "VALID", "SUSPECT", "TAINTED", "FALSIFIED"
    timestamp: float = field(default_factory=time.time)

    def estimated_tokens(self) -> int:
        """Heuristic token count for context budget management."""
        return max(1, len(self.content.split()) + len(self.title.split()) + 10)


@dataclass
class RetrievedContext:
    """A packed context bundle ready for consumption by an LLM or agent."""
    items: List[FabricItem]
    total_tokens: int
    active_compartments: List[str]
    structural_links_traversed: int
    state_boost_applied: bool
    summary_text: str


class SemanticContextFabric:
    """
    Persistent Semantic Address Space & Context Fabric.
    
    Organizes information across multi-aspect semantic frequency bands,
    enabling hierarchical routing, state-conditioned context assembly,
    and innate context extraction without query reformulation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        available_bands: Optional[List[str]] = None,
    ):
        self.hidden_dim = hidden_dim
        self.bands = available_bands or [b.value for b in SemanticBand]
        
        # Compartmentalized storage: aspect_name -> Set of item_ids
        self.compartments: Dict[str, Set[str]] = {b: set() for b in self.bands}
        
        # Primary registry: item_id -> FabricItem
        self.items: Dict[str, FabricItem] = {}
        
        # Band prototype anchors in R^D
        self.band_anchors: Dict[str, torch.Tensor] = {}
        self._initialize_band_anchors()

        # Telemetry
        self.total_queries = 0
        self.total_compartment_hits = 0
        self.total_local_scans = 0

    def _initialize_band_anchors(self) -> None:
        """Initialize orthogonal reference directions for semantic frequency bands."""
        torch.manual_seed(1337)
        for band in self.bands:
            v = torch.randn(self.hidden_dim)
            self.band_anchors[band] = F.normalize(v, dim=0)

    def register_item(
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
        """
        Store an information item in its natural semantic compartments.
        An item occupies every band for which it provides an aspect vector.
        """
        norm_aspects: Dict[str, torch.Tensor] = {}
        for band, vec in aspect_vectors.items():
            norm_aspects[band] = F.normalize(vec.detach().float().reshape(-1), dim=0)

        # Determine primary aspect if not provided
        if not primary_aspect:
            if norm_aspects:
                best_band = max(
                    norm_aspects.keys(),
                    key=lambda b: torch.dot(norm_aspects[b], self.band_anchors.get(b, norm_aspects[b])).item()
                )
                primary_aspect = best_band
            else:
                primary_aspect = SemanticBand.GENERAL.value

        item = FabricItem(
            item_id=item_id,
            title=title,
            content=content,
            aspect_vectors=norm_aspects,
            primary_aspect=primary_aspect,
            metadata=metadata or {},
            causal_node_id=causal_node_id,
            dynamic_energy=0.0,
            validity_status=validity_status,
        )

        self.items[item_id] = item

        # Compartmentalize
        for band in norm_aspects.keys():
            if band not in self.compartments:
                self.compartments[band] = set()
            self.compartments[band].add(item_id)
        if primary_aspect not in self.compartments:
            self.compartments[primary_aspect] = set()
        self.compartments[primary_aspect].add(item_id)

        return item

    def update_dynamic_state(
        self,
        item_id: str,
        energy_delta: float = 0.0,
        validity_status: Optional[str] = None,
    ) -> None:
        """Update dynamic activation energy h_i(t) or validity of an item."""
        if item_id in self.items:
            item = self.items[item_id]
            if energy_delta != 0.0:
                item.dynamic_energy = max(0.0, item.dynamic_energy + energy_delta)
            if validity_status is not None:
                item.validity_status = validity_status

    # -------------------------------------------------------------------------
    # Hierarchical Context Assembly Pipeline
    # -------------------------------------------------------------------------

    def assemble_context(
        self,
        query: str,
        query_embedding: Optional[torch.Tensor] = None,
        target_aspects: Optional[List[str]] = None,
        token_budget: int = 1024,
        state_weight: float = 0.40,
        epistemic_manifold: Optional[Any] = None,
        include_structural_neighbors: bool = True,
    ) -> RetrievedContext:
        """
        Hierarchical 5-Stage Context Assembly:
        Stage 1: Semantic Compartment Routing
        Stage 2: State-Conditioned Activation & Filtering
        Stage 3: Local ANN Search within candidate compartments
        Stage 4: Structural Expansion along causal/provenance graph
        Stage 5: Context Packing within token budget
        """
        self.total_queries += 1

        if query_embedding is None:
            import hashlib
            seed = int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            q_emb = F.normalize(torch.randn(self.hidden_dim), dim=0)
        else:
            q_emb = F.normalize(query_embedding.detach().float().reshape(-1), dim=0)

        # STAGE 1: Semantic Compartment Routing
        active_comps_set: Set[str] = set(target_aspects or [])
        if not active_comps_set:
            band_scores = {}
            for band, anchor in self.band_anchors.items():
                band_scores[band] = torch.dot(q_emb, anchor).item()
            sorted_bands = sorted(band_scores.items(), key=lambda x: x[1], reverse=True)
            active_comps_set = {b[0] for b in sorted_bands[:2]}

        # State-Conditioned Bias: if any compartment contains active strain or anomaly, include it
        if state_weight > 0:
            for item in self.items.values():
                if item.dynamic_energy > 0.3 or item.validity_status in ("TAINTED", "FALSIFIED", "SUSPECT"):
                    active_comps_set.add(item.primary_aspect)
                    for ab in item.aspect_vectors.keys():
                        active_comps_set.add(ab)

        active_compartments = list(active_comps_set)

        # Candidate item pool: items residing in the active compartments
        candidate_ids: Set[str] = set()
        for comp in active_compartments:
            candidate_ids.update(self.compartments.get(comp, set()))

        if not candidate_ids:
            candidate_ids = set(self.items.keys())

        self.total_compartment_hits += len(candidate_ids)
        self.total_local_scans += len(candidate_ids)

        # STAGE 2 & 3: State-Conditioned Activation & Local Search
        scored_items: List[Tuple[float, FabricItem]] = []
        for iid in candidate_ids:
            item = self.items[iid]
            
            # Semantic alignment across item's aspect vectors
            best_sim = 0.0
            for aspect_name, a_vec in item.aspect_vectors.items():
                sim = torch.dot(q_emb, a_vec).item()
                if sim > best_sim:
                    best_sim = sim
            
            # Dynamic state term: h_i(t) and validity
            state_bonus = 0.0
            if item.dynamic_energy > 0:
                state_bonus += item.dynamic_energy * state_weight
            
            # Severe penalty or boost based on current world state:
            if item.validity_status in ("TAINTED", "FALSIFIED"):
                state_bonus += 0.50 * state_weight

            final_score = (1.0 - state_weight) * best_sim + state_weight * state_bonus
            scored_items.append((final_score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)

        # STAGE 4: Structural Expansion (Follow Causal Links in G)
        structural_items: List[FabricItem] = []
        structural_count = 0
        if include_structural_neighbors and epistemic_manifold:
            top_candidates = [item for _, item in scored_items[:5]]
            for item in top_candidates:
                if item.causal_node_id and item.causal_node_id in epistemic_manifold.nodes:
                    related_node_ids = set()
                    for edge in epistemic_manifold.edges:
                        if edge.target_id == item.causal_node_id:
                            related_node_ids.add(edge.source_id)
                        elif edge.source_id == item.causal_node_id:
                            related_node_ids.add(edge.target_id)
                    
                    for r_id in related_node_ids:
                        for fb_item in self.items.values():
                            if fb_item.causal_node_id == r_id and fb_item.item_id != item.item_id:
                                if fb_item not in structural_items and fb_item not in [x[1] for x in scored_items]:
                                    structural_items.append(fb_item)
                                    structural_count += 1

        # STAGE 5: Context Packing within Token Budget
        selected_items: List[FabricItem] = []
        used_tokens = 0
        seen_ids: Set[str] = set()

        pool: List[FabricItem] = []
        high_strain = [it for _, it in scored_items if it.dynamic_energy > 0.3 or it.validity_status != "VALID"]
        standard = [it for _, it in scored_items if it not in high_strain]
        
        pool.extend(high_strain)
        pool.extend(standard[:3])
        pool.extend(structural_items)
        pool.extend(standard[3:])

        for item in pool:
            if item.item_id in seen_ids:
                continue
            toks = item.estimated_tokens()
            if used_tokens + toks <= token_budget:
                selected_items.append(item)
                used_tokens += toks
                seen_ids.add(item.item_id)
            elif used_tokens < token_budget * 0.5:
                selected_items.append(item)
                used_tokens += toks
                seen_ids.add(item.item_id)
                break

        summary_lines = []
        for idx, it in enumerate(selected_items, 1):
            flag = f" [{it.validity_status}]" if it.validity_status != "VALID" else ""
            strain = f" [Strain: {it.dynamic_energy:.2f}]" if it.dynamic_energy > 0 else ""
            summary_lines.append(f"{idx}. [{it.primary_aspect.upper()}]{flag}{strain} {it.title}: {it.content}")
        summary_text = "\n".join(summary_lines)

        return RetrievedContext(
            items=selected_items,
            total_tokens=used_tokens,
            active_compartments=active_compartments,
            structural_links_traversed=structural_count,
            state_boost_applied=(state_weight > 0),
            summary_text=summary_text,
        )

    # -------------------------------------------------------------------------
    # Innate Context Assembly (Unprompted Event Context)
    # -------------------------------------------------------------------------

    def assemble_innate_context(
        self,
        trigger_entity_id: str,
        epistemic_manifold: Optional[Any] = None,
        token_budget: int = 1024,
    ) -> RetrievedContext:
        """
        Assemble decision-relevant context directly from an incoming event or entity
        WITHOUT any user query or LLM search formulation.
        Uses Z_entity + h_t + G to surface the innate neighborhood.
        """
        if trigger_entity_id not in self.items:
            matching = [iid for iid in self.items if trigger_entity_id.lower() in iid.lower()]
            if matching:
                trigger_entity_id = matching[0]
            else:
                return RetrievedContext([], 0, [], 0, False, "No innate context found.")

        trigger_item = self.items[trigger_entity_id]
        
        active_comps = list(trigger_item.aspect_vectors.keys())
        if not active_comps:
            active_comps = [trigger_item.primary_aspect]

        structural_items: List[FabricItem] = [trigger_item]
        structural_traversed = 0
        
        if epistemic_manifold and trigger_item.causal_node_id:
            root_node_id = trigger_item.causal_node_id
            
            upstream_nodes = set()
            downstream_nodes = set()
            
            for edge in epistemic_manifold.edges:
                if edge.target_id == root_node_id:
                    upstream_nodes.add(edge.source_id)
                elif edge.source_id == root_node_id:
                    downstream_nodes.add(edge.target_id)

            for u_id in upstream_nodes:
                for fb in self.items.values():
                    if fb.causal_node_id == u_id and fb not in structural_items:
                        structural_items.append(fb)
                        structural_traversed += 1

            for d_id in downstream_nodes:
                for fb in self.items.values():
                    if fb.causal_node_id == d_id and fb not in structural_items:
                        structural_items.append(fb)
                        structural_traversed += 1

        compartment_items: List[FabricItem] = []
        for comp in active_comps:
            for iid in self.compartments.get(comp, set()):
                it = self.items[iid]
                if it not in structural_items and (it.dynamic_energy > 0 or it.validity_status != "VALID"):
                    compartment_items.append(it)

        packed_items: List[FabricItem] = []
        used_tokens = 0
        seen: Set[str] = set()

        for it in (structural_items + compartment_items):
            if it.item_id in seen:
                continue
            toks = it.estimated_tokens()
            if used_tokens + toks <= token_budget:
                packed_items.append(it)
                used_tokens += toks
                seen.add(it.item_id)

        summary_lines = []
        for idx, it in enumerate(packed_items, 1):
            flag = f" [{it.validity_status}]" if it.validity_status != "VALID" else ""
            summary_lines.append(f"{idx}. [{it.primary_aspect.upper()}]{flag} {it.title}: {it.content}")
        summary_text = "\n".join(summary_lines)

        return RetrievedContext(
            items=packed_items,
            total_tokens=used_tokens,
            active_compartments=active_comps,
            structural_links_traversed=structural_traversed,
            state_boost_applied=True,
            summary_text=summary_text,
        )
