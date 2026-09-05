"""
Continuous Semantic Reaction Field Harness for Warp Cortex.

Implements a persistent, continuous reaction field over semantic space M = S^{D-1}:
- Agents, entities, and hypotheses possess multi-prototype semantic coordinates Z_i = {z_i1, ..., z_ik}.
- Real-world/player events inject localized perturbations via a radial Gaussian kernel on S^{D-1}:
      I_i(e) = max_{z in Z_i} exp(-d_M(z, z_e)^2 / (2 * sigma^2)) * u_e
- Perturbations propagate dynamically across the persistent reaction field h_t(z):
      h^{t+1} = (1 - gamma) * h^t + alpha * (W_norm @ h^t)
- When an entity's potential crosses its activation threshold (h_i >= theta_i), it awakens.
- Awakened entities can emit secondary perturbations that propagate downstream to multi-hop dependents.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ManifoldEntity:
    """An AI character, specialized agent, or concept on the continuous semantic sphere S^{D-1}."""
    entity_id: str
    name: str
    role: str
    embedding: torch.Tensor                        # Centroid coordinate [hidden_dim] on S^{D-1}
    prototypes: Dict[str, torch.Tensor] = field(default_factory=dict) # Multi-prototype aspects
    activation_threshold: float = 0.5              # Potential threshold to trigger execution
    current_energy: float = 0.0                    # Dynamic reaction potential h_t(z_i)
    base_prompt: str = ""                          # Character / agent system prompt anchor
    state_metadata: Dict[str, Any] = field(default_factory=dict)

    def is_triggered(self) -> bool:
        return self.current_energy >= self.activation_threshold


@dataclass
class ManifoldImpulse:
    """A perturbation injected into the continuous field by an event or agent reaction."""
    event_id: str
    text: str
    embedding: torch.Tensor                        # Semantic coordinate on S^{D-1}
    magnitude: float = 1.0                         # Energy magnitude u_e
    timestamp: float = field(default_factory=time.time)
    source: str = "world"


class ContinuousReactionManifold:
    """
    Continuous Semantic Reaction Field Engine.
    
    Maintains a dynamic state field h_t(z) over stationary semantic coordinates on S^{D-1}.
    Events perturb the field via radial Gaussian kernels; energy diffuses across coupled
    topological and prototype similarities.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        decay_rate: float = 0.20,        # Field damping factor gamma (strictly contractive)
        diffusion_rate: float = 0.15,    # Viscosity / coupling strength alpha (alpha <= gamma ensures stability)
        semantic_threshold: float = 0.30,# Minimum prototype similarity to establish a coupling edge
        kernel_sigma: float = 0.75,      # Radial kernel bandwidth sigma on S^{D-1}
    ):
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.diffusion_rate = diffusion_rate
        self.semantic_threshold = semantic_threshold
        self.kernel_sigma = kernel_sigma

        self.entities: Dict[str, ManifoldEntity] = {}
        self.history_impulses: List[ManifoldImpulse] = []
        self._adjacency_matrix: Optional[torch.Tensor] = None
        self._entity_keys: List[str] = []

    def register_entity(
        self,
        entity_id: str,
        name: str,
        role: str,
        embedding: Optional[torch.Tensor] = None,
        prototypes: Optional[Dict[str, torch.Tensor]] = None,
        activation_threshold: float = 0.35,
        base_prompt: str = "",
        state_metadata: Optional[Dict[str, Any]] = None,
        rebuild_topology: bool = True,
    ) -> ManifoldEntity:
        """
        Add an entity with multi-prototype semantic aspects to the continuous field.
        If prototypes are omitted, a single 'core' aspect is constructed from embedding.
        """
        if embedding is not None:
            emb = F.normalize(embedding.detach().float().reshape(-1), dim=0)
        else:
            emb = F.normalize(torch.randn(self.hidden_dim), dim=0)

        proto_dict: Dict[str, torch.Tensor] = {}

        if prototypes:
            for aspect_name, proto_vec in prototypes.items():
                proto_dict[aspect_name] = F.normalize(proto_vec.detach().float().reshape(-1), dim=0)
            # Centroid is normalized sum of all aspect prototypes
            stacked = torch.stack(list(proto_dict.values()), dim=0)
            emb = F.normalize(stacked.mean(dim=0), dim=0)
        else:
            proto_dict["core"] = emb

        entity = ManifoldEntity(
            entity_id=entity_id,
            name=name,
            role=role,
            embedding=emb,
            prototypes=proto_dict,
            activation_threshold=activation_threshold,
            base_prompt=base_prompt,
            state_metadata=state_metadata or {},
        )
        self.entities[entity_id] = entity
        if rebuild_topology:
            self._rebuild_topology()
        return entity

    def _rebuild_topology(self) -> None:
        """
        Construct the cross-entity coupling matrix W over multi-prototype aspect overlaps.
        W_{ij} = max_{a in Z_i, b in Z_j} max(0, a . b)
        Symmetrically normalized: W_norm = D^{-1/2} W D^{-1/2}.
        """
        self._entity_keys = list(self.entities.keys())
        n = len(self._entity_keys)
        if n == 0:
            self._adjacency_matrix = None
            return

        adj = torch.zeros((n, n), dtype=torch.float32)
        # Pre-stack all entity prototypes once to avoid redundant stacking in O(N^2) loops
        stacks = [torch.stack(list(self.entities[k].prototypes.values()), dim=0) for k in self._entity_keys]

        for i in range(n):
            stack_i = stacks[i]  # [K_i, D]
            for j in range(i + 1, n):
                stack_j = stacks[j]  # [K_j, D]

                # Pairwise similarities between all aspects of i and j
                sim_matrix = torch.mm(stack_i, stack_j.t())  # [K_i, K_j]
                max_sim = float(sim_matrix.max().item())

                if max_sim >= self.semantic_threshold:
                    adj[i, j] = max_sim
                    adj[j, i] = max_sim

        # Symmetrically normalize adjacency for stable diffusion (D^{-1/2} A D^{-1/2})
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        deg_inv_sqrt = deg.pow(-0.5)
        self._adjacency_matrix = deg_inv_sqrt * adj * deg_inv_sqrt.t()

    def inject_impulse(
        self,
        text: str,
        embedding: torch.Tensor,
        magnitude: float = 1.0,
        source: str = "world",
        event_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Inject an event perturbation into the semantic reaction field using a radial Gaussian kernel on S^{D-1}.
        
        For each entity i:
            d(z, z_e) = arccos(clamp(z . z_e, -1.0, 1.0))
            K(d) = exp(-d^2 / (2 * sigma^2))
            I_i(e) = max_{z in Z_i} K(d(z, z_e)) * magnitude
        """
        emb = F.normalize(embedding.detach().float().reshape(-1), dim=0)
        impulse = ManifoldImpulse(
            event_id=event_id or f"impulse_{len(self.history_impulses)}",
            text=text,
            embedding=emb,
            magnitude=magnitude,
            source=source,
        )
        self.history_impulses.append(impulse)

        direct_hits: Dict[str, float] = {}
        for entity_id, entity in self.entities.items():
            best_k = 0.0
            for proto in entity.prototypes.values():
                dot = float(torch.dot(proto, emb).clamp(-1.0, 1.0).item())
                # Geodesic / angular distance on the sphere
                dist = math.acos(dot)
                # Radial Gaussian kernel
                k_val = math.exp(-(dist ** 2) / (2.0 * (self.kernel_sigma ** 2)))
                if k_val > best_k:
                    best_k = k_val

            delta_e = magnitude * best_k
            entity.current_energy += delta_e
            direct_hits[entity_id] = delta_e

        return direct_hits

    def step_diffusion(self, steps: int = 1) -> List[ManifoldEntity]:
        """
        Diffuse energy across the continuous field over discrete time steps:
            h^{t+1} = (1 - gamma) * h^t + alpha * (W_norm @ h^t)
        Returns entities whose potential crossed their activation threshold.
        """
        if not self.entities or self._adjacency_matrix is None:
            return []

        energy_vec = torch.tensor(
            [self.entities[k].current_energy for k in self._entity_keys],
            dtype=torch.float32,
        )

        for _ in range(steps):
            diffused = torch.matmul(self._adjacency_matrix, energy_vec)
            energy_vec = torch.clamp(
                (1.0 - self.decay_rate) * energy_vec + self.diffusion_rate * diffused,
                min=0.0,
                max=5.0,
            )

        triggered_entities: List[ManifoldEntity] = []
        for i, k in enumerate(self._entity_keys):
            entity = self.entities[k]
            entity.current_energy = float(energy_vec[i].item())
            if entity.is_triggered():
                triggered_entities.append(entity)

        return triggered_entities

    def emit_reaction(
        self,
        entity_id: str,
        text: str,
        aspect: Optional[str] = None,
        magnitude: float = 0.6,
    ) -> Dict[str, float]:
        """
        When an entity awakens and reacts, it emits a secondary localized perturbation
        into the field from its specific aspect coordinate, rippling to downstream entities.
        """
        if entity_id not in self.entities:
            raise KeyError(f"Entity {entity_id} not registered in reaction field")

        entity = self.entities[entity_id]
        if aspect and aspect in entity.prototypes:
            coord = entity.prototypes[aspect]
        else:
            coord = entity.embedding

        return self.inject_impulse(
            text=text,
            embedding=coord,
            magnitude=magnitude,
            source=f"agent:{entity_id}",
        )

    def cool_down_entity(self, entity_id: str, factor: float = 0.2) -> None:
        """Discharge potential after an entity finishes executing."""
        if entity_id in self.entities:
            self.entities[entity_id].current_energy *= factor

    def get_manifold_state_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot of the continuous reaction field."""
        return {
            "entity_count": len(self.entities),
            "entities": {
                k: {
                    "name": e.name,
                    "role": e.role,
                    "aspects": list(e.prototypes.keys()),
                    "energy": round(e.current_energy, 4),
                    "threshold": e.activation_threshold,
                    "triggered": e.is_triggered(),
                }
                for k, e in self.entities.items()
            },
            "impulse_count": len(self.history_impulses),
            "recent_impulse": self.history_impulses[-1].text if self.history_impulses else None,
        }
