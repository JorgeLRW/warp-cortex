from __future__ import annotations

import hashlib
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold_topology import (
    ManifoldTopologyView,
    build_manifold_topology,
    canonicalize_tokens,
    normalize_entity_refs,
    normalize_node_ids,
    overlap_score,
)
from .synapse import TopologicalSynapse
from .shared_manifold_store import SQLiteSharedManifoldStore
from .turbo_quant import TurboQuantCache, compress_landmarks, summarize_kv_cache


@dataclass
class AgentEpisode:
    """Persistent per-agent memory record."""

    text: str
    embedding: torch.Tensor
    score: float = 1.0
    source: str = "observation"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifoldNode:
    """Bounded shared-memory landmark stored outside any single agent identity."""

    text: str
    embedding: torch.Tensor
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    score: float = 1.0
    source: str = "observation"
    node_type: str = "memory"
    agent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPrediction:
    """Adapter prediction matched back to the nearest stored episode."""

    predicted_embedding: torch.Tensor
    best_episode: Optional[AgentEpisode]
    similarity: float
    ready: bool
    trained_steps: int


class LowRankMemoryAdapter(nn.Module):
    """
    Tiny per-agent adapter trained on detached states.

    The adapter learns a low-rank map from a query-like hidden state to a
    memory embedding. This lets an agent accumulate task-local adaptation
    without updating the shared backbone weights.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        rank: int = 8,
        lr: float = 1e-3,
        warmup_steps: int = 8,
        normalize_input: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.rank = rank
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.normalize_input = normalize_input
        self._device = device
        self._trained_steps = 0
        self.down_proj: Optional[nn.Linear] = None
        self.up_proj: Optional[nn.Linear] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

        if input_dim is not None:
            self._build(input_dim, device=device)

    @property
    def trained_steps(self) -> int:
        return self._trained_steps

    @property
    def ready(self) -> bool:
        return self._trained_steps >= self.warmup_steps

    def _build(self, input_dim: int, device: Optional[str] = None):
        if self.down_proj is not None and self.up_proj is not None:
            return

        target_device = device or self._device
        self.down_proj = nn.Linear(input_dim, self.rank, bias=False)
        self.up_proj = nn.Linear(self.rank, input_dim, bias=False)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
        if target_device is not None:
            self.down_proj = self.down_proj.to(target_device)
            self.up_proj = self.up_proj.to(target_device)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _prepare_vector(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.dim() != 1:
            vector = vector.reshape(-1)
        vector = vector.detach().float()
        if self.normalize_input:
            vector = F.normalize(vector, dim=0)
        if self.down_proj is None or self.up_proj is None:
            self._build(vector.shape[0], device=str(vector.device))
        assert self.down_proj is not None and self.up_proj is not None
        return vector.to(self.down_proj.weight.device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        vector = self._prepare_vector(hidden_state)
        assert self.down_proj is not None and self.up_proj is not None
        low_rank = self.down_proj(vector.unsqueeze(0))
        reconstructed = self.up_proj(low_rank).squeeze(0)
        return F.normalize(reconstructed, dim=0)

    @torch.no_grad()
    def predict(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.forward(hidden_state).detach()

    def partial_fit(self, hidden_state: torch.Tensor, target_embedding: torch.Tensor) -> float:
        vector = self._prepare_vector(hidden_state)
        target = self._prepare_vector(target_embedding)
        assert self.down_proj is not None and self.up_proj is not None
        assert self._optimizer is not None

        self.train()
        prediction = self.up_proj(self.down_proj(vector.unsqueeze(0))).squeeze(0)
        loss = F.mse_loss(prediction, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self.eval()
        self._trained_steps += 1
        return float(loss.item())

    def export_state(self) -> Dict[str, Any]:
        state_dict = None
        if self.down_proj is not None and self.up_proj is not None:
            state_dict = {
                key: value.detach().cpu()
                for key, value in self.state_dict().items()
            }

        return {
            "rank": self.rank,
            "lr": self.lr,
            "warmup_steps": self.warmup_steps,
            "normalize_input": self.normalize_input,
            "trained_steps": self._trained_steps,
            "state_dict": state_dict,
        }

    def load_export_state(self, payload: Dict[str, Any], device: Optional[str] = None):
        self.rank = int(payload.get("rank", self.rank))
        self.lr = float(payload.get("lr", self.lr))
        self.warmup_steps = int(payload.get("warmup_steps", self.warmup_steps))
        self.normalize_input = bool(payload.get("normalize_input", self.normalize_input))
        self._trained_steps = int(payload.get("trained_steps", 0))

        state_dict = payload.get("state_dict")
        if state_dict is None:
            return

        input_dim = int(state_dict["up_proj.weight"].shape[0])
        target_device = device or self._device
        self.down_proj = None
        self.up_proj = None
        self._optimizer = None
        self._build(input_dim, device=target_device)
        assert self.down_proj is not None and self.up_proj is not None
        restored_state = {
            key: value.to(self.down_proj.weight.device)
            for key, value in state_dict.items()
        }
        self.load_state_dict(restored_state)
        self.eval()


class PersistentAgentState:
    """Shared-weight agent identity with isolated memory and adaptation state."""

    def __init__(
        self,
        agent_id: str,
        hidden_dim: int,
        *,
        role: str = "agent",
        profile: str = "",
        device: str = "cpu",
        max_episodes: int = 128,
        adapter_rank: int = 8,
        synapse_ttl_seconds: float = 3600.0,
    ):
        self.agent_id = agent_id
        self.role = role
        self.profile = profile
        self.device = device
        self.max_episodes = max_episodes
        self.synapse = TopologicalSynapse(
            dim=hidden_dim,
            device=device,
            ttl_seconds=synapse_ttl_seconds,
        )
        self.adapter = LowRankMemoryAdapter(
            input_dim=hidden_dim,
            rank=adapter_rank,
            device=device,
        )
        self.episodes: List[AgentEpisode] = []
        self._lock = threading.Lock()

    def remember(
        self,
        *,
        text: str,
        embedding: torch.Tensor,
        score: float = 1.0,
        source: str = "observation",
        metadata: Optional[Dict[str, Any]] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> AgentEpisode:
        episode_embedding = embedding.detach().float().reshape(-1).cpu()
        episode = AgentEpisode(
            text=text,
            embedding=episode_embedding,
            score=float(score),
            source=source,
            metadata=dict(metadata or {}),
        )

        with self._lock:
            self.episodes.append(episode)
            if len(self.episodes) > self.max_episodes:
                self._evict_episode()

        self.synapse.inject_embedding(embedding.detach().float().to(self.device), score=float(score))

        if hidden_state is not None:
            self.adapter.partial_fit(hidden_state, embedding)

        return episode

    def recall(self, query_embedding: Optional[torch.Tensor], top_k: int = 3) -> List[AgentEpisode]:
        with self._lock:
            if not self.episodes:
                return []
            episodes = list(self.episodes)

        if query_embedding is None:
            return sorted(episodes, key=lambda item: (item.score, item.timestamp), reverse=True)[:top_k]

        query = F.normalize(query_embedding.detach().float().reshape(-1).cpu(), dim=0)
        matrix = torch.stack([F.normalize(item.embedding, dim=0) for item in episodes], dim=0)
        sims = torch.matmul(matrix, query)
        scores = torch.tensor([item.score for item in episodes], dtype=sims.dtype)
        combined = sims + 0.05 * scores
        limit = min(top_k, len(episodes))
        _, indices = torch.topk(combined, limit)
        return [episodes[int(idx)] for idx in indices.tolist()]

    def predict_memory(self, hidden_state: torch.Tensor) -> Optional[MemoryPrediction]:
        with self._lock:
            if not self.episodes:
                return None
            episodes = list(self.episodes)

        predicted = self.adapter.predict(hidden_state).cpu()
        matrix = torch.stack([F.normalize(item.embedding, dim=0) for item in episodes], dim=0)
        sims = torch.matmul(matrix, predicted)
        best_idx = int(torch.argmax(sims).item())
        best_episode = episodes[best_idx]
        similarity = float(sims[best_idx].item())
        return MemoryPrediction(
            predicted_embedding=predicted,
            best_episode=best_episode,
            similarity=similarity,
            ready=self.adapter.ready,
            trained_steps=self.adapter.trained_steps,
        )

    def build_context(self, query_embedding: Optional[torch.Tensor], top_k: int = 3) -> str:
        lines: List[str] = []
        if self.profile:
            lines.append(f"[Agent Profile] {self.profile}")

        recalls = self.recall(query_embedding, top_k=top_k)
        if recalls:
            lines.append("[Relevant Agent Memory]")
            for episode in recalls:
                lines.append(f"- {episode.text}")

        if query_embedding is not None:
            prediction = self.predict_memory(query_embedding)
            if prediction is not None and prediction.ready and prediction.best_episode is not None:
                recalled_texts = {episode.text for episode in recalls}
                if prediction.best_episode.text not in recalled_texts and prediction.similarity >= 0.35:
                    lines.append(f"[Adapter Recall Hint] {prediction.best_episode.text}")

        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            episode_count = len(self.episodes)
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "profile": self.profile,
            "episode_count": episode_count,
            "adapter_ready": self.adapter.ready,
            "adapter_trained_steps": self.adapter.trained_steps,
            "synapse_injections": self.synapse.injection_count,
        }

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            episodes = [
                {
                    "text": episode.text,
                    "embedding": episode.embedding.detach().cpu(),
                    "score": float(episode.score),
                    "source": episode.source,
                    "timestamp": float(episode.timestamp),
                    "metadata": dict(episode.metadata),
                }
                for episode in self.episodes
            ]

        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "profile": self.profile,
            "max_episodes": self.max_episodes,
            "episodes": episodes,
            "adapter": self.adapter.export_state(),
        }

    def restore_state(self, payload: Dict[str, Any]):
        self.role = payload.get("role", self.role)
        self.profile = payload.get("profile", self.profile)
        self.max_episodes = int(payload.get("max_episodes", self.max_episodes))
        self.episodes = []
        self.synapse = TopologicalSynapse(
            dim=self.synapse.dim,
            max_landmarks=self.synapse.max_landmarks,
            max_injections=self.synapse.max_injections,
            device=self.device,
            adaptive_k=self.synapse.adaptive_k,
            k_min=self.synapse.k_min,
            k_max=self.synapse.k_max,
            ttl_seconds=self.synapse.ttl_seconds,
        )

        adapter_payload = payload.get("adapter", {})
        self.adapter = LowRankMemoryAdapter(
            rank=int(adapter_payload.get("rank", self.adapter.rank)),
            lr=float(adapter_payload.get("lr", self.adapter.lr)),
            warmup_steps=int(adapter_payload.get("warmup_steps", self.adapter.warmup_steps)),
            normalize_input=bool(adapter_payload.get("normalize_input", self.adapter.normalize_input)),
            device=self.device,
        )
        self.adapter.load_export_state(adapter_payload, device=self.device)

        restored_episodes: List[AgentEpisode] = []
        for episode_payload in payload.get("episodes", []):
            episode = AgentEpisode(
                text=episode_payload["text"],
                embedding=episode_payload["embedding"].detach().float().reshape(-1).cpu(),
                score=float(episode_payload.get("score", 1.0)),
                source=episode_payload.get("source", "observation"),
                timestamp=float(episode_payload.get("timestamp", time.time())),
                metadata=dict(episode_payload.get("metadata") or {}),
            )
            restored_episodes.append(episode)
            self.synapse.inject_embedding(episode.embedding.to(self.device), score=episode.score)

        with self._lock:
            self.episodes = restored_episodes

    def _evict_episode(self):
        if not self.episodes:
            return
        self.episodes.sort(key=lambda item: (item.score, item.timestamp))
        self.episodes.pop(0)


class PersistentAgentCloud:
    """
    Shared-weight population manager for many semi-persistent agents.

    Each agent has isolated memory, its own synapse, and a tiny low-rank
    adapter trained on detached query-to-memory pairs. The backbone remains
    frozen and shared across the whole population.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        tokenizer=None,
        embed_layer=None,
        device: str = "cpu",
        max_episodes_per_agent: int = 128,
        shared_manifold_capacity: int = 256,
        shared_hot_capacity: int = 8,
        adapter_rank: int = 8,
        synapse_ttl_seconds: float = 3600.0,
        shared_store_path: Optional[str] = None,
        shared_store_cache_key: str = "default",
        shared_energy_feedback_enabled: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.device = device
        self.tokenizer = tokenizer
        self.embed_layer = embed_layer
        self.max_episodes_per_agent = max_episodes_per_agent
        self.shared_manifold_capacity = shared_manifold_capacity
        self.shared_hot_capacity = max(1, int(shared_hot_capacity))
        self.adapter_rank = adapter_rank
        self.synapse_ttl_seconds = synapse_ttl_seconds
        self.shared_store_path = shared_store_path
        self.shared_store_cache_key = shared_store_cache_key
        self.shared_energy_feedback_enabled = bool(shared_energy_feedback_enabled)
        self._agents: Dict[str, PersistentAgentState] = {}
        self._lock = threading.Lock()
        self._shared_nodes: List[ManifoldNode] = []
        self._shared_lock = threading.Lock()
        self._shared_store = (
            SQLiteSharedManifoldStore(shared_store_path)
            if shared_store_path
            else None
        )
        self._shared_hot_state: Dict[str, Any] = self._empty_hot_state()
        self._shared_hot_turbo_state: Optional[Dict[str, Any]] = None
        self._shared_projection_residues: Dict[str, Dict[str, Any]] = {}
        self.shared_energy_cap = 4.0
        self.shared_energy_decay = 0.55
        self.shared_energy_score_weight = 0.20
        self.shared_energy_floor = 0.02
        self.shared_energy_prompt_delta = 0.06
        self.shared_energy_refresh_delta = 0.08
        self.shared_energy_projection_delta = 0.10
        self.shared_energy_task_result_delta = 0.18
        self.shared_energy_task_failure_delta = -0.14
        self.shared_energy_store_delta = 0.10
        self.shared_energy_relation_weights = {
            "depends_on": 0.75,
            "supports": 0.65,
            "caused_by": 0.70,
            "blocks": -0.60,
            "related_to": 0.40,
            "projection_member": 0.35,
            "projection_bridge": 0.45,
            "component_member": 0.25,
        }
        self._manifold_maintenance_thread: Optional[threading.Thread] = None
        self._manifold_maintenance_stop = threading.Event()
        self._manifold_maintenance_seconds = 0.0
        self._manifold_maintenance_energy_decay = 0.98

        self._proj: Optional[nn.Linear] = None
        if embed_layer is not None:
            embed_dim = int(embed_layer.weight.shape[1])
            self._proj = nn.Linear(embed_dim, hidden_dim, bias=False)
            with torch.no_grad():
                if embed_dim == hidden_dim:
                    self._proj.weight.copy_(torch.eye(hidden_dim, dtype=embed_layer.weight.dtype))
                else:
                    nn.init.xavier_uniform_(self._proj.weight)
            self._proj = self._proj.to(
                device=str(embed_layer.weight.device),
                dtype=embed_layer.weight.dtype,
            )

        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()

    def ensure_agent(self, agent_id: str, *, role: str = "agent", profile: str = "") -> PersistentAgentState:
        with self._lock:
            state = self._agents.get(agent_id)
            if state is None:
                state = PersistentAgentState(
                    agent_id=agent_id,
                    hidden_dim=self.hidden_dim,
                    role=role,
                    profile=profile,
                    device=self.device,
                    max_episodes=self.max_episodes_per_agent,
                    adapter_rank=self.adapter_rank,
                    synapse_ttl_seconds=self.synapse_ttl_seconds,
                )
                self._agents[agent_id] = state
            else:
                if profile and not state.profile:
                    state.profile = profile
                if role and state.role == "agent":
                    state.role = role
            return state

    def get_agent(self, agent_id: str) -> Optional[PersistentAgentState]:
        return self._agents.get(agent_id)

    def get_agent_synapse(self, agent_id: str) -> TopologicalSynapse:
        return self.ensure_agent(agent_id).synapse

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if self.tokenizer is not None and self.embed_layer is not None and self._proj is not None:
            ids = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            ).input_ids.to(self.embed_layer.weight.device)
            token_embeds = self.embed_layer(ids)
            pooled = token_embeds.mean(dim=1)
            projected = self._proj(pooled).squeeze(0)
            return F.normalize(projected.float(), dim=0).cpu()

        seed = int(hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest(), 16) % (2 ** 31)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        vec = torch.randn(self.hidden_dim, generator=generator)
        return F.normalize(vec, dim=0)

    def _prepare_embedding(self, vector: torch.Tensor) -> torch.Tensor:
        prepared = vector.detach().float().reshape(-1).cpu()
        if float(prepared.norm().item()) > 0.0:
            prepared = F.normalize(prepared, dim=0)
        return prepared

    def _new_node_id(self) -> str:
        return uuid.uuid4().hex

    def _normalize_node_link_list(self, raw: Any) -> List[str]:
        return sorted(normalize_node_ids(raw))

    def _normalize_shared_metadata(self, text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        normalized = dict(metadata or {})

        task_id = normalized.get("task_id")
        if task_id is not None:
            normalized["task_id"] = str(task_id).strip()

        node_id = normalized.get("node_id")
        if node_id is not None:
            normalized["node_id"] = str(node_id).strip()

        for relation_key in (
            "depends_on",
            "supports",
            "caused_by",
            "blocks",
            "related_to",
            "projection_node_ids",
            "projection_bridge_node_ids",
            "component_node_ids",
        ):
            relation_values = self._normalize_node_link_list(normalized.get(relation_key))
            if relation_values:
                normalized[relation_key] = relation_values

        for list_key in ("acceptance_criteria", "trigger_terms", "selected_patches"):
            raw_value = normalized.get(list_key)
            if raw_value is None:
                continue
            if isinstance(raw_value, list):
                normalized[list_key] = [str(item).strip() for item in raw_value if str(item).strip()]
            else:
                value = str(raw_value).strip()
                normalized[list_key] = [value] if value else []

        raw_keywords = normalized.get("keywords")
        keyword_tokens = normalize_entity_refs(raw_keywords) if raw_keywords else canonicalize_tokens(text)
        if keyword_tokens:
            normalized["keywords"] = sorted(keyword_tokens)

        entity_tokens = normalize_entity_refs(
            normalized.get("entity_refs")
            or normalized.get("entities")
            or normalized.get("entity_ref")
        )
        if entity_tokens:
            normalized["entity_refs"] = sorted(entity_tokens)

        raw_energy = normalized.get("energy")
        if raw_energy is not None:
            try:
                energy_value = float(raw_energy)
            except (TypeError, ValueError):
                normalized.pop("energy", None)
            else:
                normalized["energy"] = max(-self.shared_energy_cap, min(self.shared_energy_cap, energy_value))

        raw_energy_updated_at = normalized.get("energy_updated_at")
        if raw_energy_updated_at is not None:
            try:
                normalized["energy_updated_at"] = float(raw_energy_updated_at)
            except (TypeError, ValueError):
                normalized.pop("energy_updated_at", None)

        return normalized

    def _node_energy(self, node: ManifoldNode) -> float:
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        try:
            return float(metadata.get("energy", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _set_node_energy(self, node: ManifoldNode, energy: float):
        metadata = dict(node.metadata or {})
        energy_value = max(-self.shared_energy_cap, min(self.shared_energy_cap, float(energy)))
        if abs(energy_value) < 1e-9:
            metadata.pop("energy", None)
            metadata.pop("energy_updated_at", None)
        else:
            metadata["energy"] = energy_value
            metadata["energy_updated_at"] = float(time.time())
        node.metadata = self._normalize_shared_metadata(node.text, metadata)

    def _edge_energy_weight(self, labels: List[str]) -> float:
        if not labels:
            return 0.18
        weights = [self.shared_energy_relation_weights.get(label, 0.25) for label in labels]
        return max(weights, key=lambda value: abs(value))

    def _energy_overlay(self, nodes: List[ManifoldNode], topology_view: ManifoldTopologyView) -> torch.Tensor:
        if not nodes:
            return torch.zeros(0, dtype=torch.float32)

        base = torch.tensor([self._node_energy(node) for node in nodes], dtype=torch.float32)
        if len(nodes) < 2 or not torch.any(base.abs() > 1e-9):
            return base

        overlay = base.clone()
        for edge_key, strength in topology_view.edge_strengths.items():
            left, right = edge_key
            propagation = self._edge_energy_weight(topology_view.edge_types.get(edge_key, []))
            propagation *= max(float(strength), 0.25) * self.shared_energy_decay
            overlay[left] += base[right] * propagation
            overlay[right] += base[left] * propagation
        return torch.clamp(overlay, -self.shared_energy_cap, self.shared_energy_cap)

    def _shared_energy_stats(
        self,
        nodes: List[ManifoldNode],
        *,
        topology_view: Optional[ManifoldTopologyView] = None,
    ) -> Dict[str, float]:
        if not nodes:
            return {
                "energized_node_count": 0.0,
                "energy_total": 0.0,
                "energy_abs_total": 0.0,
                "energy_peak": 0.0,
                "energy_overlay_peak": 0.0,
            }

        base = torch.tensor([self._node_energy(node) for node in nodes], dtype=torch.float32)
        if topology_view is None:
            topology_view = self._build_shared_topology_view(nodes)
        overlay = self._energy_overlay(nodes, topology_view)
        return {
            "energized_node_count": float(torch.count_nonzero(base.abs() > 1e-9).item()),
            "energy_total": float(base.sum().item()),
            "energy_abs_total": float(base.abs().sum().item()),
            "energy_peak": float(base.abs().max().item()),
            "energy_overlay_peak": float(overlay.abs().max().item()),
        }

    def _persist_shared_node_updates(self, nodes: List[ManifoldNode], *, refresh_hot_state: bool = True):
        if self._shared_store is not None and nodes:
            deduped_nodes = {node.node_id: node for node in nodes}
            for node in deduped_nodes.values():
                self._shared_store.upsert_node(node, capacity=self.shared_manifold_capacity)
            self._sync_shared_nodes_from_store(sync_hot_cache=False)
        if refresh_hot_state:
            self._refresh_shared_hot_state()

    def _dedupe_feedback_nodes(self, nodes: List[ManifoldNode]) -> List[ManifoldNode]:
        ordered_nodes: List[ManifoldNode] = []
        seen_node_ids: set[str] = set()
        for node in nodes:
            node_id = str(getattr(node, "node_id", "")).strip()
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            ordered_nodes.append(node)
        return ordered_nodes

    def deform_manifold_for_nodes(
        self,
        nodes: List[ManifoldNode],
        delta: float,
        *,
        max_depth: int = 1,
        edge_decay: float = 0.85,
        min_delta: float = 0.02,
        refresh_hot_state: bool = True,
    ) -> Dict[str, Any]:
        ordered_nodes = self._dedupe_feedback_nodes(nodes)
        if not ordered_nodes or abs(float(delta)) < 1e-9:
            return {
                "target_node_ids": [],
                "affected_node_count": 0,
                "node_energies": {},
            }

        node_energies: Dict[str, float] = {}
        affected_node_ids: set[str] = set()
        target_count = len(ordered_nodes)
        for index, node in enumerate(ordered_nodes):
            scaled_delta = float(delta) * max(0.45, 1.0 - 0.18 * index)
            if abs(scaled_delta) < float(min_delta):
                continue
            report = self.deform_manifold(
                node.node_id,
                scaled_delta,
                max_depth=max_depth,
                edge_decay=edge_decay,
                min_delta=min_delta,
                refresh_hot_state=refresh_hot_state and index == target_count - 1,
            )
            node_energies.update(report.get("node_energies", {}))
            affected_node_ids.update(report.get("node_energies", {}).keys())

        return {
            "target_node_ids": [node.node_id for node in ordered_nodes],
            "affected_node_count": len(affected_node_ids),
            "node_energies": node_energies,
        }

    def deform_manifold_for_query(
        self,
        query_text: str,
        delta: float,
        *,
        top_k: int = 4,
        agent_id: Optional[str] = None,
        include_projection: bool = True,
        max_depth: int = 1,
        edge_decay: float = 0.85,
        min_delta: float = 0.02,
        refresh_hot_state: bool = True,
    ) -> Dict[str, Any]:
        compact_query = str(query_text or "").strip()
        if not compact_query or abs(float(delta)) < 1e-9:
            return {
                "query_text": compact_query,
                "projection_id": "",
                "target_node_ids": [],
                "affected_node_count": 0,
                "node_energies": {},
            }

        limited_top_k = max(1, int(top_k))
        nodes = self.query_shared_manifold(
            query_text=compact_query,
            top_k=limited_top_k,
            agent_id=agent_id,
        )
        projection_id = ""
        if include_projection:
            projection = self.resolve_shared_projection(
                query_text=compact_query,
                top_k=limited_top_k,
                agent_id=agent_id,
                require_residue=False,
                materialize_missing=False,
            )
            if projection is not None:
                projection_id = str(projection.get("projection_id", "")).strip()
                nodes = [projection["node"], *(projection.get("member_nodes") or []), *nodes]

        report = self.deform_manifold_for_nodes(
            nodes,
            delta,
            max_depth=max_depth,
            edge_decay=edge_decay,
            min_delta=min_delta,
            refresh_hot_state=refresh_hot_state,
        )
        report["query_text"] = compact_query
        report["projection_id"] = projection_id
        return report

    def deform_task_board(
        self,
        task_id: str,
        delta: float,
        *,
        include_runtime_nodes: bool = True,
        max_depth: int = 1,
        edge_decay: float = 0.90,
        min_delta: float = 0.02,
        refresh_hot_state: bool = True,
    ) -> Dict[str, Any]:
        task_key = str(task_id or "").strip()
        if not task_key or abs(float(delta)) < 1e-9:
            return {
                "task_id": task_key,
                "target_node_ids": [],
                "affected_node_count": 0,
                "node_energies": {},
            }

        if self._shared_store is not None:
            self._sync_shared_nodes_from_store(sync_hot_cache=False)
        allowed_types = {"task_spec", "task_note", "task_patch"}
        if include_runtime_nodes:
            allowed_types.update({"task_claim", "task_result"})

        with self._shared_lock:
            nodes = [
                node
                for node in self._shared_nodes
                if self._task_board_task_id(node) == task_key and node.node_type in allowed_types
            ]

        feedback_priority = {
            "task_result": 0,
            "task_patch": 1,
            "task_note": 2,
            "task_spec": 3,
            "task_claim": 4,
        }
        nodes.sort(
            key=lambda node: (
                feedback_priority.get(node.node_type, 10),
                self._shared_display_order(node)[1],
            )
        )
        report = self.deform_manifold_for_nodes(
            nodes,
            delta,
            max_depth=max_depth,
            edge_decay=edge_decay,
            min_delta=min_delta,
            refresh_hot_state=refresh_hot_state,
        )
        report["task_id"] = task_key
        return report

    def _task_result_feedback_delta(self, *, status: str, score: float) -> float:
        normalized_status = str(status or "").strip().lower()
        magnitude = max(0.5, min(abs(float(score)), 2.0))
        if normalized_status in {"passed", "success", "completed", "done", "resolved"}:
            return self.shared_energy_task_result_delta * magnitude
        if normalized_status in {"failed", "error", "errored", "rejected", "blocked", "aborted", "cancelled", "canceled"}:
            return self.shared_energy_task_failure_delta * magnitude
        direction = 1.0 if float(score) >= 0.0 else -1.0
        return self.shared_energy_store_delta * magnitude * direction

    def _compact_board_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _task_node_priority(self, node_type: str) -> int:
        priorities = {
            "task_spec": 0,
            "task_note": 1,
            "task_patch": 2,
            "task_claim": 3,
            "task_result": 4,
        }
        return priorities.get(node_type, 10)

    def _task_board_task_id(self, node: ManifoldNode) -> str:
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        task_id = metadata.get("task_id", "")
        return str(task_id).strip()

    def is_task_board_node(self, node: ManifoldNode) -> bool:
        return node.node_type in {
            "task_spec",
            "task_note",
            "task_patch",
            "task_claim",
            "task_result",
        }

    def _task_board_prompt_node(self, node: ManifoldNode) -> bool:
        return node.node_type in {"task_spec", "task_note", "task_patch"}

    def _token_overlap(self, left: str, right: str) -> float:
        return overlap_score(canonicalize_tokens(left), canonicalize_tokens(right))

    def _build_shared_topology_view(self, nodes: List[ManifoldNode]) -> ManifoldTopologyView:
        semantic_enabled = self.tokenizer is not None and self.embed_layer is not None and self._proj is not None
        return build_manifold_topology(nodes, semantic_enabled=semantic_enabled)

    def _shared_centrality(self, matrix: torch.Tensor) -> torch.Tensor:
        if matrix.shape[0] < 2:
            return torch.zeros(matrix.shape[0], dtype=matrix.dtype)
        sims = matrix @ matrix.T
        sims.fill_diagonal_(0.0)
        neighbor_k = min(3, matrix.shape[0] - 1)
        top_vals, _ = torch.topk(sims, k=neighbor_k, dim=1)
        return top_vals.mean(dim=1)

    def _evict_shared_node(self):
        if not self._shared_nodes:
            return
        self._shared_nodes.sort(key=lambda item: (item.score, item.timestamp))
        self._shared_nodes.pop(0)

    def _empty_hot_state(self) -> Dict[str, Any]:
        return {
            "summary_text": "",
            "updated_at": 0.0,
            "node_count": 0,
            "hot_node_count": 0,
            "hot_node_texts": [],
            "hot_projection_id": "",
            "hot_projection_node_id": "",
            "topology": {
                "density": 0.0,
                "spread": 0.0,
                "coverage": 0.0,
                "node_count": 0.0,
            },
            "kv_stats": {
                "layer_count": 0,
                "layers": [],
                "original_bytes": 0,
                "compressed_bytes": 0,
                "compression_ratio": 1.0,
                "bits": None,
                "qjl_enabled": True,
                "compressed_layer_bytes": [],
            },
        }

    def _topology_accounting_nodes(self, nodes: List[ManifoldNode]) -> List[ManifoldNode]:
        return [
            node
            for node in nodes
            if str((node.metadata or {}).get("projection_kind", "")).strip() != "shared_hot_cache"
        ]

    def _compute_shared_topology(self, nodes: List[ManifoldNode]) -> Dict[str, float]:
        nodes = self._topology_accounting_nodes(nodes)
        if not nodes:
            return {
                "density": 0.0,
                "spread": 0.0,
                "coverage": 0.0,
                "node_count": 0.0,
                "component_count": 0.0,
                "largest_component_size": 0.0,
                "bridge_count": 0.0,
                "isolated_count": 0.0,
                "structural_edge_count": 0.0,
                "projection_node_count": 0.0,
            }

        count = len(nodes)
        coverage = count / max(self.shared_manifold_capacity, 1)
        topology_view = self._build_shared_topology_view(nodes)
        component_sizes = [len(component) for component in topology_view.components]
        largest_component = max(component_sizes, default=0)
        isolated_count = sum(1 for size in component_sizes if size == 1)
        structural_edge_count = float(len(topology_view.edge_types))
        projection_node_count = float(sum(1 for node in nodes if node.node_type == "projection_summary"))
        if count < 2:
            return {
                "density": 0.0,
                "spread": 0.0,
                "coverage": coverage,
                "node_count": float(count),
                "component_count": float(len(topology_view.components) or 1),
                "largest_component_size": float(largest_component or count),
                "bridge_count": float(len(topology_view.bridge_nodes)),
                "isolated_count": float(isolated_count or count),
                "structural_edge_count": structural_edge_count,
                "projection_node_count": projection_node_count,
            }

        matrix = torch.stack([self._prepare_embedding(node.embedding) for node in nodes], dim=0)
        sim_matrix = matrix @ matrix.T
        density = float(((sim_matrix.sum() - count) / max(count * (count - 1), 1)).item())
        centroid_raw = matrix.mean(dim=0)
        if float(centroid_raw.norm().item()) > 0.0:
            centroid = F.normalize(centroid_raw, dim=0)
            spread = float((1.0 - torch.matmul(matrix, centroid)).mean().item())
        else:
            spread = 0.0

        return {
            "density": density,
            "spread": spread,
            "coverage": coverage,
            "node_count": float(count),
            "component_count": float(len(topology_view.components)),
            "largest_component_size": float(largest_component),
            "bridge_count": float(len(topology_view.bridge_nodes)),
            "isolated_count": float(isolated_count),
            "structural_edge_count": structural_edge_count,
            "projection_node_count": projection_node_count,
        }

    def _rank_hot_nodes(self, nodes: List[ManifoldNode]) -> List[ManifoldNode]:
        if not nodes:
            return []

        matrix = torch.stack([self._prepare_embedding(node.embedding) for node in nodes], dim=0)
        centrality = self._shared_centrality(matrix)
        topology_view = self._build_shared_topology_view(nodes)
        energy_overlay = self._energy_overlay(nodes, topology_view)
        component_sizes = {
            component_id: len(component)
            for component_id, component in enumerate(topology_view.components)
        }
        now = time.time()
        scored: List[tuple[float, ManifoldNode]] = []
        for index, node in enumerate(nodes):
            age = max(now - float(node.timestamp), 0.0)
            recency_bonus = 1.0 / (1.0 + age / max(self.synapse_ttl_seconds, 1.0))
            component_id = topology_view.component_index.get(index, -1)
            component_bonus = 0.0
            if component_id >= 0:
                component_bonus = 0.05 * (component_sizes.get(component_id, 1) / max(len(nodes), 1))
            bridge_bonus = 0.18 if index in topology_view.bridge_nodes else 0.0
            energy_bonus = self.shared_energy_score_weight * float(energy_overlay[index].item())
            composite = (
                float(node.score)
                + 0.20 * float(centrality[index].item())
                + 0.15 * recency_bonus
                + component_bonus
                + bridge_bonus
                + energy_bonus
            )
            scored.append((composite, node))
        scored.sort(key=lambda item: (item[0], item[1].timestamp), reverse=True)
        return [node for _, node in scored[:self.shared_hot_capacity]]

    def _hot_cache_payload(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "hot_state": dict(self._shared_hot_state),
            "turbo_cache_state": self._shared_hot_turbo_state,
            "projection_residues": dict(self._shared_projection_residues),
        }

    def _sync_shared_nodes_from_store(self, *, sync_hot_cache: bool = True):
        if self._shared_store is None:
            return

        rows = self._shared_store.list_nodes(limit=self.shared_manifold_capacity)
        deduped_rows: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            row_node_id = str(row.get("node_id") or (row.get("metadata") or {}).get("node_id") or "").strip()
            if row_node_id:
                deduped_rows[row_node_id] = row
            else:
                deduped_rows[self._new_node_id()] = row
        nodes = [
            ManifoldNode(
                text=row["text"],
                embedding=self._prepare_embedding(row["embedding"]),
                node_id=str(row.get("node_id") or (row.get("metadata") or {}).get("node_id") or self._new_node_id()),
                score=float(row.get("score", 1.0)),
                source=row.get("source", "observation"),
                node_type=row.get("node_type", "memory"),
                agent_id=row.get("agent_id"),
                timestamp=float(row.get("timestamp", time.time())),
                metadata=self._normalize_shared_metadata(row["text"], dict(row.get("metadata") or {})),
            )
            for row in sorted(deduped_rows.values(), key=lambda item: float(item.get("timestamp", time.time())))
        ]
        with self._shared_lock:
            self._shared_nodes = nodes

        if sync_hot_cache:
            hot_payload = self._shared_store.read_hot_cache(cache_key=self.shared_store_cache_key)
            if hot_payload is not None:
                self._shared_hot_state = dict(hot_payload.get("hot_state") or self._empty_hot_state())
                self._shared_hot_turbo_state = hot_payload.get("turbo_cache_state")
                self._shared_projection_residues = dict(hot_payload.get("projection_residues") or {})

    def _refresh_shared_hot_state(
        self,
        *,
        kv_landmarks=None,
        turbo_bits: int = 4,
        turbo_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._shared_lock:
            nodes = list(self._shared_nodes)

        content_nodes = [
            node
            for node in nodes
            if str((node.metadata or {}).get("projection_kind", "")).strip() != "shared_hot_cache"
        ]

        topology_view = self._build_shared_topology_view(content_nodes) if content_nodes else ManifoldTopologyView(
            adjacency={},
            components=[],
            component_index={},
            bridge_nodes=set(),
            keyword_sets=[],
            entity_sets=[],
            node_ids=[],
            edge_types={},
            edge_strengths={},
        )
        topology = self._compute_shared_topology(content_nodes)
        hot_nodes = self._rank_hot_nodes(content_nodes)

        kv_stats = dict(self._shared_hot_state.get("kv_stats") or self._empty_hot_state()["kv_stats"])
        if kv_landmarks is not None:
            device = turbo_device or self.device
            turbo_cache = compress_landmarks(kv_landmarks, bits=turbo_bits, device=device)
            kv_stats = summarize_kv_cache(kv_landmarks, turbo_cache)
            self._shared_hot_turbo_state = turbo_cache.export_state()

        lines = [
            "[Shared Hot Cache]",
            (
                "[Topology: density="
                f"{topology['density']:.2f}, spread={topology['spread']:.2f}, coverage={topology['coverage']:.2f}, regions={int(topology.get('component_count', 0))}, largest_region={int(topology.get('largest_component_size', 0))}, bridges={int(topology.get('bridge_count', 0))}, isolated={int(topology.get('isolated_count', 0))}]"
            ),
        ]
        if kv_stats.get("layer_count", 0) > 0:
            original_mb = float(kv_stats.get("original_bytes", 0)) / (1024.0 * 1024.0)
            compressed_mb = float(kv_stats.get("compressed_bytes", 0)) / (1024.0 * 1024.0)
            ratio = float(kv_stats.get("compression_ratio", 1.0))
            bits = kv_stats.get("bits")
            lines.append(
                f"[KV: layers={kv_stats['layer_count']}, original={original_mb:.3f} MB, compressed={compressed_mb:.3f} MB, ratio={ratio:.2f}x, bits={bits}]"
            )
        for node in hot_nodes:
            lines.append(f"- [{node.node_type.replace('_', ' ')}] {node.text}")

        self._shared_hot_state = {
            "summary_text": "\n".join(lines) if hot_nodes or kv_stats.get("layer_count", 0) > 0 else "",
            "updated_at": float(time.time()),
            "node_count": len(content_nodes),
            "hot_node_count": len(hot_nodes),
            "hot_node_texts": [node.text for node in hot_nodes],
            "hot_projection_id": "",
            "hot_projection_node_id": "",
            "topology": topology,
            "kv_stats": kv_stats,
        }

        hot_projection = self._materialize_hot_projection(
            nodes=content_nodes,
            topology_view=topology_view,
            hot_nodes=hot_nodes,
            kv_stats=kv_stats,
        )
        if hot_projection is not None:
            self._shared_hot_state["hot_projection_id"] = hot_projection["projection_id"]
            self._shared_hot_state["hot_projection_node_id"] = hot_projection["node_id"]

        if self._shared_store is not None:
            self._shared_store.write_hot_cache(self._hot_cache_payload(), cache_key=self.shared_store_cache_key)
        return dict(self._shared_hot_state)

    def materialize_shared_hot_cache(
        self,
        *,
        kv_landmarks=None,
        turbo_bits: int = 4,
        turbo_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._refresh_shared_hot_state(
            kv_landmarks=kv_landmarks,
            turbo_bits=turbo_bits,
            turbo_device=turbo_device,
        )

    def get_shared_hot_state(self) -> Dict[str, Any]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        return {
            "summary_text": self._shared_hot_state.get("summary_text", ""),
            "updated_at": float(self._shared_hot_state.get("updated_at", 0.0)),
            "node_count": int(self._shared_hot_state.get("node_count", 0)),
            "hot_node_count": int(self._shared_hot_state.get("hot_node_count", 0)),
            "hot_node_texts": list(self._shared_hot_state.get("hot_node_texts", [])),
            "hot_projection_id": str(self._shared_hot_state.get("hot_projection_id", "")),
            "hot_projection_node_id": str(self._shared_hot_state.get("hot_projection_node_id", "")),
            "topology": dict(self._shared_hot_state.get("topology", {})),
            "kv_stats": dict(self._shared_hot_state.get("kv_stats", {})),
            "projection_residue_count": len(self._shared_projection_residues),
        }

    def get_shared_hot_turbo_cache(self, *, device: Optional[str] = None) -> Optional[TurboQuantCache]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        if self._shared_hot_turbo_state is None:
            return None
        return TurboQuantCache.from_state(self._shared_hot_turbo_state, device=device or self.device)

    def get_projection_residue(self, projection_id: str, *, device: Optional[str] = None) -> Optional[TurboQuantCache]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        payload = self._shared_projection_residues.get(str(projection_id).strip())
        if not payload:
            return None
        state = payload.get("turbo_cache_state")
        if state is None:
            return None
        return TurboQuantCache.from_state(state, device=device or self.device)

    def deform_manifold(
        self,
        node_id: str,
        delta: float,
        *,
        max_depth: int = 2,
        edge_decay: float = 0.85,
        min_delta: float = 0.02,
        refresh_hot_state: bool = True,
    ) -> Dict[str, Any]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store(sync_hot_cache=False)
        with self._shared_lock:
            nodes = list(self._shared_nodes)
        if not nodes:
            return {"affected_node_count": 0, "node_energies": {}}

        node_lookup = {node.node_id: index for index, node in enumerate(nodes)}
        start_index = node_lookup.get(str(node_id).strip())
        if start_index is None:
            return {"affected_node_count": 0, "node_energies": {}}

        topology_view = self._build_shared_topology_view(nodes)
        queue: List[tuple[int, float, int]] = [(start_index, float(delta), 0)]
        seen = {(start_index, 0)}
        contributions: Dict[int, float] = {}

        while queue:
            index, current_delta, depth = queue.pop(0)
            contributions[index] = contributions.get(index, 0.0) + current_delta
            if depth >= max(0, int(max_depth)):
                continue

            for neighbor in topology_view.adjacency.get(index, set()):
                edge_key = (min(index, neighbor), max(index, neighbor))
                propagated = current_delta
                propagated *= self._edge_energy_weight(topology_view.edge_types.get(edge_key, []))
                propagated *= max(float(topology_view.edge_strengths.get(edge_key, 0.0)), 0.25)
                propagated *= max(float(edge_decay), 0.0)
                if abs(propagated) < float(min_delta):
                    continue
                next_state = (neighbor, depth + 1)
                if next_state in seen:
                    continue
                seen.add(next_state)
                queue.append((neighbor, propagated, depth + 1))

        updated_nodes: List[ManifoldNode] = []
        with self._shared_lock:
            live_nodes = {node.node_id: node for node in self._shared_nodes}
            for index, contribution in contributions.items():
                target_node = live_nodes.get(nodes[index].node_id)
                if target_node is None:
                    continue
                current_energy = self._node_energy(target_node)
                self._set_node_energy(target_node, current_energy + float(contribution))
                updated_nodes.append(target_node)

        self._persist_shared_node_updates(updated_nodes, refresh_hot_state=refresh_hot_state)
        return {
            "affected_node_count": len(updated_nodes),
            "node_energies": {
                node.node_id: self._node_energy(node)
                for node in updated_nodes
            },
        }

    def run_manifold_maintenance(
        self,
        *,
        energy_decay: float = 0.98,
        energy_floor: float = 0.02,
        refresh_hot_state: bool = True,
    ) -> Dict[str, Any]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store(sync_hot_cache=False)

        updated_nodes: List[ManifoldNode] = []
        with self._shared_lock:
            energy_abs_before = sum(abs(self._node_energy(node)) for node in self._shared_nodes)
            for node in self._shared_nodes:
                current_energy = self._node_energy(node)
                if abs(current_energy) < 1e-9:
                    continue
                new_energy = current_energy * float(energy_decay)
                if abs(new_energy) < float(energy_floor):
                    new_energy = 0.0
                if abs(new_energy - current_energy) > 1e-9:
                    self._set_node_energy(node, new_energy)
                    updated_nodes.append(node)
            energy_abs_after = sum(abs(self._node_energy(node)) for node in self._shared_nodes)
            node_count = len(self._shared_nodes)

        self._persist_shared_node_updates(updated_nodes, refresh_hot_state=refresh_hot_state)
        return {
            "node_count": node_count,
            "updated_nodes": len(updated_nodes),
            "energy_abs_before": float(energy_abs_before),
            "energy_abs_after": float(energy_abs_after),
        }

    def start_manifold_maintenance_worker(
        self,
        *,
        interval_seconds: float = 60.0,
        energy_decay: float = 0.98,
    ):
        self.stop_manifold_maintenance_worker()
        self._manifold_maintenance_seconds = max(0.5, float(interval_seconds))
        self._manifold_maintenance_energy_decay = float(energy_decay)
        self._manifold_maintenance_stop.clear()
        thread = threading.Thread(
            target=self._manifold_maintenance_loop,
            name="warp_cortex_manifold_maintenance",
            daemon=True,
        )
        self._manifold_maintenance_thread = thread
        thread.start()

    def _manifold_maintenance_loop(self):
        while not self._manifold_maintenance_stop.wait(self._manifold_maintenance_seconds):
            self.run_manifold_maintenance(energy_decay=self._manifold_maintenance_energy_decay)

    def stop_manifold_maintenance_worker(self):
        self._manifold_maintenance_stop.set()
        thread = self._manifold_maintenance_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._manifold_maintenance_thread = None

    def _find_shared_node(self, *, node_id: Optional[str] = None, projection_id: Optional[str] = None) -> Optional[ManifoldNode]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        target_node_id = str(node_id or "").strip()
        target_projection_id = str(projection_id or "").strip()
        with self._shared_lock:
            for node in self._shared_nodes:
                if target_node_id and node.node_id == target_node_id:
                    return node
                if target_projection_id and str(node.metadata.get("projection_id", "")).strip() == target_projection_id:
                    return node
        return None

    def _projection_embedding(self, nodes: List[ManifoldNode]) -> torch.Tensor:
        matrix = torch.stack([self._prepare_embedding(node.embedding) for node in nodes], dim=0)
        centroid = matrix.mean(dim=0)
        return self._prepare_embedding(centroid)

    def _projection_summary_text(
        self,
        *,
        projection_kind: str,
        query_text: str,
        nodes: List[ManifoldNode],
        bridge_node_ids: List[str],
    ) -> str:
        snippets = [self._compact_board_text(node.text) for node in nodes[:3] if self._compact_board_text(node.text)]
        parts = [
            f"Projection {self._compact_board_text(projection_kind)} for {self._compact_board_text(query_text)}.",
        ]
        if snippets:
            parts.append("Focus: " + " | ".join(snippets))
        if bridge_node_ids:
            parts.append(f"Bridge count={len(bridge_node_ids)}.")
        return " ".join(part for part in parts if part)

    def _projection_candidate_score(
        self,
        node: ManifoldNode,
        *,
        query_text: str,
        query_embedding: torch.Tensor,
        query_keywords: set[str],
        query_entities: set[str],
        selected_related_ids: set[str],
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = node.metadata if isinstance(node.metadata, dict) else {}
        projection_id = str(metadata.get("projection_id", "")).strip()
        projection_node_ids = normalize_node_ids(metadata.get("projection_node_ids"))
        projection_query = self._compact_board_text(metadata.get("projection_query", ""))
        keyword_tokens = normalize_entity_refs(metadata.get("keywords"))
        entity_tokens = normalize_entity_refs(metadata.get("entity_refs"))
        semantic_score = float(torch.dot(self._prepare_embedding(node.embedding), query_embedding).item())
        lexical_score = self._token_overlap(query_text, node.text)
        query_match_score = self._token_overlap(query_text, projection_query)
        keyword_score = overlap_score(query_keywords, keyword_tokens)
        entity_score = overlap_score(query_entities, entity_tokens)
        membership_overlap = selected_related_ids & projection_node_ids
        membership_score = (
            len(membership_overlap) / max(min(len(selected_related_ids), len(projection_node_ids)), 1)
            if selected_related_ids and projection_node_ids else 0.0
        )
        same_agent_bonus = 0.05 if agent_id and node.agent_id == agent_id else 0.0
        residue_bonus = 0.12 if projection_id and projection_id in self._shared_projection_residues else 0.0
        energy_bonus = 0.10 * self._node_energy(node)
        hot_penalty = -0.04 if str(metadata.get("projection_kind", "")).strip() == "shared_hot_cache" else 0.0
        score = (
            semantic_score
            + 0.25 * lexical_score
            + 0.30 * query_match_score
            + 0.25 * membership_score
            + 0.15 * keyword_score
            + 0.10 * entity_score
            + same_agent_bonus
            + residue_bonus
            + energy_bonus
            + hot_penalty
        )
        return {
            "score": float(score),
            "projection_id": projection_id,
            "projection_node_ids": self._normalize_node_link_list(metadata.get("projection_node_ids")),
            "projection_bridge_node_ids": self._normalize_node_link_list(metadata.get("projection_bridge_node_ids")),
            "component_node_ids": self._normalize_node_link_list(metadata.get("component_node_ids")),
            "has_overlap": bool(membership_overlap),
            "has_query_hit": lexical_score > 0.0 or keyword_score > 0.0 or entity_score > 0.0,
            "has_residue": bool(projection_id and projection_id in self._shared_projection_residues),
            "projection_kind": self._compact_board_text(metadata.get("projection_kind", "projection")) or "projection",
        }

    def resolve_shared_projection(
        self,
        *,
        query_text: str,
        top_k: int = 4,
        agent_id: Optional[str] = None,
        require_residue: bool = False,
        materialize_missing: bool = False,
        projection_kind: str = "local_chart",
        kv_landmarks=None,
        turbo_bits: int = 4,
        turbo_device: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        with self._shared_lock:
            all_nodes = list(self._shared_nodes)
        if not all_nodes:
            return None

        selected_nodes, _, _ = self._select_shared_nodes(
            query_text=query_text,
            top_k=max(1, int(top_k)),
            agent_id=agent_id,
        )
        if not selected_nodes and not materialize_missing:
            return None

        selected_related_ids: set[str] = set()
        for selected_node in selected_nodes:
            selected_related_ids.add(selected_node.node_id)
            metadata = selected_node.metadata if isinstance(selected_node.metadata, dict) else {}
            selected_related_ids.update(normalize_node_ids(metadata.get("projection_node_ids")))
            selected_related_ids.update(normalize_node_ids(metadata.get("related_to")))

        query_embedding = self.encode_text(query_text)
        query_keywords = canonicalize_tokens(query_text)
        query_entities = normalize_entity_refs(query_text)
        node_lookup = {node.node_id: node for node in all_nodes}

        candidates: List[tuple[float, float, ManifoldNode, Dict[str, Any]]] = []
        for node in all_nodes:
            if node.node_type != "projection_summary":
                continue
            existing_projection_kind = str((node.metadata or {}).get("projection_kind", "")).strip()
            if existing_projection_kind == "shared_hot_cache":
                continue
            candidate = self._projection_candidate_score(
                node,
                query_text=query_text,
                query_embedding=query_embedding,
                query_keywords=query_keywords,
                query_entities=query_entities,
                selected_related_ids=selected_related_ids,
                agent_id=agent_id,
            )
            if require_residue and not candidate["has_residue"]:
                continue
            if candidate["score"] >= 0.18 or candidate["has_overlap"] or candidate["has_query_hit"]:
                candidates.append((candidate["score"], float(node.timestamp), node, candidate))

        if not candidates and materialize_missing and selected_nodes:
            payload = self.materialize_projection(
                query_text=query_text,
                top_k=max(1, int(top_k)),
                agent_id=agent_id,
                projection_kind=projection_kind,
                kv_landmarks=kv_landmarks,
                turbo_bits=turbo_bits,
                turbo_device=turbo_device,
            )
            if not payload:
                return None
            return self.resolve_shared_projection(
                query_text=query_text,
                top_k=top_k,
                agent_id=agent_id,
                require_residue=require_residue,
                materialize_missing=False,
            )

        if not candidates:
            return None

        _, _, projection_node, candidate = max(candidates, key=lambda item: (item[0], item[1]))
        member_nodes = [
            node_lookup[node_id]
            for node_id in candidate["projection_node_ids"]
            if node_id in node_lookup and node_id != projection_node.node_id
        ]
        if not member_nodes:
            member_nodes = [node for node in selected_nodes if node.node_id != projection_node.node_id]

        return {
            "projection_id": candidate["projection_id"],
            "node_id": projection_node.node_id,
            "node": projection_node,
            "summary_text": projection_node.text,
            "projection_kind": candidate["projection_kind"],
            "projection_node_ids": list(candidate["projection_node_ids"]),
            "projection_bridge_node_ids": list(candidate["projection_bridge_node_ids"]),
            "component_node_ids": list(candidate["component_node_ids"]),
            "member_nodes": member_nodes,
            "selected_nodes": list(selected_nodes),
            "score": float(candidate["score"]),
            "has_residue": bool(candidate["has_residue"]),
        }

    def _render_projection_context(self, projection: Dict[str, Any], *, heading: str) -> str:
        projection_kind = self._compact_board_text(projection.get("projection_kind", "projection")) or "projection"
        member_count = len(projection.get("projection_node_ids") or [])
        bridge_count = len(projection.get("projection_bridge_node_ids") or [])
        residue_flag = "yes" if projection.get("has_residue") else "no"
        lines = [
            heading,
            f"[Projection: kind={projection_kind}, members={member_count}, bridges={bridge_count}, residue={residue_flag}]",
            f"- [projection summary] {self._compact_board_text(projection.get('summary_text', ''))}",
        ]
        for node in projection.get("member_nodes") or []:
            label = node.node_type.replace("_", " ")
            lines.append(f"- [{label}] {node.text}")
            if len(lines) >= 5:
                break
        return "\n".join(line for line in lines if line)

    def _materialize_hot_projection(
        self,
        *,
        nodes: List[ManifoldNode],
        topology_view: ManifoldTopologyView,
        hot_nodes: List[ManifoldNode],
        kv_stats: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not hot_nodes:
            return None

        hot_node_ids = [node.node_id for node in hot_nodes]
        hot_node_id_set = set(hot_node_ids)
        bridge_node_ids = [
            nodes[index].node_id
            for index in topology_view.bridge_nodes
            if index < len(nodes) and nodes[index].node_id in hot_node_id_set
        ]
        projection_id = f"hot::{self.shared_store_cache_key}"
        projection_node_id = f"projection_hot_{self.shared_store_cache_key}"
        projection_query = "shared hot cache summary"
        summary_text = self._projection_summary_text(
            projection_kind="shared_hot_cache",
            query_text=projection_query,
            nodes=hot_nodes,
            bridge_node_ids=bridge_node_ids,
        )
        if kv_stats.get("layer_count", 0) > 0:
            summary_text = (
                f"{summary_text} KV ratio={float(kv_stats.get('compression_ratio', 1.0)):.2f}x "
                f"at {kv_stats.get('bits')} bits."
            ).strip()

        keywords = canonicalize_tokens(projection_query)
        entity_refs = set()
        for node in hot_nodes:
            metadata = node.metadata if isinstance(node.metadata, dict) else {}
            keywords.update(normalize_entity_refs(metadata.get("keywords") or node.text))
            entity_refs.update(normalize_entity_refs(metadata.get("entity_refs") or metadata.get("entities")))

        projection_node = self.remember_shared_text(
            text=summary_text,
            score=1.0,
            source="shared_hot_cache",
            node_type="projection_summary",
            metadata={
                "node_id": projection_node_id,
                "projection_id": projection_id,
                "projection_kind": "shared_hot_cache",
                "projection_query": projection_query,
                "projection_node_ids": hot_node_ids,
                "projection_bridge_node_ids": bridge_node_ids,
                "component_node_ids": hot_node_ids,
                "related_to": hot_node_ids,
                "keywords": sorted(keywords),
                "entity_refs": sorted(entity_refs),
                "kv_layer_count": int(kv_stats.get("layer_count", 0)),
                "kv_bits": kv_stats.get("bits"),
            },
            embedding=self._projection_embedding(hot_nodes),
            refresh_hot_state=False,
            replace_existing=True,
        )
        if self._shared_hot_turbo_state is not None:
            self._shared_projection_residues[projection_id] = {
                "projection_id": projection_id,
                "node_id": projection_node.node_id,
                "projection_node_ids": list(hot_node_ids),
                "updated_at": float(time.time()),
                "kv_stats": dict(kv_stats),
                "turbo_cache_state": self._shared_hot_turbo_state,
            }
        return {
            "projection_id": projection_id,
            "node_id": projection_node.node_id,
            "summary_text": summary_text,
        }

    def materialize_projection(
        self,
        *,
        query_text: str,
        top_k: int = 4,
        agent_id: Optional[str] = None,
        projection_kind: str = "local_chart",
        score: float = 1.0,
        source: str = "projection",
        kv_landmarks=None,
        turbo_bits: int = 4,
        turbo_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        nodes, topology_view, active_component = self._select_shared_nodes(
            query_text=query_text,
            top_k=top_k,
            agent_id=agent_id,
        )
        if not nodes:
            return {}

        node_ids = [node.node_id for node in nodes]
        component_node_ids = [
            topology_view.node_ids[index]
            for index in active_component
            if index < len(topology_view.node_ids)
        ]
        bridge_node_ids = [
            topology_view.node_ids[index]
            for index in topology_view.bridge_nodes
            if index < len(topology_view.node_ids) and topology_view.node_ids[index] in component_node_ids
        ]
        projection_signature = "|".join(
            [
                self._compact_board_text(projection_kind).lower(),
                self._compact_board_text(query_text).lower(),
                *sorted(node_ids),
            ]
        )
        projection_id = hashlib.blake2b(projection_signature.encode("utf-8"), digest_size=12).hexdigest()
        projection_node_id = f"projection_{projection_id}"

        keywords = canonicalize_tokens(query_text)
        entity_refs = set()
        for node in nodes:
            metadata = node.metadata if isinstance(node.metadata, dict) else {}
            keywords.update(normalize_entity_refs(metadata.get("keywords") or node.text))
            entity_refs.update(normalize_entity_refs(metadata.get("entity_refs") or metadata.get("entities")))

        summary_text = self._projection_summary_text(
            projection_kind=projection_kind,
            query_text=query_text,
            nodes=nodes,
            bridge_node_ids=bridge_node_ids,
        )
        projection_metadata = {
            "node_id": projection_node_id,
            "projection_id": projection_id,
            "projection_kind": self._compact_board_text(projection_kind),
            "projection_query": self._compact_board_text(query_text),
            "projection_node_ids": node_ids,
            "projection_bridge_node_ids": bridge_node_ids,
            "component_node_ids": component_node_ids,
            "related_to": node_ids,
            "keywords": sorted(keywords),
            "entity_refs": sorted(entity_refs),
        }

        projection_node = self._find_shared_node(projection_id=projection_id)
        if projection_node is None:
            projection_node = self.remember_shared_text(
                text=summary_text,
                score=score,
                source=source,
                node_type="projection_summary",
                agent_id=agent_id,
                metadata=projection_metadata,
                embedding=self._projection_embedding(nodes),
            )

        kv_stats: Dict[str, Any] = {}
        if kv_landmarks is not None:
            device = turbo_device or self.device
            turbo_cache = compress_landmarks(kv_landmarks, bits=turbo_bits, device=device)
            kv_stats = summarize_kv_cache(kv_landmarks, turbo_cache)
            self._shared_projection_residues[projection_id] = {
                "projection_id": projection_id,
                "node_id": projection_node.node_id,
                "projection_node_ids": list(node_ids),
                "updated_at": float(time.time()),
                "kv_stats": kv_stats,
                "turbo_cache_state": turbo_cache.export_state(),
            }
            if self._shared_store is not None:
                self._shared_store.write_hot_cache(self._hot_cache_payload(), cache_key=self.shared_store_cache_key)

        return {
            "projection_id": projection_id,
            "node_id": projection_node.node_id,
            "projection_kind": projection_metadata["projection_kind"],
            "query_text": projection_metadata["projection_query"],
            "projection_node_ids": list(node_ids),
            "projection_bridge_node_ids": list(bridge_node_ids),
            "component_node_ids": list(component_node_ids),
            "summary_text": summary_text,
            "kv_stats": kv_stats,
        }

    def remember_shared_text(
        self,
        *,
        text: str,
        score: float = 1.0,
        source: str = "observation",
        node_type: str = "memory",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[torch.Tensor] = None,
        refresh_hot_state: bool = True,
        replace_existing: bool = False,
    ) -> ManifoldNode:
        node_embedding = self._prepare_embedding(embedding) if embedding is not None else self.encode_text(text)
        normalized_metadata = self._normalize_shared_metadata(text, metadata)
        node_id = str(normalized_metadata.get("node_id") or self._new_node_id()).strip()
        normalized_metadata["node_id"] = node_id
        node = ManifoldNode(
            text=text,
            embedding=node_embedding,
            node_id=node_id,
            score=float(score),
            source=source,
            node_type=node_type,
            agent_id=agent_id,
            metadata=normalized_metadata,
        )
        if self._shared_store is not None:
            if replace_existing:
                self._shared_store.upsert_node(node, capacity=self.shared_manifold_capacity)
            else:
                self._shared_store.append_node(node, capacity=self.shared_manifold_capacity)
            self._sync_shared_nodes_from_store(sync_hot_cache=refresh_hot_state)
        else:
            with self._shared_lock:
                replaced = False
                if replace_existing:
                    for index, existing in enumerate(self._shared_nodes):
                        if existing.node_id == node.node_id:
                            self._shared_nodes[index] = node
                            replaced = True
                            break
                if not replaced:
                    self._shared_nodes.append(node)
                if len(self._shared_nodes) > self.shared_manifold_capacity:
                    self._evict_shared_node()
        if refresh_hot_state:
            self._refresh_shared_hot_state()
        return node

    def publish_task_spec(
        self,
        *,
        task_id: str,
        summary: str,
        recent_text: str = "",
        signature: str = "",
        acceptance_criteria: Optional[List[str]] = None,
        score: float = 1.0,
        source: str = "task_board",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifoldNode:
        task_key = str(task_id).strip()
        criteria = [item.strip() for item in (acceptance_criteria or []) if str(item).strip()]
        parts = [f"Task {task_key}.", self._compact_board_text(summary)]
        if recent_text:
            parts.append(f"Recent issue: {self._compact_board_text(recent_text)}")
        if signature:
            parts.append(f"Signature: {self._compact_board_text(signature)}")
        normalized_metadata = dict(metadata or {})
        normalized_metadata.update({
            "task_id": task_key,
            "task_summary": self._compact_board_text(summary),
            "task_recent_text": self._compact_board_text(recent_text),
            "task_signature": self._compact_board_text(signature),
            "acceptance_criteria": criteria,
        })
        normalized_metadata.setdefault("sequence_index", 0)
        return self.remember_shared_text(
            text=" ".join(part for part in parts if part),
            score=score,
            source=source,
            node_type="task_spec",
            agent_id=agent_id,
            metadata=normalized_metadata,
        )

    def publish_task_note(
        self,
        *,
        task_id: str,
        note_text: str,
        sequence_index: int = 0,
        score: float = 1.0,
        source: str = "task_board",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifoldNode:
        task_key = str(task_id).strip()
        normalized_metadata = dict(metadata or {})
        normalized_metadata.update({
            "task_id": task_key,
            "task_note": self._compact_board_text(note_text),
            "sequence_index": int(sequence_index),
        })
        return self.remember_shared_text(
            text=f"Task note for {task_key}. {self._compact_board_text(note_text)}",
            score=score,
            source=source,
            node_type="task_note",
            agent_id=agent_id,
            metadata=normalized_metadata,
        )

    def publish_task_patch(
        self,
        *,
        task_id: str,
        patch_name: str,
        old_text: str,
        new_text: str,
        trigger_terms: Optional[List[str]] = None,
        sequence_index: int = 0,
        score: float = 1.0,
        source: str = "task_board",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifoldNode:
        task_key = str(task_id).strip()
        trigger_list = [str(item).strip() for item in (trigger_terms or []) if str(item).strip()]
        old_compact = self._compact_board_text(old_text)
        new_compact = self._compact_board_text(new_text)
        parts = [
            f"Patch {patch_name} for task {task_key}.",
            f"Replace {old_compact} with {new_compact}.",
        ]
        if trigger_list:
            parts.append(f"Trigger terms: {', '.join(trigger_list)}.")
        normalized_metadata = dict(metadata or {})
        normalized_metadata.update({
            "task_id": task_key,
            "patch_name": str(patch_name).strip(),
            "patch_old": old_compact,
            "patch_new": new_compact,
            "trigger_terms": trigger_list,
            "sequence_index": int(sequence_index),
        })
        return self.remember_shared_text(
            text=" ".join(part for part in parts if part),
            score=score,
            source=source,
            node_type="task_patch",
            agent_id=agent_id,
            metadata=normalized_metadata,
        )

    def claim_task(
        self,
        *,
        task_id: str,
        agent_id: str,
        status: str = "claimed",
        lease_seconds: float = 300.0,
        score: float = 1.0,
        source: str = "task_board",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifoldNode:
        normalized_metadata = dict(metadata or {})
        normalized_metadata.update({
            "task_id": str(task_id).strip(),
            "claim_agent_id": str(agent_id).strip(),
            "claim_status": str(status).strip() or "claimed",
            "lease_expires_at": float(time.time() + max(lease_seconds, 0.0)),
        })
        return self.remember_shared_text(
            text=(
                f"Task claim for {str(task_id).strip()}. "
                f"Agent {str(agent_id).strip()} status={normalized_metadata['claim_status']}."
            ),
            score=score,
            source=source,
            node_type="task_claim",
            agent_id=agent_id,
            metadata=normalized_metadata,
        )

    def publish_task_result(
        self,
        *,
        task_id: str,
        agent_id: str,
        result_text: str,
        status: str,
        selected_patches: Optional[List[str]] = None,
        score: float = 1.0,
        source: str = "task_board",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManifoldNode:
        task_key = str(task_id).strip()
        normalized_metadata = dict(metadata or {})
        normalized_metadata.update({
            "task_id": task_key,
            "result_status": str(status).strip(),
            "selected_patches": [
                str(item).strip() for item in (selected_patches or []) if str(item).strip()
            ],
        })
        result_node = self.remember_shared_text(
            text=(
                f"Task result for {task_key}. "
                f"Agent {str(agent_id).strip()} status={normalized_metadata['result_status']}. "
                f"{self._compact_board_text(result_text)}"
            ),
            score=score,
            source=source,
            node_type="task_result",
            agent_id=agent_id,
            metadata=normalized_metadata,
            refresh_hot_state=not self.shared_energy_feedback_enabled,
        )
        if self.shared_energy_feedback_enabled:
            feedback_delta = self._task_result_feedback_delta(
                status=normalized_metadata["result_status"],
                score=score,
            )
            if abs(feedback_delta) > 1e-9:
                self.deform_task_board(
                    task_key,
                    feedback_delta,
                    include_runtime_nodes=True,
                    max_depth=1,
                    edge_decay=0.90,
                )
            else:
                self._refresh_shared_hot_state()
        return result_node

    def _select_shared_nodes(
        self,
        *,
        query_text: str,
        top_k: int = 4,
        agent_id: Optional[str] = None,
    ) -> tuple[List[ManifoldNode], ManifoldTopologyView, List[int]]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        with self._shared_lock:
            if not self._shared_nodes:
                empty_view = ManifoldTopologyView(
                    adjacency={},
                    components=[],
                    component_index={},
                    bridge_nodes=set(),
                    keyword_sets=[],
                    entity_sets=[],
                    node_ids=[],
                    edge_types={},
                    edge_strengths={},
                )
                return [], empty_view, []
            nodes = [
                node
                for node in self._shared_nodes
                if str((node.metadata or {}).get("projection_kind", "")).strip() != "shared_hot_cache"
            ]
        if not nodes:
            empty_view = ManifoldTopologyView(
                adjacency={},
                components=[],
                component_index={},
                bridge_nodes=set(),
                keyword_sets=[],
                entity_sets=[],
                node_ids=[],
                edge_types={},
                edge_strengths={},
            )
            return [], empty_view, []

        query_embedding = self.encode_text(query_text)
        matrix = torch.stack([self._prepare_embedding(node.embedding) for node in nodes], dim=0)
        sims = torch.matmul(matrix, query_embedding)
        centrality = self._shared_centrality(matrix)
        confidence = torch.tensor([node.score for node in nodes], dtype=sims.dtype)
        lexical = torch.tensor(
            [self._token_overlap(query_text, node.text) for node in nodes],
            dtype=sims.dtype,
        )
        same_agent = torch.tensor(
            [0.05 if agent_id and node.agent_id == agent_id else 0.0 for node in nodes],
            dtype=sims.dtype,
        )
        semantic_enabled = self.tokenizer is not None and self.embed_layer is not None and self._proj is not None
        if semantic_enabled:
            combined = sims + 0.35 * lexical + 0.05 * confidence + 0.05 * centrality + same_agent
        else:
            combined = 0.70 * lexical + 0.10 * confidence + 0.05 * centrality + same_agent

        topology_view = self._build_shared_topology_view(nodes)
        energy_overlay = self._energy_overlay(nodes, topology_view).to(dtype=combined.dtype)
        combined = combined + self.shared_energy_score_weight * energy_overlay
        global_order = torch.argsort(combined, descending=True).tolist()
        if not global_order:
            return [], topology_view, []

        anchor_idx = int(global_order[0])
        active_component_id = topology_view.component_index.get(anchor_idx, -1)
        active_component = (
            list(topology_view.components[active_component_id])
            if active_component_id >= 0 and active_component_id < len(topology_view.components)
            else [anchor_idx]
        )
        component_size = len(active_component)

        region_scores: Dict[int, float] = {}
        for index in active_component:
            bridge_bonus = 0.12 if index in topology_view.bridge_nodes else 0.0
            locality_bonus = 0.04 * (component_size / max(len(nodes), 1))
            edge_key = (min(anchor_idx, index), max(anchor_idx, index))
            structural_bonus = 0.10 if topology_view.edge_types.get(edge_key) else 0.0
            region_scores[index] = float(combined[index].item()) + bridge_bonus + locality_bonus + structural_bonus

        selected_indices: List[int] = [anchor_idx]
        if top_k > 1 and active_component:
            bridge_candidates = sorted(
                [index for index in active_component if index in topology_view.bridge_nodes and index != anchor_idx],
                key=lambda index: region_scores.get(index, float(combined[index].item())),
                reverse=True,
            )
            if bridge_candidates:
                selected_indices.append(int(bridge_candidates[0]))

        for index in sorted(active_component, key=lambda item: region_scores.get(item, float(combined[item].item())), reverse=True):
            if index not in selected_indices:
                selected_indices.append(int(index))
            if len(selected_indices) >= top_k:
                break

        for index in global_order:
            if len(selected_indices) >= top_k:
                break
            if index not in selected_indices:
                if index not in active_component:
                    base_score = float(combined[index].item())
                    lexical_score = float(lexical[index].item())
                    same_agent_bonus = float(same_agent[index].item())
                    if base_score < 0.25 and lexical_score <= 0.0 and same_agent_bonus <= 0.0:
                        continue
                selected_indices.append(int(index))

        selected: List[ManifoldNode] = []
        for index in selected_indices:
            base_score = float(combined[index].item())
            edge_key = (min(anchor_idx, index), max(anchor_idx, index))
            if (
                base_score >= 0.10
                or index == anchor_idx
                or index in topology_view.bridge_nodes
                or bool(topology_view.edge_types.get(edge_key))
            ):
                selected.append(nodes[index])
            if len(selected) >= top_k:
                break

        return selected, topology_view, active_component

    def query_shared_manifold(
        self,
        *,
        query_text: str,
        top_k: int = 4,
        agent_id: Optional[str] = None,
    ) -> List[ManifoldNode]:
        selected, _, _ = self._select_shared_nodes(query_text=query_text, top_k=top_k, agent_id=agent_id)
        return selected

    def query_task_board(
        self,
        *,
        query_text: str,
        top_k: int = 1,
        agent_id: Optional[str] = None,
        include_runtime_nodes: bool = False,
    ) -> List[ManifoldNode]:
        selected, _, _ = self._select_shared_nodes(
            query_text=query_text,
            top_k=max(1, int(top_k)),
            agent_id=agent_id,
        )
        task_ids: List[str] = []
        for node in selected:
            task_id = self._task_board_task_id(node)
            if task_id and task_id not in task_ids:
                task_ids.append(task_id)
            if len(task_ids) >= max(1, int(top_k)):
                break
        if not task_ids:
            return []

        allowed_types = {"task_spec", "task_note", "task_patch"}
        if include_runtime_nodes:
            allowed_types.update({"task_claim", "task_result"})

        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        with self._shared_lock:
            nodes = list(self._shared_nodes)

        bundled: List[ManifoldNode] = []
        for task_id in task_ids:
            task_nodes = [
                node
                for node in nodes
                if self._task_board_task_id(node) == task_id and node.node_type in allowed_types
            ]
            task_nodes.sort(key=self._shared_display_order)
            bundled.extend(task_nodes)
        return bundled

    def _render_task_board_context(self, nodes: List[ManifoldNode], *, heading: str) -> str:
        if not nodes:
            return ""

        grouped: Dict[str, List[ManifoldNode]] = {}
        for node in nodes:
            task_id = self._task_board_task_id(node)
            if not task_id:
                continue
            grouped.setdefault(task_id, []).append(node)
        if not grouped:
            return ""

        lines = [heading]
        task_ids = list(grouped.keys())
        for index, task_id in enumerate(task_ids):
            task_nodes = sorted(grouped[task_id], key=self._shared_display_order)
            spec_node = next((node for node in task_nodes if node.node_type == "task_spec"), None)
            spec_metadata = spec_node.metadata if spec_node is not None else {}

            lines.append(f"[Task: {task_id}]")
            summary = self._compact_board_text(spec_metadata.get("task_summary", ""))
            if summary:
                lines.append(f"summary={summary}")
            recent_text = self._compact_board_text(spec_metadata.get("task_recent_text", ""))
            if recent_text:
                lines.append(f"recent={recent_text}")
            signature = self._compact_board_text(spec_metadata.get("task_signature", ""))
            if signature:
                lines.append(f"signature={signature}")

            acceptance_criteria = spec_metadata.get("acceptance_criteria") or []
            for item in acceptance_criteria:
                compact_item = self._compact_board_text(item)
                if compact_item:
                    lines.append(f"acceptance={compact_item}")

            for node in task_nodes:
                metadata = node.metadata if isinstance(node.metadata, dict) else {}
                if node.node_type == "task_note":
                    note_text = self._compact_board_text(metadata.get("task_note", node.text))
                    if note_text:
                        lines.append(f"note={note_text}")
                elif node.node_type == "task_patch":
                    patch_name = self._compact_board_text(metadata.get("patch_name", node.text))
                    if patch_name:
                        lines.append(f"patch={patch_name}")
                    trigger_terms = [
                        self._compact_board_text(item) for item in (metadata.get("trigger_terms") or []) if self._compact_board_text(item)
                    ]
                    if trigger_terms:
                        lines.append(f"when={', '.join(trigger_terms)}")
                    old_text = self._compact_board_text(metadata.get("patch_old", ""))
                    if old_text:
                        lines.append(f"replace={old_text}")
                    new_text = self._compact_board_text(metadata.get("patch_new", ""))
                    if new_text:
                        lines.append(f"with={new_text}")

            if index < len(task_ids) - 1:
                lines.append("")

        return "\n".join(lines)

    def build_task_board_context(
        self,
        query_text: str,
        *,
        top_k: int = 1,
        agent_id: Optional[str] = None,
    ) -> str:
        nodes = self.query_task_board(query_text=query_text, top_k=top_k, agent_id=agent_id)
        return self._render_task_board_context(nodes, heading="[Task Board]")

    def shared_manifold_topology(self) -> Dict[str, float]:
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store()
        with self._shared_lock:
            nodes = list(self._shared_nodes)
        return self._compute_shared_topology(nodes)

    def build_shared_context(
        self,
        query_text: str,
        *,
        top_k: int = 4,
        agent_id: Optional[str] = None,
    ) -> str:
        nodes, topology_view, active_component = self._select_shared_nodes(
            query_text=query_text,
            top_k=top_k,
            agent_id=agent_id,
        )
        if not nodes:
            return ""
        has_task_board_nodes = any(self.is_task_board_node(node) for node in nodes)
        if has_task_board_nodes:
            task_board_context = self.build_task_board_context(
                query_text,
                top_k=max(1, top_k),
                agent_id=agent_id,
            )
            if task_board_context:
                return task_board_context
            if all(self.is_task_board_node(node) for node in nodes):
                return ""
            nodes = [node for node in nodes if not self.is_task_board_node(node)]
            if not nodes:
                return ""

        projection = self.resolve_shared_projection(
            query_text=query_text,
            top_k=top_k,
            agent_id=agent_id,
        )
        if projection is not None:
            return self._render_projection_context(projection, heading="[Shared Projection]")

        ordered_nodes = sorted(nodes, key=self._shared_display_order)

        topo = self.shared_manifold_topology()
        region_count = len(topology_view.components)
        bridge_count = len(topology_view.bridge_nodes)
        active_region_size = len(active_component)
        lines = [
            "[Shared Manifold]",
            (
                "[Topology: density="
                f"{topo['density']:.2f}, spread={topo['spread']:.2f}, coverage={topo['coverage']:.2f}, regions={region_count}, active_region={active_region_size}, bridges={bridge_count}]"
            ),
        ]
        for node in ordered_nodes:
            origin = f" from {node.agent_id}" if node.agent_id and node.agent_id != agent_id else ""
            label = node.node_type.replace("_", " ")
            lines.append(f"- [{label}{origin}] {node.text}")
        return "\n".join(lines)

    def plan_shared_injection(
        self,
        *,
        query_text: str,
        used_texts: Optional[set[str]] = None,
        top_k: int = 2,
        agent_id: Optional[str] = None,
    ) -> tuple[str, List[ManifoldNode]]:
        """Plan a compact shared-memory refresh for active decoding without repeating prior recalls."""
        nodes = self.query_shared_manifold(query_text=query_text, top_k=top_k, agent_id=agent_id)
        if not nodes:
            return "", []

        seen = used_texts or set()
        fresh_nodes = [node for node in nodes if node.text not in seen]
        if not fresh_nodes:
            return "", []
        if any(self.is_task_board_node(node) for node in fresh_nodes):
            board_nodes = [
                node
                for node in self.query_task_board(
                    query_text=query_text,
                    top_k=max(1, top_k),
                    agent_id=agent_id,
                )
                if node.text not in seen
            ]
            if not board_nodes:
                return "", []
            return self._render_task_board_context(board_nodes, heading="[Task Board Recall]"), board_nodes

        projection = self.resolve_shared_projection(
            query_text=query_text,
            top_k=top_k,
            agent_id=agent_id,
        )
        if projection is not None and projection["summary_text"] not in seen:
            projection_nodes = [projection["node"]]
            projection_nodes.extend(
                node for node in projection.get("member_nodes") or [] if node.text not in seen
            )
            return self._render_projection_context(projection, heading="[Shared Projection Recall]"), projection_nodes

        ordered_nodes = sorted(fresh_nodes, key=self._shared_display_order)

        lines = ["[Shared Recall]"]
        for node in ordered_nodes:
            label = node.node_type.replace("_", " ")
            lines.append(f"- [{label}] {node.text}")
        return "\n".join(lines), ordered_nodes

    def _shared_display_order(self, node: ManifoldNode) -> tuple[int, float]:
        priority = self._task_node_priority(node.node_type)
        sequence_value = node.metadata.get("sequence_index") if isinstance(node.metadata, dict) else None
        if sequence_value is None:
            return (priority, float(node.timestamp))
        try:
            return (priority, float(sequence_value))
        except (TypeError, ValueError):
            return (priority, float(node.timestamp))

    def shared_manifold_stats(self) -> Dict[str, Any]:
        topo = self.shared_manifold_topology()
        if self._shared_store is not None:
            self._sync_shared_nodes_from_store(sync_hot_cache=False)
        with self._shared_lock:
            nodes = self._topology_accounting_nodes(list(self._shared_nodes))
        energy_stats = self._shared_energy_stats(nodes)
        topo["node_count"] = int(topo["node_count"])
        topo["capacity"] = self.shared_manifold_capacity
        topo["component_count"] = int(topo.get("component_count", 0))
        topo["largest_component_size"] = int(topo.get("largest_component_size", 0))
        topo["bridge_count"] = int(topo.get("bridge_count", 0))
        topo["isolated_count"] = int(topo.get("isolated_count", 0))
        topo["structural_edge_count"] = int(topo.get("structural_edge_count", 0))
        topo["projection_node_count"] = int(topo.get("projection_node_count", 0))
        hot_state = self.get_shared_hot_state()
        kv_stats = hot_state.get("kv_stats", {})
        topo["hot_node_count"] = int(hot_state.get("hot_node_count", 0))
        topo["hot_summary_ready"] = bool(hot_state.get("summary_text"))
        topo["shared_store_path"] = self.shared_store_path or ""
        topo["kv_layer_count"] = int(kv_stats.get("layer_count", 0))
        topo["kv_original_bytes"] = int(kv_stats.get("original_bytes", 0))
        topo["kv_compressed_bytes"] = int(kv_stats.get("compressed_bytes", 0))
        topo["kv_compression_ratio"] = float(kv_stats.get("compression_ratio", 1.0))
        topo["kv_bits"] = kv_stats.get("bits")
        topo["projection_residue_count"] = int(hot_state.get("projection_residue_count", 0))
        topo["energized_node_count"] = int(energy_stats.get("energized_node_count", 0))
        topo["energy_total"] = float(energy_stats.get("energy_total", 0.0))
        topo["energy_abs_total"] = float(energy_stats.get("energy_abs_total", 0.0))
        topo["energy_peak"] = float(energy_stats.get("energy_peak", 0.0))
        topo["energy_overlay_peak"] = float(energy_stats.get("energy_overlay_peak", 0.0))
        return topo

    def compose_prompt(
        self,
        agent_id: str,
        *,
        task: str,
        role_prompt: str = "",
        upstream_context: str = "",
        top_k: int = 3,
    ) -> str:
        state = self.ensure_agent(agent_id)
        query_embedding = self.encode_text(task)
        context = state.build_context(query_embedding, top_k=top_k)
        shared_context = self.build_shared_context(task, top_k=top_k, agent_id=agent_id)
        parts: List[str] = []
        if role_prompt:
            parts.append(role_prompt)
        parts.append(f"[Persistent Agent: {agent_id}]")
        if state.role:
            parts.append(f"[Role: {state.role}]")
        if context:
            parts.append(context)
        if shared_context:
            parts.append(shared_context)
        if upstream_context:
            parts.append(upstream_context.strip())
        parts.append(f"Task: {task}. Analysis: ")
        return "\n".join(part for part in parts if part)

    def store_task_result(
        self,
        *,
        agent_id: str,
        task_text: str,
        result_text: str,
        result_vector: Optional[torch.Tensor],
        role: str = "agent",
        score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentEpisode:
        state = self.ensure_agent(agent_id, role=role)
        query_embedding = self.encode_text(task_text)
        result_embedding = (
            result_vector.detach().float().reshape(-1)
            if result_vector is not None
            else self.encode_text(result_text)
        )
        episode = state.remember(
            text=result_text,
            embedding=result_embedding,
            score=score,
            source="subagent",
            metadata=metadata,
            hidden_state=query_embedding,
        )
        shared_node = self.remember_shared_text(
            text=result_text,
            score=score,
            source="subagent",
            node_type="task_result",
            agent_id=agent_id,
            metadata=metadata,
            embedding=result_embedding,
            refresh_hot_state=not self.shared_energy_feedback_enabled,
        )
        if self.shared_energy_feedback_enabled:
            feedback_delta = self.shared_energy_store_delta * max(-1.5, min(1.5, float(score)))
            if abs(feedback_delta) > 1e-9:
                self.deform_manifold_for_nodes(
                    [shared_node],
                    feedback_delta,
                    max_depth=1,
                    edge_decay=0.80,
                    refresh_hot_state=False,
                )
                self.deform_manifold_for_query(
                    task_text,
                    feedback_delta * 0.80,
                    top_k=3,
                    agent_id=agent_id,
                    include_projection=True,
                    max_depth=1,
                    edge_decay=0.85,
                    refresh_hot_state=True,
                )
            else:
                self._refresh_shared_hot_state()
        return episode

    def remember_text(
        self,
        *,
        agent_id: str,
        text: str,
        score: float = 1.0,
        source: str = "observation",
        metadata: Optional[Dict[str, Any]] = None,
        hidden_state: Optional[torch.Tensor] = None,
        role: str = "agent",
        profile: str = "",
    ) -> AgentEpisode:
        state = self.ensure_agent(agent_id, role=role, profile=profile)
        embedding = self.encode_text(text)
        episode = state.remember(
            text=text,
            embedding=embedding,
            score=score,
            source=source,
            metadata=metadata,
            hidden_state=hidden_state,
        )
        self.remember_shared_text(
            text=text,
            score=score,
            source=source,
            node_type="agent_memory",
            agent_id=agent_id,
            metadata=metadata,
            embedding=embedding,
        )
        return episode

    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            states = list(self._agents.values())
        with self._shared_lock:
            shared_nodes = [
                {
                    "node_id": node.node_id,
                    "text": node.text,
                    "embedding": node.embedding.detach().cpu(),
                    "score": float(node.score),
                    "source": node.source,
                    "node_type": node.node_type,
                    "agent_id": node.agent_id,
                    "timestamp": float(node.timestamp),
                    "metadata": dict(node.metadata),
                }
                for node in self._shared_nodes
            ]

        proj_state = None
        if self._proj is not None:
            proj_state = {
                key: value.detach().cpu()
                for key, value in self._proj.state_dict().items()
            }

        return {
            "version": 4,
            "hidden_dim": self.hidden_dim,
            "max_episodes_per_agent": self.max_episodes_per_agent,
            "shared_manifold_capacity": self.shared_manifold_capacity,
            "shared_hot_capacity": self.shared_hot_capacity,
            "adapter_rank": self.adapter_rank,
            "synapse_ttl_seconds": self.synapse_ttl_seconds,
            "proj_state_dict": proj_state,
            "shared_nodes": shared_nodes,
            "shared_hot_state": dict(self._shared_hot_state),
            "shared_hot_turbo_state": self._shared_hot_turbo_state,
            "shared_projection_residues": dict(self._shared_projection_residues),
            "agents": [state.export_state() for state in states],
        }

    def save(self, file_path: str) -> str:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.export_state(), file_path)
        return file_path

    def load(self, file_path: str, *, merge: bool = False) -> Dict[str, int]:
        payload = torch.load(file_path, map_location="cpu")
        version = int(payload.get("version", 1))
        if version not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported agent cloud snapshot version: {payload.get('version')}")

        hidden_dim = int(payload.get("hidden_dim", self.hidden_dim))
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Snapshot hidden_dim={hidden_dim} does not match cloud hidden_dim={self.hidden_dim}"
            )

        proj_state = payload.get("proj_state_dict")
        if self._proj is not None and proj_state is not None:
            restored_proj_state = {
                key: value.to(device=self._proj.weight.device, dtype=self._proj.weight.dtype)
                for key, value in proj_state.items()
            }
            self._proj.load_state_dict(restored_proj_state)

        with self._lock:
            if not merge:
                self._agents = {}
        with self._shared_lock:
            if not merge:
                self._shared_nodes = []
        if not merge:
            self._shared_projection_residues = {}

        self.shared_manifold_capacity = int(payload.get("shared_manifold_capacity", self.shared_manifold_capacity))
        self.shared_hot_capacity = int(payload.get("shared_hot_capacity", self.shared_hot_capacity))

        shared_loaded = 0
        for node_payload in payload.get("shared_nodes", []):
            node = ManifoldNode(
                text=node_payload["text"],
                embedding=self._prepare_embedding(node_payload["embedding"]),
                node_id=str(node_payload.get("node_id") or (node_payload.get("metadata") or {}).get("node_id") or self._new_node_id()),
                score=float(node_payload.get("score", 1.0)),
                source=node_payload.get("source", "observation"),
                node_type=node_payload.get("node_type", "memory"),
                agent_id=node_payload.get("agent_id"),
                timestamp=float(node_payload.get("timestamp", time.time())),
                metadata=self._normalize_shared_metadata(
                    node_payload["text"],
                    dict(node_payload.get("metadata") or {}),
                ),
            )
            with self._shared_lock:
                self._shared_nodes.append(node)
            shared_loaded += 1

        loaded_agents = 0
        for agent_payload in payload.get("agents", []):
            state = PersistentAgentState(
                agent_id=agent_payload["agent_id"],
                hidden_dim=self.hidden_dim,
                role=agent_payload.get("role", "agent"),
                profile=agent_payload.get("profile", ""),
                device=self.device,
                max_episodes=int(agent_payload.get("max_episodes", self.max_episodes_per_agent)),
                adapter_rank=int(payload.get("adapter_rank", self.adapter_rank)),
                synapse_ttl_seconds=float(payload.get("synapse_ttl_seconds", self.synapse_ttl_seconds)),
            )
            state.restore_state(agent_payload)
            with self._lock:
                self._agents[state.agent_id] = state
            loaded_agents += 1

        with self._lock:
            total_agents = len(self._agents)

        projection_payload = dict(payload.get("shared_projection_residues") or {})
        if merge:
            self._shared_projection_residues.update(projection_payload)
        else:
            self._shared_projection_residues = projection_payload
        self._shared_hot_state = dict(payload.get("shared_hot_state") or self._empty_hot_state())
        self._shared_hot_turbo_state = payload.get("shared_hot_turbo_state")
        if not self._shared_hot_state.get("summary_text"):
            self._refresh_shared_hot_state()

        return {
            "loaded_agents": loaded_agents,
            "total_agents": total_agents,
            "shared_nodes": shared_loaded,
        }

    def population_stats(self) -> Dict[str, Any]:
        with self._lock:
            states = list(self._agents.values())
        return {
            "agent_count": len(states),
            "total_episodes": sum(len(state.episodes) for state in states),
            "ready_adapters": sum(int(state.adapter.ready) for state in states),
            "shared_manifold": self.shared_manifold_stats(),
            "agents": [state.stats() for state in states],
        }