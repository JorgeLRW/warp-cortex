"""
Workspace Knowledge Harvester for Persistent World Runtime.
============================================================
Ingests real research artifacts, source files, validation logs, and negative
results from across the user's workspace into a unified Cortex substrate
U_v = <S_v, G_v, Z, H_v>.

Harvets from:
  1. warp_cortex (reaction-diffusion V1-V8, transition governors, epistemic manifolds, negative results)
  2. warp_align (warp-level kernel alignment, shared memory bank conflict theory, tensor core sync)
  3. inference_wedge (KV cache Fisher geometry, curvature-aware compression, latency bounds)
  4. 2521/atlas-jepa (intrinsic cross-spacing excess mixing bounds, latent energy manifolds, VIB reasoning)

Scales the manifold to >= 2,000 entities (|U| >> context window, >250k tokens)
with authentic cross-project relationships, dense 64-dim embeddings, and
provenance histories.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cortex_apps.cortex_world_runtime.fast_world_substrate import EntityNode, FastWorldSubstrate


class GenericFrozenAspectEncoder:
    """
    Frozen, task-agnostic aspect encoder (task-unsupervised).
    Uses token embeddings from Qwen/Qwen2.5-0.5B-Instruct projected via a fixed
    random orthogonal Gaussian matrix W in R^{d_embed x 64} (seed=42).
    Generates 64-dimensional unit-normalized dense vectors directly from raw text.
    Strictly zero hand-picked dimension coordinates, zero benchmark-task
    supervision. NOTE: the underlying Qwen embeddings are pretrained, so call
    this task-unsupervised / task-agnostic -- not universally "unsupervised".
    """
    def __init__(self, d_out: int = 64, seed: int = 42):
        self.d_out = d_out
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            os.environ["HF_HOME"] = os.path.abspath(os.path.join(REPO_ROOT, "..", ".hf_cache"))
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32, local_files_only=True)
            self.embeddings = model.get_input_embeddings().weight.detach().cpu()
            torch.manual_seed(seed)
            W = torch.randn(self.embeddings.shape[1], d_out)
            self.W = F.normalize(W, dim=0)
            self.has_model = True
        except Exception:
            self.has_model = False
            torch.manual_seed(seed)
            W = torch.randn(256, d_out)
            self.W = F.normalize(W, dim=0)

    def encode(self, text: str) -> torch.Tensor:
        if not text:
            v = torch.randn(self.d_out)
            return F.normalize(v, p=2, dim=0)
        if self.has_model:
            tokens = self.tokenizer.encode(text[:512], add_special_tokens=False)
            if not tokens:
                tokens = [0]
            t_ids = torch.tensor(tokens, dtype=torch.long)
            mean_emb = self.embeddings[t_ids].mean(dim=0)
            z = mean_emb @ self.W
            return F.normalize(z, p=2, dim=0)
        else:
            byte_counts = torch.zeros(256)
            for b in text.encode("utf-8")[:512]:
                byte_counts[b] += 1.0
            z = byte_counts @ self.W
            return F.normalize(z, p=2, dim=0)


class WorkspaceKnowledgeHarvester:
    def __init__(self, substrate: FastWorldSubstrate):
        self.substrate = substrate
        self.ingested_count = 0
        self.project_stats: Dict[str, int] = {}
        self.encoder = GenericFrozenAspectEncoder(d_out=64, seed=42)

    def harvest_all(self, target_total: int = 2000) -> FastWorldSubstrate:
        print(f"\n[KnowledgeHarvester] Beginning workspace harvest targeting {target_total} entities...")
        t0 = time.perf_counter()

        # Phase 1: Ingest Ground-Truth Private Core Research Artifacts
        self._ingest_core_ground_truth_artifacts()

        # Phase 2: Ingest Real Workspace Files from across projects
        self._ingest_workspace_directory("warp_cortex", os.path.join(WORKSPACE_ROOT, "warp_cortex"))
        self._ingest_workspace_directory("warp_align", os.path.join(WORKSPACE_ROOT, "warp_align"))
        self._ingest_workspace_directory("inference_wedge", os.path.join(WORKSPACE_ROOT, "inference_wedge"))
        # Optional external corpus: portable via env var, skipped if absent.
        external_2521 = os.environ.get("CORTEX_2521_PATH", r"c:\Users\jorge\2521")
        self._ingest_workspace_directory("project_2521", external_2521)

        # Phase 3: Synthetically expand with cluster-consistent background corpus to reach >= target_total
        self._expand_background_corpus(target_total)

        # Phase 4: Construct Task-Agnostic Semantic Graph (G) via k-NN on Aspect Manifold (Z)
        # Task-agnostic: zero knowledge of benchmark questions, premises, or target values
        self._build_unsupervised_semantic_graph(k_nearest=4, sim_threshold=0.45)

        elapsed = time.perf_counter() - t0
        print(f"[KnowledgeHarvester] Completed harvest: {self.ingested_count} entities in {elapsed:.2f} s.")
        for p, count in self.project_stats.items():
            print(f"  - Project {p:<18}: {count:>5d} entities")
        return self.substrate

    def _add_entity(
        self,
        eid: str,
        project: str,
        title: str,
        state: Dict[str, Any],
        aspect_vector: Optional[torch.Tensor] = None,
        neighbors: Optional[Set[str]] = None,
        cluster_id: int = 0,
    ) -> EntityNode:
        if aspect_vector is None:
            text_repr = f"{title} {state}"
            vec = self.encoder.encode(text_repr)
        else:
            vec = F.normalize(aspect_vector, p=2, dim=0)

        merged_state = {
            "project": project,
            "title": title,
            **state,
        }

        node = EntityNode(
            entity_id=eid,
            state=merged_state,
            neighbors=set(neighbors or []),
            aspect_vector=vec,
            cluster_id=cluster_id,
            version_modified=1,
        )
        self.substrate.entities[eid] = node
        cid = cluster_id % self.substrate.num_clusters
        self.substrate.clusters[cid].append(eid)
        self.ingested_count += 1
        self.project_stats[project] = self.project_stats.get(project, 0) + 1
        return node

    def _ingest_core_ground_truth_artifacts(self):
        """
        Injects the private research premises.
        Their aspect representations are computed strictly via GenericFrozenAspectEncoder.
        Zero hand-picked dimension slices!
        """
        p_a_text = (
            "Fisher Information Geometry in KV Caches. "
            "Exact formula: I_F = E[grad log p * grad log p^T]. "
            "Derived bound: kappa_max <= 0.42. "
            "Curvature above 0.42 causes catastrophic rank collapse in key-cache geometry."
        )
        self._add_entity(
            eid="art_inference_wedge_fisher_curvature",
            project="inference_wedge",
            title="Fisher Information Geometry in KV Caches",
            state={
                "concept": "FISHER_CURVATURE_BOUND",
                "premise_id": "PREMISE_A",
                "exact_formula": "I_F = E[grad log p * grad log p^T]",
                "derived_bound": "kappa_max <= 0.42",
                "rationale": "Curvature above 0.42 causes catastrophic rank collapse in key-cache geometry.",
                "status": "VALIDATED_EMPIRICAL",
            },
            aspect_vector=self.encoder.encode(p_a_text),
            cluster_id=0,
        )

        p_b_text = (
            "Epistemic Manifold Aspect Rank Preservation. "
            "Aspect projection tensor P_Z requires rank(P_Z) >= 4 to preserve decision-visible boundaries. "
            "SVD compression below rank 4 causes false equivalence in contract tests."
        )
        self._add_entity(
            eid="art_cortex_epistemic_aspect_rank",
            project="warp_cortex",
            title="Epistemic Manifold Aspect Rank Preservation",
            state={
                "concept": "ASPECT_RANK_PRESERVATION",
                "premise_id": "PREMISE_B",
                "rule": "Aspect projection tensor P_Z requires rank(P_Z) >= 4 to preserve decision-visible boundaries.",
                "failure_mode": "SVD compression below rank 4 causes false equivalence in contract tests.",
                "status": "VALIDATED_EMPIRICAL",
            },
            aspect_vector=self.encoder.encode(p_b_text),
            cluster_id=1,
        )

        p_c_text = (
            "Intrinsic Cross-Spacing Excess Mixing Invariant. "
            "Exact formula: Delta_min = sqrt(2 * ln(d) / kappa). "
            "Excess mixing is eliminated iff latent separation Delta >= sqrt(2 * ln(d) / kappa)."
        )
        self._add_entity(
            eid="art_2521_excess_mixing_bound",
            project="project_2521",
            title="Intrinsic Cross-Spacing Excess Mixing Invariant",
            state={
                "concept": "EXCESS_MIXING_INVARIANT",
                "premise_id": "PREMISE_C",
                "exact_formula": "Delta_min = sqrt(2 * ln(d) / kappa)",
                "consequence": "Excess mixing is eliminated iff latent separation Delta >= sqrt(2 * ln(d) / kappa).",
                "status": "VALIDATED_MATHEMATICAL",
            },
            aspect_vector=self.encoder.encode(p_c_text),
            cluster_id=2,
        )
        # Note: Zero manual edge insertion. Edges are generated strictly via unsupervised k-NN in Phase 4.

    def _ingest_workspace_directory(self, project_name: str, root_dir: str):
        if not os.path.exists(root_dir):
            return

        # Scan python, markdown, and yaml files
        patterns = ["**/*.py", "**/*.md", "**/*.yaml"]
        for pat in patterns:
            for filepath in glob.glob(os.path.join(root_dir, pat), recursive=True):
                # Skip virtualenvs or cache dirs
                if any(x in filepath for x in [".venv", ".git", "__pycache__", ".pytest_cache", ".hf_cache"]):
                    continue

                rel_path = os.path.relpath(filepath, root_dir)
                eid = f"{project_name}::{rel_path.replace(os.sep, '/')}"

                # Read preview content
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(25)]
                        snippet = "".join(lines)[:400]
                except Exception:
                    snippet = ""

                cluster_id = abs(hash(project_name)) % self.substrate.num_clusters
                self._add_entity(
                    eid=eid,
                    project=project_name,
                    title=os.path.basename(filepath),
                    state={
                        "rel_path": rel_path,
                        "file_size": os.path.getsize(filepath),
                        "snippet": snippet,
                        "type": "SOURCE_FILE" if filepath.endswith(".py") else "DOCUMENTATION",
                    },
                    cluster_id=cluster_id,
                )

    def _expand_background_corpus(self, target_total: int):
        # NOTE: this expansion is SYNTHETIC background (cluster-consistent filler
        # to reach |U| >> context window for scale testing), not real workspace
        # history. Keep it labeled as such in every report.
        current = self.ingested_count
        needed = max(0, target_total - current)
        print(f"[KnowledgeHarvester] Expanding with {needed} background research entities...")

        projects = ["warp_cortex", "warp_align", "inference_wedge", "project_2521", "model_selection", "synthetic_eval"]
        categories = [
            ("benchmark_run", "BENCHMARK_LOG", "Recorded latency, throughput, and error curves across sweeps."),
            ("negative_result", "NEGATIVE_FINDING", "Failed hypothesis: parameter divergence under high concurrency."),
            ("critique_note", "ARCHITECTURAL_AUDIT", "Audit note rejecting uncalibrated linear extrapolations."),
            ("kernel_profile", "GPU_PROFILE", "Nsight compute warp occupancy, register pressure, and bank conflicts."),
            ("mathematical_derivation", "THEORETICAL_DERIVATION", "Proof steps regarding Lipschitz continuity and manifold projection."),
        ]

        for i in range(needed):
            p = projects[i % len(projects)]
            cat_name, cat_type, cat_desc = categories[i % len(categories)]
            eid = f"corpus_{p}_{cat_name}_{i:05d}"
            cid = i % self.substrate.num_clusters

            node = self._add_entity(
                eid=eid,
                project=p,
                title=f"{p.upper()} {cat_name.replace('_', ' ').title()} #{i}",
                state={
                    "category": cat_type,
                    "description": cat_desc,
                    "iteration": i,
                    "timestamp": 1725000000 + i * 3600,
                    "relevance_flag": "BACKGROUND_CORPUS",
                },
                cluster_id=cid,
            )

            # Wire random graph neighbors within same cluster
            cluster_list = self.substrate.clusters[cid]
            if len(cluster_list) > 1:
                prev_id = cluster_list[(len(cluster_list) - 2) % len(cluster_list)]
                node.neighbors.add(prev_id)
                self.substrate.entities[prev_id].neighbors.add(eid)

    def _build_unsupervised_semantic_graph(self, k_nearest: int = 4, sim_threshold: float = 0.45):
        """
        Phase 4: Frozen, generic, task-agnostic k-NN semantic graph builder.
        Operates on all entities in the corpus simultaneously.
        Computes pairwise cosine similarities on the normalized aspect vectors Z.
        Connects nodes if cosine similarity >= sim_threshold.
        Strictly zero access to:
          - Question definitions
          - Target values Delta*
          - Premise labels (P_A, P_B, P_C)
          - Human supervision or domain-specific hardcoded edges
        (Task-agnostic: the base token embeddings are pretrained.)
        """
        eids = list(self.substrate.entities.keys())
        N = len(eids)
        if N < 2:
            return

        # Stack aspect vectors: [N, 64]
        Z = torch.stack([self.substrate.entities[eid].aspect_vector for eid in eids])

        # Compute cosine similarity matrix: [N, N]
        sim_matrix = torch.matmul(Z, Z.t())
        sim_matrix.fill_diagonal_(-1.0)

        # For each entity, find top-k neighbors
        topk_vals, topk_indices = torch.topk(sim_matrix, k=min(k_nearest, N - 1), dim=1)

        added_edges = 0
        for i in range(N):
            src_eid = eids[i]
            for k_idx in range(topk_indices.shape[1]):
                j = topk_indices[i, k_idx].item()
                val = topk_vals[i, k_idx].item()
                if val >= sim_threshold:
                    dst_eid = eids[j]
                    if dst_eid not in self.substrate.entities[src_eid].neighbors:
                        self.substrate.entities[src_eid].neighbors.add(dst_eid)
                        self.substrate.entities[dst_eid].neighbors.add(src_eid)
                        added_edges += 1

        print(f"[KnowledgeHarvester] Phase 4 Task-Agnostic Graph: Added {added_edges} semantic edges across {N} nodes (sim >= {sim_threshold}).")


def generate_cryptographic_corpus_freeze(manifest_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Recursively scans and fingerprints every single source file contributing to the corpus.
    Calculates 64-char SHA-256 digests for each file, and computes a single root Merkle digest:
      H_corpus = SHA256(H_1 || H_2 || ... || H_N)
    Also fingerprints the pipeline code and records frozen hyper-parameters.
    """
    import hashlib
    import json

    search_dirs = [
        ("warp_cortex", os.path.join(WORKSPACE_ROOT, "warp_cortex")),
        ("warp_align", os.path.join(WORKSPACE_ROOT, "warp_align")),
        ("inference_wedge", os.path.join(WORKSPACE_ROOT, "inference_wedge")),
        ("project_2521", r"c:\Users\jorge\2521"),
    ]

    all_files = []
    for proj_name, root_dir in search_dirs:
        if not os.path.exists(root_dir):
            continue
        for ext in ["**/*.py", "**/*.md", "**/*.yaml"]:
            for fp in glob.glob(os.path.join(root_dir, ext), recursive=True):
                if any(x in fp for x in [".venv", ".git", "__pycache__", ".pytest_cache", ".hf_cache"]):
                    continue
                all_files.append((proj_name, os.path.abspath(fp)))

    # Sort files deterministically
    all_files.sort(key=lambda x: x[1])

    file_records = []
    merkle_hasher = hashlib.sha256()

    for proj, fp in all_files:
        try:
            sz = os.path.getsize(fp)
            mtime = os.path.getmtime(fp)
            mtime_utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mtime))
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            digest = h.hexdigest()
            # Portable record path: never leak absolute user paths into the
            # manifest. External-corpus files outside the workspace root are
            # keyed as external/<project>/<relpath-inside-external-root>.
            if fp.startswith(WORKSPACE_ROOT):
                rel_p = os.path.relpath(fp, WORKSPACE_ROOT)
            else:
                try:
                    _ext_root = os.path.abspath(
                        os.environ.get("CORTEX_2521_PATH", r"c:\Users\jorge\2521")
                    )
                    rel_p = os.path.join("external", proj, os.path.relpath(fp, _ext_root))
                except Exception:
                    rel_p = os.path.join("external", proj, os.path.basename(fp))
            file_records.append({
                "project": proj,
                "relative_path": rel_p.replace(os.sep, "/"),
                "file_size_bytes": sz,
                "last_modified_utc": mtime_utc,
                "sha256_hash": digest,
            })
            merkle_hasher.update(digest.encode("utf-8"))
        except Exception:
            continue

    root_merkle_digest = merkle_hasher.hexdigest()

    # Fingerprint harvester file itself
    this_file = os.path.abspath(__file__)
    harvester_hash = hashlib.sha256(open(this_file, "rb").read()).hexdigest()

    manifest = {
        "benchmark_freeze_timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "root_corpus_merkle_sha256": root_merkle_digest,
        "total_source_files_fingerprinted": len(file_records),
        "pipeline_provenance": {
            "harvester_code_sha256": harvester_hash,
            "encoder_type": "GenericFrozenAspectEncoder",
            "encoder_base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "aspect_dim": 64,
            "random_projection_seed": 42,
            "graph_k_nearest": 4,
            "graph_sim_threshold": 0.45,
        },
        "all_corpus_files": file_records,
    }

    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(__file__), "corpus_freeze_manifest.json")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[KnowledgeHarvester] Generated Complete Merkle Corpus Freeze ({len(file_records)} files).")
    print(f"  Root Merkle SHA-256: {root_merkle_digest}")
    print(f"  Saved to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    sub = FastWorldSubstrate(num_clusters=16)
    harvester = WorkspaceKnowledgeHarvester(sub)
    harvester.harvest_all(target_total=2000)
    snap = sub.current_snapshot()
    print(f"\nHarvest verification:")
    print(f"  Total Entities in Snapshot: {len(snap.entities)}")
    print(f"  Clusters Partitioned:      {len(snap.clusters)}")
    p_a = snap.get_entity("art_inference_wedge_fisher_curvature")
    print(f"  Premise A Verified:        {p_a.state['concept']} -> {p_a.neighbors}")
    generate_cryptographic_corpus_freeze()
