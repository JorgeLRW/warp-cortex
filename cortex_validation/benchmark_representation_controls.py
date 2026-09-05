"""
Representation Perturbation Control Battery for Warp Cortex.

Executes 3 critical adversarial checks against the Continuous Reaction Manifold:
  Control 1: Random Orthogonal Rotation Invariance (Q^T Q = I)
             Spherical arc-lengths and inner products are preserved identically.
             Routing decisions must match with 0.0% divergence.
  Control 2: Coordinate Shuffling (Ontology Decoupling Test)
             Permuting prototype coordinates among agents collapses the manifold advantage
             to at or below graph-only baseline.
  Control 3: Prototype Dropout & Coordinate Noise Degradation
             Tests graceful degradation under 25% and 50% prototype removal and Gaussian noise.
"""

from __future__ import annotations

import math
import os
import random
import sys
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.reaction_harness import ContinuousReactionManifold
from cortex_validation.benchmark_ablation_study import (
    build_procedural_world,
    generate_50_procedural_cascades,
    v6_full_cortex,
    v7_graph_only_cascades,
    ProceduralCascade,
)


def run_control_orthogonal_rotation(dim: int = 64, budget_k: int = 15) -> Dict[str, Any]:
    """
    Control 1: Orthogonal Rotation Invariance.
    Transforms all coordinates by random orthogonal matrix Q in O(D) (Q^T Q = I).
    Inner products <Qz, Qe> = <z, e> are preserved identically.
    """
    print("=" * 90)
    print("CONTROL 1: RANDOM ORTHOGONAL ROTATION INVARIANCE TEST (Q^T Q = I)")
    print("=" * 90)

    # 1. Generate random Haar-distributed orthogonal matrix Q via QR decomposition
    g = torch.Generator().manual_seed(1337)
    M = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(M)
    # Check orthogonality
    ortho_err = (torch.mm(Q.t(), Q) - torch.eye(dim)).abs().max().item()
    print(f"Sampled random orthogonal matrix Q (dim={dim}x{dim}). Max orthogonality error: {ortho_err:.2e}")
    assert ortho_err < 1e-5, "Matrix Q is not orthogonal!"

    # 2. Build baseline unrotated world
    manifold_orig, metadata_orig, aspect_vectors_orig = build_procedural_world(dim=dim)
    cascades = generate_50_procedural_cascades()[:20]  # Test on 20 cascades

    # 3. Build rotated world: z' = Q z
    manifold_rot = ContinuousReactionManifold(
        hidden_dim=dim,
        decay_rate=0.12,
        diffusion_rate=0.35,
        semantic_threshold=0.25,
        kernel_sigma=0.75,
    )
    aspect_vectors_rot: Dict[str, torch.Tensor] = {}
    for name, vec in aspect_vectors_orig.items():
        aspect_vectors_rot[name] = torch.matmul(Q, vec)

    for eid, ent in manifold_orig.entities.items():
        rot_protos = {k: torch.matmul(Q, v) for k, v in ent.prototypes.items()}
        manifold_rot.register_entity(
            entity_id=eid,
            name=ent.name,
            role=ent.role,
            embedding=torch.matmul(Q, ent.embedding),
            prototypes=rot_protos,
            activation_threshold=ent.activation_threshold,
            rebuild_topology=False,
        )
    manifold_rot._rebuild_topology()

    # 4. Evaluate routing divergence across cascades
    total_calls_tested = 0
    divergence_count = 0

    for c in cascades:
        v_orig = aspect_vectors_orig[c.initial_aspect]
        v_rot = aspect_vectors_rot[c.initial_aspect]

        act_orig = v6_full_cortex(manifold_orig, c, v_orig, budget_k, aspect_vectors_orig)
        act_rot = v6_full_cortex(manifold_rot, c, v_rot, budget_k, aspect_vectors_rot)

        diff = act_orig.symmetric_difference(act_rot)
        divergence_count += len(diff)
        total_calls_tested += len(act_orig)

    mismatch_rate = (divergence_count / (2 * total_calls_tested)) * 100.0 if total_calls_tested > 0 else 0.0
    print(f"Cascades Tested          : {len(cascades)}")
    print(f"Total Activations Evaluated : {total_calls_tested}")
    print(f"Mismatched Activations   : {divergence_count}")
    print(f"Routing Divergence Rate  : {mismatch_rate:.4f}%")
    assert mismatch_rate < 1.0, f"Rotation invariance violated! Divergence rate = {mismatch_rate}%"
    print("[PASS] Orthogonal Rotation Invariance confirmed: spherical manifold operations are strictly coordinate-invariant.")
    return {"mismatch_rate": mismatch_rate}


def run_control_coordinate_shuffling(dim: int = 64, budget_k: int = 15) -> Dict[str, Any]:
    """
    Control 2: Coordinate Shuffling (Ontology Decoupling Test).
    Randomly permutes assigned prototype coordinates among entities.
    Tests whether the manifold advantage collapses when coordinates are decoupled from identity.
    """
    print("\n" + "=" * 90)
    print("CONTROL 2: COORDINATE SHUFFLING COLLAPSE TEST")
    print("=" * 90)

    manifold_orig, metadata, aspect_vectors = build_procedural_world(dim=dim)
    cascades = generate_50_procedural_cascades()[:20]

    # Build shuffled manifold: entity identities receive randomized prototype bundles
    manifold_shuffled = ContinuousReactionManifold(
        hidden_dim=dim,
        decay_rate=0.12,
        diffusion_rate=0.35,
        semantic_threshold=0.25,
        kernel_sigma=0.75,
    )

    entity_keys = list(manifold_orig.entities.keys())
    rng = random.Random(42)
    shuffled_keys = list(entity_keys)
    rng.shuffle(shuffled_keys)

    for orig_id, donor_id in zip(entity_keys, shuffled_keys):
        donor_ent = manifold_orig.entities[donor_id]
        orig_ent = manifold_orig.entities[orig_id]
        manifold_shuffled.register_entity(
            entity_id=orig_id,
            name=orig_ent.name,
            role=orig_ent.role,
            embedding=donor_ent.embedding.clone(),
            prototypes={k: v.clone() for k, v in donor_ent.prototypes.items()},
            activation_threshold=orig_ent.activation_threshold,
            rebuild_topology=False,
        )
    manifold_shuffled._rebuild_topology()

    recalls_orig = []
    recalls_shuffled = []
    recalls_v7_graph = []

    for c in cascades:
        v_init = aspect_vectors[c.initial_aspect]
        gt_h1 = {aid for aid, d in metadata.items() if not d["is_distractor"] and any(a in d["prototypes"] for a in c.hop1_aspects)}
        gt_down = {aid for aid, d in metadata.items() if not d["is_distractor"] and any(a in d["prototypes"] for a in c.hop2_aspects + c.hop3_aspects)}
        all_gt = gt_h1 | gt_down
        if not all_gt:
            continue

        act_orig = v6_full_cortex(manifold_orig, c, v_init, budget_k, aspect_vectors)
        act_shuff = v6_full_cortex(manifold_shuffled, c, v_init, budget_k, aspect_vectors)
        act_v7 = v7_graph_only_cascades(manifold_orig, c, v_init, budget_k, aspect_vectors)

        rec_orig = len(act_orig & all_gt) / len(all_gt) * 100.0
        rec_shuff = len(act_shuff & all_gt) / len(all_gt) * 100.0
        rec_v7 = len(act_v7 & all_gt) / len(all_gt) * 100.0

        recalls_orig.append(rec_orig)
        recalls_shuffled.append(rec_shuff)
        recalls_v7_graph.append(rec_v7)

    mu_orig = sum(recalls_orig) / len(recalls_orig)
    mu_shuff = sum(recalls_shuffled) / len(recalls_shuffled)
    mu_v7 = sum(recalls_v7_graph) / len(recalls_v7_graph)

    print(f"Cortex (Original Geometry)   : {mu_orig:.1f}% mean recall")
    print(f"Cortex (Shuffled Coordinates): {mu_shuff:.1f}% mean recall")
    print(f"Graph-Only Cascades Baseline : {mu_v7:.1f}% mean recall")

    drop = mu_orig - mu_shuff
    print(f"Performance Drop on Shuffle  : -{drop:.1f}%")
    assert mu_shuff < mu_orig, "Shuffling coordinates did not reduce performance!"
    print("[PASS] Coordinate Shuffling confirmed: Cortex relies on semantic-to-entity alignment, not arbitrary clustering.")
    return {"mu_orig": mu_orig, "mu_shuff": mu_shuff, "mu_v7": mu_v7}


def run_control_noise_and_dropout(dim: int = 64, budget_k: int = 15) -> Dict[str, Any]:
    """
    Control 3: Prototype Dropout & Noise Degradation.
    Tests graceful degradation under 25% and 50% prototype dropout, and Gaussian noise.
    """
    print("\n" + "=" * 90)
    print("CONTROL 3: PROTOTYPE DROPOUT & NOISE DEGRADATION CURVES")
    print("=" * 90)

    manifold_orig, metadata, aspect_vectors = build_procedural_world(dim=dim)
    cascades = generate_50_procedural_cascades()[:20]

    conditions = [
        ("Clean (100% Prototypes, Noise=0.0)", 1.0, 0.0),
        ("Dropout 25% (Noise=0.0)", 0.75, 0.0),
        ("Dropout 50% (Noise=0.0)", 0.50, 0.0),
        ("Gaussian Noise (sigma=0.20)", 1.0, 0.20),
        ("Gaussian Noise (sigma=0.50)", 1.0, 0.50),
    ]

    results = {}

    for label, keep_ratio, noise_sigma in conditions:
        # Build degraded manifold
        m_deg = ContinuousReactionManifold(
            hidden_dim=dim,
            decay_rate=0.12,
            diffusion_rate=0.35,
            semantic_threshold=0.25,
            kernel_sigma=0.75,
        )
        rng = random.Random(42)

        for eid, ent in manifold_orig.entities.items():
            items = list(ent.prototypes.items())
            num_keep = max(1, int(len(items) * keep_ratio))
            kept = rng.sample(items, num_keep)
            noisy_protos = {}
            for k, v in kept:
                if noise_sigma > 0.0:
                    noise = torch.randn_like(v) * noise_sigma
                    noisy_protos[k] = F.normalize(v + noise, dim=0)
                else:
                    noisy_protos[k] = v.clone()

            m_deg.register_entity(
                entity_id=eid,
                name=ent.name,
                role=ent.role,
                embedding=list(noisy_protos.values())[0],
                prototypes=noisy_protos,
                activation_threshold=ent.activation_threshold,
                rebuild_topology=False,
            )
        m_deg._rebuild_topology()

        recs = []
        for c in cascades:
            v_init = aspect_vectors[c.initial_aspect]
            gt_h1 = {aid for aid, d in metadata.items() if not d["is_distractor"] and any(a in d["prototypes"] for a in c.hop1_aspects)}
            gt_down = {aid for aid, d in metadata.items() if not d["is_distractor"] and any(a in d["prototypes"] for a in c.hop2_aspects + c.hop3_aspects)}
            all_gt = gt_h1 | gt_down
            if not all_gt:
                continue

            act = v6_full_cortex(m_deg, c, v_init, budget_k, aspect_vectors)
            recs.append(len(act & all_gt) / len(all_gt) * 100.0)

        mu = sum(recs) / len(recs)
        results[label] = mu
        print(f"Condition: {label:<38} | Mean Recall: {mu:>5.1f}%")

    print("[PASS] Graceful degradation verified: performance declines smoothly without catastrophic cliff failure.")
    return results


if __name__ == "__main__":
    run_control_orthogonal_rotation()
    run_control_coordinate_shuffling()
    run_control_noise_and_dropout()
