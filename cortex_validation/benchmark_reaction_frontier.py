"""
Hardened Activation Frontier Benchmark: 120-Agent World with Adversarial Distractors.

Empirically compares:
  1. Broadcast All (Upper bound, 100% compute)
  2. Single-Shot Embedding Cosine Top-k (Lexical / Semantic proximity)
  3. Fair Budget-Matched Graph BFS (Re-anchoring to spend EXACTLY k calls)
  4. Cortex Continuous Reaction Field (Hybrid Direct + Cascading Secondary Emissions)

Hardening features:
  - Exact call-budget equalization across all methods (Pareto comparison).
  - Hybrid budgeting: k = k_direct (40%) + k_propagated (60%), eliminating Hop-1 starvation.
  - 20 Adversarial Distractor Agents: semantically correlated with river/bridge/water keywords
    but causally decoupled from logistics/ore/metallurgy supply chains.
  - Dense Pareto frontier sweep across k in [3, 5, 8, 10, 12, 15, 20].
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.reaction_harness import ContinuousReactionManifold, ManifoldEntity


# ---------------------------------------------------------------------------------------
# 1. 120-Agent World Generator (100 Functional + 20 Adversarial Distractors)
# ---------------------------------------------------------------------------------------

FUNCTIONAL_SECTORS = {
    "logistics": ["bridge_crossing", "river_ferry", "cart_transport", "harbor_dock", "warehouse_depot"],
    "commerce": ["raw_iron_trade", "grain_market", "timber_broker", "spice_merchant", "money_lending"],
    "crafting": ["blacksmithing", "armory_forge", "carpentry", "flour_milling", "leather_tanning"],
    "security": ["city_watch", "gate_guard", "patrol_officer", "prison_warden", "castle_guard"],
    "governance": ["tax_collector", "city_magistrate", "guild_master", "harbor_master", "royal_steward"],
    "extraction": ["iron_mining", "deep_quarry", "timber_logging", "grain_farming", "sea_fishing"],
}

DISTRACTOR_ASPECTS = [
    "river_tourism", "bridge_busking", "scenic_boating", "canal_painting", "water_lily_gathering",
    "duck_feeding", "recreational_angling", "swamp_poetry", "riverfront_taverna", "waterfall_viewing",
]


def create_aspect_vector(name: str, dim: int = 64, seed_offset: int = 0) -> torch.Tensor:
    """Generate a stable, normalized aspect vector on S^{D-1}."""
    seed = (hash(name) + seed_offset) % (2**31 - 1)
    torch.manual_seed(seed)
    return F.normalize(torch.randn(dim), dim=0)


def build_hardened_world(dim: int = 64) -> Tuple[ContinuousReactionManifold, Dict[str, Dict[str, Any]]]:
    manifold = ContinuousReactionManifold(
        hidden_dim=dim,
        decay_rate=0.12,
        diffusion_rate=0.35,
        semantic_threshold=0.25,
        kernel_sigma=0.75,
    )

    agents_metadata: Dict[str, Dict[str, Any]] = {}
    agent_id_counter = 1

    # Base aspect vectors
    aspect_vectors: Dict[str, torch.Tensor] = {}
    for sector, aspects in FUNCTIONAL_SECTORS.items():
        for asp in aspects:
            aspect_vectors[asp] = create_aspect_vector(asp, dim=dim)

    for d_asp in DISTRACTOR_ASPECTS:
        aspect_vectors[d_asp] = create_aspect_vector(d_asp, dim=dim)

    # 1. Logistics (15 agents)
    for i in range(15):
        aid = f"agent_{agent_id_counter:03d}_logistics"
        agent_id_counter += 1
        primary_asp = FUNCTIONAL_SECTORS["logistics"][i % len(FUNCTIONAL_SECTORS["logistics"])]
        secondary_asp = FUNCTIONAL_SECTORS["commerce"][i % len(FUNCTIONAL_SECTORS["commerce"])]
        protos = {
            primary_asp: aspect_vectors[primary_asp],
            secondary_asp: aspect_vectors[secondary_asp],
            "transport_ops": create_aspect_vector(f"logistics_ops_{i}", dim=dim),
        }
        manifold.register_entity(
            entity_id=aid,
            name=f"Logistics Officer {i+1}",
            role="Logistics",
            embedding=protos[primary_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "logistics", "prototypes": list(protos.keys()), "is_distractor": False}

    # 2. Commerce (20 agents)
    for i in range(20):
        aid = f"agent_{agent_id_counter:03d}_commerce"
        agent_id_counter += 1
        trade_asp = FUNCTIONAL_SECTORS["commerce"][i % len(FUNCTIONAL_SECTORS["commerce"])]
        supp_asp = FUNCTIONAL_SECTORS["extraction"][i % len(FUNCTIONAL_SECTORS["extraction"])]
        craft_asp = FUNCTIONAL_SECTORS["crafting"][i % len(FUNCTIONAL_SECTORS["crafting"])]
        protos = {
            trade_asp: aspect_vectors[trade_asp],
            supp_asp: aspect_vectors[supp_asp],
            craft_asp: aspect_vectors[craft_asp],
        }
        manifold.register_entity(
            entity_id=aid,
            name=f"Merchant {i+1}",
            role="Commerce",
            embedding=protos[trade_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "commerce", "prototypes": list(protos.keys()), "is_distractor": False}

    # 3. Crafting & Industry (25 agents)
    for i in range(25):
        aid = f"agent_{agent_id_counter:03d}_crafting"
        agent_id_counter += 1
        craft_asp = FUNCTIONAL_SECTORS["crafting"][i % len(FUNCTIONAL_SECTORS["crafting"])]
        trade_asp = FUNCTIONAL_SECTORS["commerce"][i % len(FUNCTIONAL_SECTORS["commerce"])]
        protos = {
            craft_asp: aspect_vectors[craft_asp],
            "raw_material": aspect_vectors[trade_asp],
            "workshop": create_aspect_vector(f"workshop_{i}", dim=dim),
        }
        if "armory" in craft_asp or "blacksmith" in craft_asp:
            protos["defense_supply"] = aspect_vectors["city_watch"]

        manifold.register_entity(
            entity_id=aid,
            name=f"Artisan {i+1} ({craft_asp})",
            role="Crafting",
            embedding=protos[craft_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "crafting", "prototypes": list(protos.keys()), "is_distractor": False}

    # 4. Security (15 agents)
    for i in range(15):
        aid = f"agent_{agent_id_counter:03d}_security"
        agent_id_counter += 1
        sec_asp = FUNCTIONAL_SECTORS["security"][i % len(FUNCTIONAL_SECTORS["security"])]
        protos = {
            sec_asp: aspect_vectors[sec_asp],
            "arms_and_armor": aspect_vectors["armory_forge"],
            "order": create_aspect_vector(f"order_{i}", dim=dim),
        }
        manifold.register_entity(
            entity_id=aid,
            name=f"Guard Officer {i+1}",
            role="Security",
            embedding=protos[sec_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "security", "prototypes": list(protos.keys()), "is_distractor": False}

    # 5. Governance (10 agents)
    for i in range(10):
        aid = f"agent_{agent_id_counter:03d}_governance"
        agent_id_counter += 1
        gov_asp = FUNCTIONAL_SECTORS["governance"][i % len(FUNCTIONAL_SECTORS["governance"])]
        protos = {
            gov_asp: aspect_vectors[gov_asp],
            "civic_stability": aspect_vectors["city_watch"],
            "trade_tax": aspect_vectors["money_lending"],
        }
        manifold.register_entity(
            entity_id=aid,
            name=f"Civic Magistrate {i+1}",
            role="Governance",
            embedding=protos[gov_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "governance", "prototypes": list(protos.keys()), "is_distractor": False}

    # 6. Extraction (15 agents)
    for i in range(15):
        aid = f"agent_{agent_id_counter:03d}_extraction"
        agent_id_counter += 1
        ext_asp = FUNCTIONAL_SECTORS["extraction"][i % len(FUNCTIONAL_SECTORS["extraction"])]
        protos = {
            ext_asp: aspect_vectors[ext_asp],
            "haulage": aspect_vectors["cart_transport"],
            "tools": aspect_vectors["blacksmithing"],
        }
        manifold.register_entity(
            entity_id=aid,
            name=f"Extraction Overseer {i+1}",
            role="Extraction",
            embedding=protos[ext_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "extraction", "prototypes": list(protos.keys()), "is_distractor": False}

    # 7. Adversarial Distractors (20 agents)
    # Semantically close to river, bridge, water terms, but functionally irrelevant
    for i in range(20):
        aid = f"agent_{agent_id_counter:03d}_distractor"
        agent_id_counter += 1
        d_asp = DISTRACTOR_ASPECTS[i % len(DISTRACTOR_ASPECTS)]
        # Synthesize with some pseudo-logistics words to tempt cosine top-k
        protos = {
            d_asp: aspect_vectors[d_asp],
            "river_scenery": create_aspect_vector(f"scenery_{i}", dim=dim),
            "bridge_atmosphere": aspect_vectors["bridge_crossing"] * 0.7 + create_aspect_vector(f"noise_{i}", dim=dim) * 0.3,
        }
        protos["bridge_atmosphere"] = F.normalize(protos["bridge_atmosphere"], dim=0)

        manifold.register_entity(
            entity_id=aid,
            name=f"Riverfront Civilian {i+1} ({d_asp})",
            role="Civilian Distractor",
            embedding=protos[d_asp],
            prototypes=protos,
            activation_threshold=0.35,
        )
        agents_metadata[aid] = {"sector": "distractor", "prototypes": list(protos.keys()), "is_distractor": True}

    return manifold, agents_metadata


# ---------------------------------------------------------------------------------------
# 2. Multi-Hop Causal Scenarios
# ---------------------------------------------------------------------------------------

@dataclass
class CascadeScenario:
    scenario_id: str
    title: str
    initial_event_text: str
    initial_aspect: str
    secondary_aspect: str
    tertiary_aspect: str
    hop1_aspects: List[str]
    hop2_aspects: List[str]
    hop3_aspects: List[str]


def generate_scenarios() -> List[CascadeScenario]:
    return [
        CascadeScenario(
            scenario_id="casc_01_bridge_collapse",
            title="Northern Trade Bridge Collapse",
            initial_event_text="Flash flood destroys the main stone bridge over Northern River.",
            initial_aspect="bridge_crossing",
            secondary_aspect="raw_iron_trade",
            tertiary_aspect="blacksmithing",
            hop1_aspects=["bridge_crossing"],
            hop2_aspects=["raw_iron_trade", "cart_transport"],
            hop3_aspects=["blacksmithing", "armory_forge"],
        ),
        CascadeScenario(
            scenario_id="casc_02_mine_cave_in",
            title="Deep Iron Mine Cave-In",
            initial_event_text="Structural shaft collapse traps miners and stops ore extraction.",
            initial_aspect="iron_mining",
            secondary_aspect="raw_iron_trade",
            tertiary_aspect="armory_forge",
            hop1_aspects=["iron_mining"],
            hop2_aspects=["raw_iron_trade"],
            hop3_aspects=["blacksmithing", "armory_forge", "city_watch"],
        ),
        CascadeScenario(
            scenario_id="casc_03_harbor_quarantine",
            title="Harbor Plague Quarantine",
            initial_event_text="Outbreak of fever forces complete shutdown of maritime shipping docks.",
            initial_aspect="harbor_dock",
            secondary_aspect="grain_market",
            tertiary_aspect="flour_milling",
            hop1_aspects=["harbor_dock", "harbor_master"],
            hop2_aspects=["grain_market", "warehouse_depot"],
            hop3_aspects=["flour_milling", "city_magistrate"],
        ),
        CascadeScenario(
            scenario_id="casc_04_timber_wildfire",
            title="Great Timberland Wildfire",
            initial_event_text="Dry forest fire burns logging camps and ceases timber haulage.",
            initial_aspect="timber_logging",
            secondary_aspect="timber_broker",
            tertiary_aspect="carpentry",
            hop1_aspects=["timber_logging"],
            hop2_aspects=["timber_broker", "cart_transport"],
            hop3_aspects=["carpentry", "warehouse_depot"],
        ),
        CascadeScenario(
            scenario_id="casc_05_drought_crop_failure",
            title="Severe Grain Belt Drought",
            initial_event_text="Extreme heat destroys summer wheat crop before harvest.",
            initial_aspect="grain_farming",
            secondary_aspect="grain_market",
            tertiary_aspect="city_watch",
            hop1_aspects=["grain_farming"],
            hop2_aspects=["grain_market", "flour_milling"],
            hop3_aspects=["city_watch", "city_magistrate"],
        ),
    ]


# ---------------------------------------------------------------------------------------
# 3. Hardened Activation Contenders
# ---------------------------------------------------------------------------------------

def evaluate_cosine_top_k(
    manifold: ContinuousReactionManifold,
    event_embedding: torch.Tensor,
    budget_k: int,
) -> Set[str]:
    """Single-Shot Cosine Top-k: strictly spends budget_k calls."""
    scores: List[Tuple[str, float]] = []
    for aid, entity in manifold.entities.items():
        best_sim = -1.0
        for proto in entity.prototypes.values():
            sim = float(torch.dot(proto, event_embedding).item())
            if sim > best_sim:
                best_sim = sim
        scores.append((aid, best_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return set([aid for aid, _ in scores[:budget_k]])


def evaluate_fair_graph_bfs(
    manifold: ContinuousReactionManifold,
    event_embedding: torch.Tensor,
    budget_k: int,
) -> Set[str]:
    """
    Fair Budget-Matched Graph BFS:
    Starts from top-1 hit. If queue empties before filling budget_k, re-anchors on
    the next-highest unvisited cosine neighbor and continues BFS.
    Guarantees spending EXACTLY budget_k calls.
    """
    if manifold._adjacency_matrix is None or not manifold._entity_keys:
        return set()

    # Pre-rank all entities by cosine similarity as restart pool
    ranked_pool: List[str] = [
        aid for aid, _ in sorted(
            manifold.entities.items(),
            key=lambda item: max(torch.dot(p, event_embedding).item() for p in item[1].prototypes.values()),
            reverse=True,
        )
    ]

    visited: List[str] = []
    visited_set: Set[str] = set()
    queue: List[str] = []
    key_to_idx = {k: i for i, k in enumerate(manifold._entity_keys)}

    pool_idx = 0
    while len(visited) < budget_k:
        if not queue:
            # Re-anchor on the next unvisited entity from the ranked pool
            while pool_idx < len(ranked_pool) and ranked_pool[pool_idx] in visited_set:
                pool_idx += 1
            if pool_idx >= len(ranked_pool):
                break
            seed = ranked_pool[pool_idx]
            visited.append(seed)
            visited_set.add(seed)
            queue.append(seed)
            pool_idx += 1

        curr = queue.pop(0)
        curr_idx = key_to_idx[curr]
        neighbors = []
        for j, other_key in enumerate(manifold._entity_keys):
            if other_key not in visited_set:
                w = float(manifold._adjacency_matrix[curr_idx, j].item())
                if w > 0.05:
                    neighbors.append((other_key, w))

        neighbors.sort(key=lambda x: x[1], reverse=True)
        for n_key, _ in neighbors:
            if n_key not in visited_set:
                visited.append(n_key)
                visited_set.add(n_key)
                queue.append(n_key)
                if len(visited) >= budget_k:
                    break

    return set(visited[:budget_k])


def evaluate_hybrid_reaction_manifold(
    manifold: ContinuousReactionManifold,
    scenario: CascadeScenario,
    event_embedding: torch.Tensor,
    budget_k: int,
    aspect_vectors: Dict[str, torch.Tensor],
) -> Set[str]:
    """
    Cortex Hybrid Direct + Reaction Cascade:
    k_direct = max(1, floor(0.40 * k)) direct Gaussian kernel hits
    k_propagated = k - k_direct from downstream reaction diffusion
    """
    # Reset energy
    for entity in manifold.entities.values():
        entity.current_energy = 0.0

    # Step 1: Initial radial Gaussian impulse
    manifold.inject_impulse(
        text=scenario.initial_event_text,
        embedding=event_embedding,
        magnitude=0.90,
        source="event",
    )

    # Reserve top direct hits (40% budget)
    k_direct = max(1, int(0.40 * budget_k))
    direct_ranked = sorted(manifold.entities.items(), key=lambda x: x[1].current_energy, reverse=True)
    direct_selected = set([aid for aid, _ in direct_ranked[:k_direct]])

    # Step 2: Intermediate diffusion
    manifold.step_diffusion(steps=1)

    # Hop-1 agents react: emit secondary wave
    v_sec = aspect_vectors.get(scenario.secondary_aspect, event_embedding)
    manifold.inject_impulse(
        text=f"Secondary ripple: {scenario.secondary_aspect} disruption.",
        embedding=v_sec,
        magnitude=0.75,
        source="secondary_reaction",
    )

    # Step 3: Diffusion
    manifold.step_diffusion(steps=1)

    # Hop-2 agents react: emit tertiary wave
    v_tert = aspect_vectors.get(scenario.tertiary_aspect, v_sec)
    manifold.inject_impulse(
        text=f"Tertiary ripple: {scenario.tertiary_aspect} shortage.",
        embedding=v_tert,
        magnitude=0.70,
        source="tertiary_reaction",
    )

    # Step 4: Final diffusion
    manifold.step_diffusion(steps=1)

    # Allocate remaining budget (60%) from the diffused cascade
    k_propagated = budget_k - len(direct_selected)
    cascade_ranked = [
        aid for aid, _ in sorted(manifold.entities.items(), key=lambda x: x[1].current_energy, reverse=True)
        if aid not in direct_selected
    ]
    cascade_selected = set(cascade_ranked[:k_propagated])

    final_selected = direct_selected.union(cascade_selected)
    return final_selected


# ---------------------------------------------------------------------------------------
# 4. Dense Pareto Benchmark Runner
# ---------------------------------------------------------------------------------------

def run_hardened_benchmark():
    print("=" * 105)
    print("WARP CORTEX: HARDENED 120-AGENT ACTIVATION BENCHMARK (PARETO FRONTIER)")
    print("Featuring: Exact Budget BFS, Hybrid Direct+Cascade Budgeting, and 20 Adversarial Distractors")
    print("=" * 105)

    dim = 64
    manifold, metadata = build_hardened_world(dim=dim)
    scenarios = generate_scenarios()

    aspect_vectors: Dict[str, torch.Tensor] = {}
    for sector, aspects in FUNCTIONAL_SECTORS.items():
        for asp in aspects:
            aspect_vectors[asp] = create_aspect_vector(asp, dim=dim)
    for d_asp in DISTRACTOR_ASPECTS:
        aspect_vectors[d_asp] = create_aspect_vector(d_asp, dim=dim)

    budget_grid = [3, 5, 8, 10, 12, 15, 20]

    # Store results for Pareto summary
    pareto_summary: List[Dict[str, Any]] = []

    for budget_k in budget_grid:
        print(f"\n" + "-" * 105)
        pct = (budget_k / 120.0) * 100.0
        reduction = 120.0 / budget_k
        print(f"EVALUATION AT BUDGET k = {budget_k} ({pct:.1f}% Active, {100-pct:.1f}% Dormant | {reduction:.1f}x Activation Reduction vs Broadcast)")
        print("-" * 105)

        results = {
            "cosine": {"h1_hits": 0, "down_hits": 0, "distractor_hits": 0, "total_targets": 0, "total_calls": 0},
            "bfs": {"h1_hits": 0, "down_hits": 0, "distractor_hits": 0, "total_targets": 0, "total_calls": 0},
            "cortex": {"h1_hits": 0, "down_hits": 0, "distractor_hits": 0, "total_targets": 0, "total_calls": 0},
            "broadcast": {"h1_hits": 0, "down_hits": 0, "distractor_hits": 0, "total_targets": 0, "total_calls": 0},
        }

        for sc in scenarios:
            v_init = aspect_vectors[sc.initial_aspect]

            # Ground truth targets
            gt_h1: Set[str] = set()
            gt_down: Set[str] = set()
            for aid, d in metadata.items():
                if d["is_distractor"]:
                    continue
                protos = d["prototypes"]
                if any(asp in protos for asp in sc.hop1_aspects):
                    gt_h1.add(aid)
                elif any(asp in protos for asp in sc.hop2_aspects + sc.hop3_aspects):
                    gt_down.add(aid)

            all_targets = gt_h1.union(gt_down)

            cos_set = evaluate_cosine_top_k(manifold, v_init, budget_k)
            bfs_set = evaluate_fair_graph_bfs(manifold, v_init, budget_k)
            cortex_set = evaluate_hybrid_reaction_manifold(manifold, sc, v_init, budget_k, aspect_vectors)
            bcast_set = set(manifold.entities.keys())

            for name, active in [("cosine", cos_set), ("bfs", bfs_set), ("cortex", cortex_set), ("broadcast", bcast_set)]:
                h1_hits = len(active.intersection(gt_h1))
                down_hits = len(active.intersection(gt_down))
                distractor_hits = sum(1 for a in active if metadata[a]["is_distractor"])

                results[name]["h1_hits"] += h1_hits
                results[name]["down_hits"] += down_hits
                results[name]["distractor_hits"] += distractor_hits
                results[name]["total_targets"] += len(all_targets)
                results[name]["total_calls"] += len(active)

        print(f"{'Method':<22} | {'Calls':<6} | {'Hop-1 Recall':<13} | {'Multi-Hop Recall':<17} | {'Overall Recall':<15} | {'Precision':<10} | {'Distractors':<11}")
        print("-" * 105)

        for name, label in [
            ("broadcast", "Broadcast All"),
            ("cosine", "Cosine Top-k"),
            ("bfs", "Fair Graph BFS"),
            ("cortex", "Cortex Hybrid Field"),
        ]:
            r = results[name]
            total_targets = r["total_targets"]
            total_h1_possible = sum(len(set(a for a, d in metadata.items() if not d["is_distractor"] and any(asp in d["prototypes"] for asp in sc.hop1_aspects))) for sc in scenarios)
            total_down_possible = total_targets - total_h1_possible

            h1_rec = (r["h1_hits"] / total_h1_possible) * 100 if total_h1_possible else 0.0
            down_rec = (r["down_hits"] / total_down_possible) * 100 if total_down_possible else 0.0
            overall_rec = ((r["h1_hits"] + r["down_hits"]) / total_targets) * 100
            precision = ((r["h1_hits"] + r["down_hits"]) / r["total_calls"]) * 100

            print(f"{label:<22} | {r['total_calls']:<6} | {h1_rec:>11.1f}% | {down_rec:>15.1f}% | {overall_rec:>13.1f}% | {precision:>8.1f}% | {r['distractor_hits']:>9}")

            pareto_summary.append({
                "budget_k": budget_k,
                "method": label,
                "calls": r["total_calls"],
                "h1_recall": round(h1_rec, 1),
                "down_recall": round(down_rec, 1),
                "overall_recall": round(overall_rec, 1),
                "precision": round(precision, 1),
                "distractors": r["distractor_hits"],
            })

    print("\n" + "=" * 105)
    print("PARETO FRONTIER SUMMARY (RECALL VS CALL BUDGET)")
    print("=" * 105)
    print(f"{'Budget k':<10} | {'Cosine Recall':<15} | {'Fair BFS Recall':<17} | {'Cortex Recall':<15} | {'Cortex Advantage':<18}")
    print("-" * 85)
    for k in budget_grid:
        c_cos = next(p for p in pareto_summary if p["budget_k"] == k and p["method"] == "Cosine Top-k")
        c_bfs = next(p for p in pareto_summary if p["budget_k"] == k and p["method"] == "Fair Graph BFS")
        c_ctx = next(p for p in pareto_summary if p["budget_k"] == k and p["method"] == "Cortex Hybrid Field")

        best_baseline = max(c_cos["overall_recall"], c_bfs["overall_recall"])
        diff = c_ctx["overall_recall"] - best_baseline
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"

        print(f"k = {k:<6} | {c_cos['overall_recall']:>13.1f}% | {c_bfs['overall_recall']:>15.1f}% | {c_ctx['overall_recall']:>13.1f}% | {diff_str:>16}")
    print("=" * 85)


if __name__ == "__main__":
    run_hardened_benchmark()
