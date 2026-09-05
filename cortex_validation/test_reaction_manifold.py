"""
Validation & Demonstration of Continuous Reaction Manifold Harness.

Simulates a dynamic game world where AI characters (Guard, Barkeep, Scholar)
exist as semantic coordinates on a continuous manifold.
Player actions inject continuous impulses; heat diffusion ripples across
the topology, activating only relevant agents.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cortex_core.reaction_harness import ContinuousReactionManifold


def make_synthetic_semantic_vector(category_seed: int, hidden_dim: int = 128) -> torch.Tensor:
    """Generate reproducible directional vectors on a sphere."""
    torch.manual_seed(category_seed)
    v = torch.randn(hidden_dim)
    return F.normalize(v, dim=0)


class TestContinuousReactionManifold(unittest.TestCase):
    def setUp(self):
        self.dim = 128
        self.manifold = ContinuousReactionManifold(
            hidden_dim=self.dim,
            decay_rate=0.10,
            diffusion_rate=0.40,
            semantic_threshold=0.20,
        )

        # Base semantic directions:
        # Direction 1: Violence, security, weapons (Guard)
        # Direction 2: Hospitality, tavern, tavern brawl (Barkeep - partially correlated with violence)
        # Direction 3: Ancient lore, magic, books (Scholar - orthogonal)
        v_combat = make_synthetic_semantic_vector(101, self.dim)
        v_social = make_synthetic_semantic_vector(202, self.dim)
        v_scholar = make_synthetic_semantic_vector(303, self.dim)

        # Barkeep is a blend of social hospitality with some tavern roughness
        v_barkeep = F.normalize(0.7 * v_social + 0.3 * v_combat, dim=0)

        # Register entities
        self.guard = self.manifold.register_entity(
            entity_id="guard_01",
            name="Captain John",
            role="Town Guard",
            embedding=v_combat,
            activation_threshold=0.35,
            base_prompt="You are Captain John of the Town Guard. Maintain order.",
        )

        self.barkeep = self.manifold.register_entity(
            entity_id="barkeep_01",
            name="Maeve the Barkeep",
            role="Tavern Owner",
            embedding=v_barkeep,
            activation_threshold=0.25,
            base_prompt="You are Maeve. Keep the patrons drinking and stop brawls.",
        )

        self.scholar = self.manifold.register_entity(
            entity_id="scholar_01",
            name="Master Lorian",
            role="Court Scholar",
            embedding=v_scholar,
            activation_threshold=0.45,
            base_prompt="You are Master Lorian. You study forgotten magical artifacts.",
        )

    def test_impulse_and_diffusion_bar_fight(self):
        """Player initiates a tavern fight: energy should hit Barkeep and Guard, sparing Scholar."""
        # Combat impulse aligned with combat/tavern violence
        v_combat = self.guard.embedding
        impulse_vec = F.normalize(v_combat + 0.1 * torch.randn(self.dim), dim=0)

        direct_hits = self.manifold.inject_impulse(
            text="Player smashes a bottle and draws a sword in the tavern!",
            embedding=impulse_vec,
            magnitude=0.85,
            source="player",
        )

        print("\n--- Event 1: Tavern Brawl Impulse ---")
        for eid, delta in direct_hits.items():
            name = self.manifold.entities[eid].name
            print(f"  {name:<20}: +{delta:.4f} energy")

        # Guard and Barkeep should receive significant direct energy
        self.assertGreater(self.guard.current_energy, 0.35)
        self.assertGreater(self.barkeep.current_energy, 0.05)
        # Scholar should receive very little direct perturbation
        self.assertLess(self.scholar.current_energy, 0.10)

        # Step 1 of temporal heat diffusion across the manifold topology
        active = self.manifold.step_diffusion(steps=1)
        print("  After Diffusion Step 1:")
        for eid, entity in self.manifold.entities.items():
            trig_str = "ACTIVE / WOKEN UP" if entity.is_triggered() else "quiescent"
            print(f"    {entity.name:<20}: energy={entity.current_energy:.4f} [{trig_str}]")

        # Guard is triggered!
        self.assertTrue(self.guard.is_triggered(), "Guard should trigger on combat impulse")
        # Scholar remains calm and dormant (zero GPU waste!)
        self.assertFalse(self.scholar.is_triggered(), "Scholar should not trigger on bar fight")

    def test_impulse_scholar_question(self):
        """Player asks about ancient magic: energy excites Scholar, leaves Guard quiescent."""
        v_scholar = self.scholar.embedding
        impulse_vec = F.normalize(v_scholar + 0.05 * torch.randn(self.dim), dim=0)

        direct_hits = self.manifold.inject_impulse(
            text="Player asks about the ancient magical glyphs on the tomb door.",
            embedding=impulse_vec,
            magnitude=0.90,
            source="player",
        )

        print("\n--- Event 2: Ancient Lore Impulse ---")
        for eid, delta in direct_hits.items():
            name = self.manifold.entities[eid].name
            print(f"  {name:<20}: +{delta:.4f} energy")

        self.assertTrue(self.scholar.is_triggered(), "Scholar must awaken for lore query")
        self.assertFalse(self.guard.is_triggered(), "Guard must not awaken for lore query")

    def test_multi_prototype_and_cascading_reaction(self):
        """
        Multi-hop causal reaction test:
        1. Event 'bridge destroyed' hits Transport aspect.
        2. Merchant directly awakens; Blacksmith is initially quiescent.
        3. Merchant reacts and emits secondary perturbation on Prices/Scarcity.
        4. Blacksmith (possessing trade/prices aspect) awakens on the secondary wave!
        """
        torch.manual_seed(999)
        dim = 64
        v_transport = F.normalize(torch.randn(dim), dim=0)
        v_prices = F.normalize(torch.randn(dim), dim=0)
        v_metal = F.normalize(torch.randn(dim), dim=0)
        v_weapons = F.normalize(torch.randn(dim), dim=0)

        manifold = ContinuousReactionManifold(hidden_dim=dim, decay_rate=0.10, diffusion_rate=0.30)

        # Merchant: cares about transport and prices
        merchant = manifold.register_entity(
            entity_id="merchant",
            name="Garrick the Merchant",
            role="Merchant",
            embedding=v_transport,
            prototypes={"transport": v_transport, "prices": v_prices},
            activation_threshold=0.40,
        )

        # Blacksmith: cares about metal, weapons, and fuel/trade prices
        blacksmith = manifold.register_entity(
            entity_id="blacksmith",
            name="Thorin the Blacksmith",
            role="Smith",
            embedding=v_metal,
            prototypes={"metal": v_metal, "weapons": v_weapons, "trade": v_prices},
            activation_threshold=0.35,
        )

        # 1. Player burns bridge (hits transport directly)
        hits = manifold.inject_impulse(
            text="The northern trade bridge was destroyed by raiders!",
            embedding=v_transport,
            magnitude=0.90,
            source="player",
        )

        print(f"\n--- Multi-Hop Test: Initial Bridge Fire ---")
        print(f"  Merchant energy : {merchant.current_energy:.4f} (Triggered: {merchant.is_triggered()})")
        print(f"  Blacksmith energy: {blacksmith.current_energy:.4f} (Triggered: {blacksmith.is_triggered()})")

        # Merchant wakes up immediately; Blacksmith does NOT wake up directly from bridge fire
        self.assertTrue(merchant.is_triggered(), "Merchant should wake from transport catastrophe")
        self.assertFalse(blacksmith.is_triggered(), "Blacksmith should NOT wake directly from bridge fire")

        # 2. Merchant reacts: emits secondary perturbation on Prices / Scarcity
        sec_hits = manifold.emit_reaction(
            entity_id="merchant",
            text="Supply route cut! Raw iron and coal prices soaring due to scarcity.",
            aspect="prices",
            magnitude=0.85,
        )

        print(f"--- After Merchant Reaction on Prices/Scarcity ---")
        print(f"  Blacksmith energy: {blacksmith.current_energy:.4f} (Triggered: {blacksmith.is_triggered()})")

        # Now the Blacksmith awakens via the secondary semantic wave!
        self.assertTrue(blacksmith.is_triggered(), "Blacksmith must awaken from downstream trade scarcity ripple!")


if __name__ == "__main__":
    unittest.main()

