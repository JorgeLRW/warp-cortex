"""
Catalog of 25 Real Repository Engineering Tasks for warp_cortex.
================================================================
Ground truth tasks operating on actual files across warp_cortex:
  - cortex_apps/research_agent_system/memory_baselines.py (including SharedFrozenEventResolver)
  - cortex_core/semantic_fabric.py
  - cortex_core/transition_governor.py
  - cortex_core/epistemic_manifold.py
  - cortex_core/cortex_runtime.py
  - cortex_validation/test_automation.py

Categories:
  1. Core API & Signature Updates (Tasks 1-5)
  2. The Killer Scenario: SharedFrozenEventResolver Updates & Invariants (Tasks 6-10)
  3. Structural AST & Dependency Refactoring (Tasks 11-15)
  4. Test Fixtures, Regression Guards & Invariant Fixes (Tasks 16-20)
  5. Semantic Band & Latent Topology Tweaks (Tasks 21-25)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from cortex_apps.cortex_dev_runtime.dev_runtime_api import PatchDiff


@dataclass
class DevTaskDefinition:
    task_id: str
    category: str
    title: str
    description: str
    target_files: List[str]
    expected_success: bool
    is_killer_scenario: bool
    patch: PatchDiff


def build_real_tasks_suite(root_dir: str) -> List[DevTaskDefinition]:
    tasks: List[DevTaskDefinition] = []

    # ------------------------------------------------------------------------
    # Category 1: Core API & Signature Updates (Tasks 1-5)
    # ------------------------------------------------------------------------
    tasks.append(
        DevTaskDefinition(
            task_id="TASK_01",
            category="API_UPDATE",
            title="Add Telemetry Hook to TransitionGovernor",
            description="Expose record_transition_metric in cortex_core/transition_governor.py with docstring.",
            target_files=["cortex_core/transition_governor.py"],
            expected_success=True,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_01",
                description="Add transition telemetry helper",
                modified_files={
                    "cortex_core/transition_governor.py": (
                        "# [TASK_01 Patch: Telemetry Hook]\n"
                        "from cortex_core.cortex_runtime import *\n\n"
                        "def record_transition_metric(metric_name: str, value: float) -> dict:\n"
                        '    """Records transition latency or stability metric."""\n'
                        "    return {'metric': metric_name, 'val': float(value), 'ts': 1.0}\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_02",
            category="API_UPDATE",
            title="Syntax Error Invariant Test",
            description="Propose patch with deliberate syntax error in cortex_core/cortex_runtime.py.",
            target_files=["cortex_core/cortex_runtime.py"],
            expected_success=False,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_02",
                description="Malformed syntax patch",
                modified_files={
                    "cortex_core/cortex_runtime.py": "def invalid_syntax_func(:\n    pass\n"
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_03",
            category="API_UPDATE",
            title="Expose Band Dimensions in SemanticFabric",
            description="Add helper to inspect available semantic bands in cortex_core/semantic_fabric.py.",
            target_files=["cortex_core/semantic_fabric.py"],
            expected_success=True,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_03",
                description="Expose semantic band introspection",
                modified_files={
                    "cortex_core/semantic_fabric.py": (
                        "# [TASK_03 Patch: Semantic Fabric helper]\n"
                        "def get_supported_semantic_bands() -> list:\n"
                        '    """Returns the list of 4 canonical semantic bands."""\n'
                        "    return ['ARCH', 'LOGIC', 'INVARIANTS', 'PERF']\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_04",
            category="API_UPDATE",
            title="Add Batch Assertion to EpistemicManifold",
            description="Add dimension assertion helper in cortex_core/epistemic_manifold.py.",
            target_files=["cortex_core/epistemic_manifold.py"],
            expected_success=True,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_04",
                description="Add batch shape validation helper",
                modified_files={
                    "cortex_core/epistemic_manifold.py": (
                        "# [TASK_04 Patch: Dimension assertion]\n"
                        "def validate_manifold_shape(shape: tuple) -> bool:\n"
                        "    assert len(shape) >= 2, 'Manifold tensor must be at least 2D'\n"
                        "    return True\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_05",
            category="API_UPDATE",
            title="Import Failure Injection",
            description="Introduce import of non-existent module in cortex_core/adaptive_engine.py.",
            target_files=["cortex_core/adaptive_engine.py"],
            expected_success=False,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_05",
                description="Broken import test",
                modified_files={
                    "cortex_core/adaptive_engine.py": (
                        "import non_existent_quantum_tensor_lib_xyz_123\n\n"
                        "def bad_func():\n    return 0\n"
                    )
                },
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Category 2: Killer Scenario & SharedFrozenEventResolver (Tasks 6-10)
    # ------------------------------------------------------------------------
    tasks.append(
        DevTaskDefinition(
            task_id="TASK_06",
            category="KILLER_SCENARIO",
            title="Killer Scenario: Incompatible Return in SharedFrozenEventResolver",
            description=(
                "Modify SharedFrozenEventResolver in memory_baselines.py: change resolve_event "
                "to return a tuple instead of EventVector. Locally looks fine, but breaks "
                "downstream consumers and invariant tests."
            ),
            target_files=["cortex_apps/research_agent_system/memory_baselines.py"],
            expected_success=False,
            is_killer_scenario=True,
            patch=PatchDiff(
                patch_id="PATCH_06_KILLER",
                description="Subtle return type breaking change in SharedFrozenEventResolver",
                modified_files={
                    "cortex_apps/research_agent_system/memory_baselines.py": (
                        "# [KILLER SCENARIO PATCH]\n"
                        "class SharedFrozenEventResolver:\n"
                        "    def __init__(self, catalog=None):\n"
                        "        self.catalog = catalog or {}\n"
                        "    def resolve_event(self, raw_ev):\n"
                        "        # Intentional breaking change: tuple instead of object\n"
                        "        return ('broken_payload', 0.0)\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_07",
            category="KILLER_SCENARIO",
            title="Safe Cache Addition to SharedFrozenEventResolver",
            description="Add thread-safe hit counter and hit ratio to SharedFrozenEventResolver preserving API contract.",
            target_files=["cortex_apps/research_agent_system/memory_baselines.py"],
            expected_success=True,
            is_killer_scenario=True,
            patch=PatchDiff(
                patch_id="PATCH_07",
                description="Preserve contract and add hit ratio tracking",
                modified_files={
                    "cortex_apps/research_agent_system/memory_baselines.py": (
                        "# [TASK_07 Patch: Thread-safe Cache Counters]\n"
                        "class SharedFrozenEventResolver:\n"
                        "    def __init__(self, catalog=None):\n"
                        "        self.catalog = catalog or {}\n"
                        "        self.hits = 0\n"
                        "        self.misses = 0\n\n"
                        "    def get_hit_ratio(self) -> float:\n"
                        "        total = self.hits + self.misses\n"
                        "        return float(self.hits) / total if total > 0 else 1.0\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_08",
            category="KILLER_SCENARIO",
            title="SharedFrozenEventResolver Cache Eviction Logic",
            description="Add LRU eviction cap to SharedFrozenEventResolver.",
            target_files=["cortex_apps/research_agent_system/memory_baselines.py"],
            expected_success=True,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_08",
                description="Add max_entries and prune logic",
                modified_files={
                    "cortex_apps/research_agent_system/memory_baselines.py": (
                        "# [TASK_08 Patch: Cache cap]\n"
                        "class SharedFrozenEventResolver:\n"
                        "    def __init__(self, catalog=None, max_entries: int = 10000):\n"
                        "        self.catalog = catalog or {}\n"
                        "        self.max_entries = max_entries\n"
                        "    def prune(self):\n"
                        "        if len(self.catalog) > self.max_entries:\n"
                        "            self.catalog.clear()\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_09",
            category="KILLER_SCENARIO",
            title="SharedFrozenEventResolver Key Mismatch Bug",
            description="Hash collision or corrupted key formatting in resolver.",
            target_files=["cortex_apps/research_agent_system/memory_baselines.py"],
            expected_success=False,
            is_killer_scenario=True,
            patch=PatchDiff(
                patch_id="PATCH_09",
                description="Corrupt key normalization",
                modified_files={
                    "cortex_apps/research_agent_system/memory_baselines.py": (
                        "class SharedFrozenEventResolver:\n"
                        "    def resolve_key(self, k):\n"
                        "        raise KeyError(f'Resolver corrupted on key {k}')\n"
                    )
                },
            ),
        )
    )

    tasks.append(
        DevTaskDefinition(
            task_id="TASK_10",
            category="KILLER_SCENARIO",
            title="SharedFrozenEventResolver Audit Export",
            description="Export all resolved event IDs as frozen set.",
            target_files=["cortex_apps/research_agent_system/memory_baselines.py"],
            expected_success=True,
            is_killer_scenario=False,
            patch=PatchDiff(
                patch_id="PATCH_10",
                description="Export frozen set of keys",
                modified_files={
                    "cortex_apps/research_agent_system/memory_baselines.py": (
                        "# [TASK_10 Patch: Audit Export]\n"
                        "class SharedFrozenEventResolver:\n"
                        "    def __init__(self, catalog=None):\n"
                        "        self.catalog = catalog or {}\n"
                        "    def export_keys(self) -> frozenset:\n"
                        "        return frozenset(self.catalog.keys())\n"
                    )
                },
            ),
        )
    )

    # ------------------------------------------------------------------------
    # Category 3: Structural AST & Dependency Refactoring (Tasks 11-15)
    # ------------------------------------------------------------------------
    for i in range(11, 16):
        succ = (i % 2 == 1)
        tasks.append(
            DevTaskDefinition(
                task_id=f"TASK_{i:02d}",
                category="AST_REFACTOR",
                title=f"AST Structure Refactor Step {i}",
                description=f"Refactor symbol resolution pipeline step {i} across cortex_core/manifold_topology.py.",
                target_files=["cortex_core/manifold_topology.py"],
                expected_success=succ,
                is_killer_scenario=False,
                patch=PatchDiff(
                    patch_id=f"PATCH_{i:02d}",
                    description=f"Refactor step {i}",
                    modified_files={
                        "cortex_core/manifold_topology.py": (
                            f"# Refactor step {i}\n"
                            + ("def valid_step(): return True\n" if succ else "def broken_step() return False\n")
                        )
                    },
                ),
            )
        )

    # ------------------------------------------------------------------------
    # Category 4: Test Fixtures, Regression Guards & Invariants (Tasks 16-20)
    # ------------------------------------------------------------------------
    for i in range(16, 21):
        succ = (i != 18)  # Task 18 introduces a failing test assertion
        tasks.append(
            DevTaskDefinition(
                task_id=f"TASK_{i:02d}",
                category="TEST_INVARIANTS",
                title=f"Regression Guard & Invariant Test {i}",
                description=f"Update test harness and invariant assertions in cortex_validation/test_automation.py.",
                target_files=["cortex_validation/test_automation.py"],
                expected_success=succ,
                is_killer_scenario=False,
                patch=PatchDiff(
                    patch_id=f"PATCH_{i:02d}",
                    description=f"Test automation patch {i}",
                    modified_files={
                        "cortex_validation/test_automation.py": (
                            "# [TASK Automation Patch]\n"
                            "import pytest\n\n"
                            f"def test_engine_automation():\n"
                            f"    assert {1 if succ else 0} == 1\n"
                        )
                    },
                ),
            )
        )

    # ------------------------------------------------------------------------
    # Category 5: Semantic Band & Latent Topology Tweaks (Tasks 21-25)
    # ------------------------------------------------------------------------
    for i in range(21, 26):
        succ = (i != 23)
        tasks.append(
            DevTaskDefinition(
                task_id=f"TASK_{i:02d}",
                category="SEMANTIC_BAND",
                title=f"Semantic Band Normalization Tweaks {i}",
                description=f"Fine-tune band normalization scaling in cortex_core/semantic_fabric.py.",
                target_files=["cortex_core/semantic_fabric.py"],
                expected_success=succ,
                is_killer_scenario=False,
                patch=PatchDiff(
                    patch_id=f"PATCH_{i:02d}",
                    description=f"Semantic band normalization {i}",
                    modified_files={
                        "cortex_core/semantic_fabric.py": (
                            f"# Semantic Band Patch {i}\n"
                            + ("def normalize_aspect_band(t): return t / (t.norm() + 1e-6)\n" if succ else "def normalize_aspect_band(t): raise ValueError('Invalid band')\n")
                        )
                    },
                ),
            )
        )

    return tasks
