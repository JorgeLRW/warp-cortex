"""
Unit & Integration Tests for Cortex Dev Runtime.
=================================================
Tests all components across warp_cortex:
  1. AST Graph Extractor (imports, symbols, caller-callee, test mappings)
  2. Multi-Aspect Code Indexer (Z)
  3. Event History Log (H_v)
  4. Test Status Tracker (S_v)
  5. The 5 Core Services (context, impact, route, verify, explain)
  6. Service 6 (why_changed)
"""

import os
import pytest
import torch

from cortex_apps.cortex_dev_runtime.ast_graph_extractor import ASTGraphExtractor
from cortex_apps.cortex_dev_runtime.conventional_dev_runtime import ConventionalDevRuntime
from cortex_apps.cortex_dev_runtime.dev_agents import DevAgentCoordinator
from cortex_apps.cortex_dev_runtime.dev_runtime_api import (
    CodeSymbol,
    DevEvent,
    FileNode,
    FileStatus,
    PatchDiff,
    TestNode,
    TestResultStatus,
)
from cortex_apps.cortex_dev_runtime.event_history_log import EventHistoryLog
from cortex_apps.cortex_dev_runtime.semantic_code_indexer import MultiAspectCodeIndexer
from cortex_apps.cortex_dev_runtime.service6_why_changed import (
    why_changed_conventional,
    why_changed_unified,
)
from cortex_apps.cortex_dev_runtime.test_status_tracker import TestStatusTracker
from cortex_apps.cortex_dev_runtime.unified_dev_substrate import UnifiedDevContextSubstrate


@pytest.fixture
def repo_root():
    # Return warp_cortex root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def test_ast_graph_extractor(repo_root):
    extractor = ASTGraphExtractor(repo_root)
    files = extractor.scan_repository()

    assert len(files) > 50, f"Expected >50 files in warp_cortex, got {len(files)}"
    assert len(extractor.symbols) > 500, f"Expected >500 symbols, got {len(extractor.symbols)}"
    assert len(extractor.test_nodes) > 10, f"Expected discovered test nodes, got {len(extractor.test_nodes)}"

    # Check key file exists
    assert "cortex_core/semantic_fabric.py" in files or "cortex_core/cortex_runtime.py" in files


def test_multi_aspect_indexer(repo_root):
    indexer = MultiAspectCodeIndexer(d_band=32)
    sample_file = FileNode(
        file_path="sample.py",
        content=(
            "import os\n"
            "class Engine:\n"
            "    def run(self):\n"
            "        assert True\n"
            "        for i in range(100): pass\n"
        ),
        token_count=20,
    )
    vec = indexer.extract_bands_from_file(sample_file)
    assert vec.shape == (4, 32)
    # Check normalization
    for b in range(4):
        norm = torch.norm(vec[b], p=2).item()
        assert norm > 0.0


def test_event_history_log():
    log = EventHistoryLog()
    ev1 = log.append_event("FILE_EDIT", "cortex_core/cortex_runtime.py", {"line": 10})
    assert ev1.version == 1
    assert log.current_version == 1

    patch = PatchDiff(
        patch_id="p1",
        description="test patch",
        modified_files={"cortex_core/cortex_runtime.py": "# new code"},
    )
    ev2 = log.record_patch(patch)
    assert ev2.version == 2
    assert log.current_version == 2

    between = log.get_events_between(0, 2)
    assert len(between) == 2


def test_test_status_tracker(repo_root):
    tracker = TestStatusTracker(repo_root)
    test_node = TestNode(
        test_id="cortex_validation/test_automation.py::test_router_logic",
        file_path="cortex_validation/test_automation.py",
        test_name="test_router_logic",
    )
    tracker.register_tests({test_node.test_id: test_node})

    passed, failed, traces = tracker.run_tests_programmatic([test_node.test_id], timeout_s=25.0)
    assert len(passed) == 1, f"Expected test_router_logic to pass, got failed={failed}, traces={traces}"
    assert len(failed) == 0


def test_unified_dev_substrate_services(repo_root):
    substrate = UnifiedDevContextSubstrate(repo_root)

    # 1. Context Service
    ctx = substrate.context("semantic fabric bands and latent topology", token_budget=1500)
    assert len(ctx) > 0

    # 2. Impact Service
    impact = substrate.impact(["cortex_core/semantic_fabric.py"])
    assert len(impact.modified_files) == 1
    assert len(impact.direct_dependants) > 0 or len(impact.modified_symbols) > 0

    # 3. Route Service
    ev = DevEvent("e1", 1.0, "FILE_EDIT", "cortex_core/semantic_fabric.py")
    agents = substrate.route(ev)
    assert "VerificationAgent" in agents

    # 4. Verify Service (valid syntax patch)
    valid_patch = PatchDiff(
        patch_id="p_valid",
        description="Add harmless helper",
        modified_files={
            "cortex_core/semantic_fabric.py": "def harmless_helper(): return 42\n"
        },
    )
    report = substrate.verify(valid_patch)
    assert len(report.syntax_errors) == 0

    # 5. Explain Service
    explanation = substrate.explain("cortex_core/semantic_fabric.py")
    assert "target" in explanation
    assert "causal_chain" in explanation


def test_service6_why_changed(repo_root):
    unified = UnifiedDevContextSubstrate(repo_root)
    p = PatchDiff(
        patch_id="p_test_6",
        description="service 6 patch",
        modified_files={"cortex_core/cortex_runtime.py": "# change\n"},
    )
    v1 = unified.version
    unified.apply_patch(p)
    v2 = unified.version

    res_u = why_changed_unified(unified, "cortex_core/cortex_runtime.py", v1, v2)
    assert res_u.stores_touched == 1
    assert res_u.synchronization_paths == 0
    assert len(res_u.modifying_events) >= 1


def test_persistent_conventional_runtime(repo_root):
    from cortex_apps.cortex_dev_runtime.persistent_conventional_runtime import PersistentConventionalDevRuntime
    from cortex_apps.cortex_dev_runtime.service6_why_changed import why_changed_persistent_conventional

    runtime_c = PersistentConventionalDevRuntime(repo_root)

    # 1. Context
    ctx = runtime_c.context("transition governor telemetry", token_budget=1500)
    assert len(ctx) > 0

    # 2. Impact
    imp = runtime_c.impact(["cortex_core/transition_governor.py"])
    assert len(imp.modified_files) == 1

    # 3. Patch & Invariants
    p = PatchDiff(
        patch_id="p_test_c",
        description="test patch c",
        modified_files={"cortex_core/transition_governor.py": "def test_c_helper(): return 1\n"},
    )
    v1 = runtime_c.version
    v2 = runtime_c.apply_patch(p)
    assert v2 > v1

    # 4. Service 6
    res_c = why_changed_persistent_conventional(runtime_c, "cortex_core/transition_governor.py", v1, v2)
    assert res_c.stores_touched == 4
    assert res_c.synchronization_paths == 3
    assert len(res_c.modifying_events) >= 1
