"""
Test Status & Build State Tracker (S_v).
=========================================
Tracks operational reality across warp_cortex:
  - Per-file edit & lint status (NOMINAL, MODIFIED, SYNTAX_ERROR)
  - Per-test execution outcomes (PASSED, FAILED, ERROR, UNTESTED)
  - Failure tracebacks, error contexts, and execution durations.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from cortex_apps.cortex_dev_runtime.dev_runtime_api import (
    FileStatus,
    TestNode,
    TestResultStatus,
)


class TestStatusTracker:
    """
    Maintains the operational truth table S_v for code files and test suites.
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.file_states: Dict[str, FileStatus] = {}
        self.test_states: Dict[str, TestNode] = {}
        self.last_run_timestamp: float = 0.0
        self._cached_results: Dict[str, Tuple[bool, Optional[str]]] = {}

    def register_tests(self, test_nodes: Dict[str, TestNode]):
        """Registers discovered test nodes."""
        self.test_states.update(test_nodes)

    def mark_file_status(self, file_path: str, status: FileStatus):
        self.file_states[file_path] = status
        # Invalidate cached test outcomes when file changes
        keys_to_del = [tid for tid in self._cached_results if file_path in tid]
        for k in keys_to_del:
            del self._cached_results[k]

    def get_failing_tests(self) -> List[TestNode]:
        """Returns all currently failing tests."""
        return [t for t in self.test_states.values() if t.status in (TestResultStatus.FAILED, TestResultStatus.ERROR)]

    def run_tests_programmatic(
        self,
        test_ids: List[str],
        timeout_s: float = 25.0,
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Executes pytest targeted on specific test IDs with incremental result caching.
        Returns (passed_test_ids, failed_test_ids, failure_traces).
        """
        t0 = time.perf_counter()
        passed: List[str] = []
        failed: List[str] = []
        traces: Dict[str, str] = {}

        if not test_ids:
            return passed, failed, traces

        tests_to_execute = []
        for tid in test_ids:
            if tid in self._cached_results:
                is_pass, tr = self._cached_results[tid]
                if is_pass:
                    passed.append(tid)
                    if tid in self.test_states:
                        self.test_states[tid].status = TestResultStatus.PASSED
                else:
                    failed.append(tid)
                    if tr:
                        traces[tid] = tr
                    if tid in self.test_states:
                        self.test_states[tid].status = TestResultStatus.FAILED
                        self.test_states[tid].failure_trace = tr
            else:
                tests_to_execute.append(tid)

        if not tests_to_execute:
            return passed, failed, traces

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "-o",
            "asyncio_default_fixture_loop_scope=function",
        ] + tests_to_execute

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = proc.stdout + "\n" + proc.stderr

            for tid in tests_to_execute:
                tname = tid.split("::")[-1] if "::" in tid else tid
                fpath = tid.split("::")[0]
                if proc.returncode == 0 or (f"{tid} PASSED" in output or f"PASSED" in output and f"FAILED {tid}" not in output and f"FAILED {fpath}::{tname}" not in output and "ERRORS" not in output):
                    passed.append(tid)
                    self._cached_results[tid] = (True, None)
                    if tid in self.test_states:
                        self.test_states[tid].status = TestResultStatus.PASSED
                        self.test_states[tid].failure_trace = None
                else:
                    failed.append(tid)
                    tr = output[-800:]
                    traces[tid] = tr
                    self._cached_results[tid] = (False, tr)
                    if tid in self.test_states:
                        self.test_states[tid].status = TestResultStatus.FAILED
                        self.test_states[tid].failure_trace = tr
        except subprocess.TimeoutExpired:
            for tid in tests_to_execute:
                failed.append(tid)
                tr = f"Pytest execution timed out after {timeout_s}s."
                traces[tid] = tr
                self._cached_results[tid] = (False, tr)
                if tid in self.test_states:
                    self.test_states[tid].status = TestResultStatus.ERROR
                    self.test_states[tid].failure_trace = tr
        except Exception as e:
            for tid in test_ids:
                failed.append(tid)
                traces[tid] = str(e)
                if tid in self.test_states:
                    self.test_states[tid].status = TestResultStatus.ERROR
                    self.test_states[tid].failure_trace = traces[tid]

        self.last_run_timestamp = time.time()
        return passed, failed, traces
