"""
Cortex Dev Runtime: Software-Engineering Context Substrate Interfaces.
======================================================================
Defines core interfaces, data structures, and service contracts for watching,
querying, verifying, and explaining real codebases (U_v = <S_v, G_v, Z, H_v>).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch


class TestResultStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    UNTESTED = "UNTESTED"


class FileStatus(str, Enum):
    NOMINAL = "NOMINAL"
    MODIFIED = "MODIFIED"
    LINT_FAIL = "LINT_FAIL"
    SYNTAX_ERROR = "SYNTAX_ERROR"


@dataclass
class CodeSymbol:
    symbol_id: str             # e.g., "cortex_core.semantic_fabric::SemanticBand"
    name: str                  # e.g., "SemanticBand"
    kind: str                  # "function", "class", "method"
    file_path: str             # e.g., "cortex_core/semantic_fabric.py"
    start_line: int
    end_line: int
    docstring: str = ""
    calls: List[str] = field(default_factory=list)      # symbol_ids called
    imports: List[str] = field(default_factory=list)    # imported module/symbol names


@dataclass
class FileNode:
    file_path: str
    symbols: Dict[str, CodeSymbol] = field(default_factory=dict)
    imports: List[str] = field(default_factory=list)
    token_count: int = 100
    status: FileStatus = FileStatus.NOMINAL
    content: str = ""


@dataclass
class TestNode:
    test_id: str               # e.g., "cortex_validation/test_automation.py::test_engine_automation"
    file_path: str
    test_name: str
    mapped_symbols: List[str] = field(default_factory=list) # Code symbols tested
    status: TestResultStatus = TestResultStatus.UNTESTED
    duration_s: float = 0.0
    failure_trace: Optional[str] = None


@dataclass
class PatchDiff:
    patch_id: str
    description: str
    modified_files: Dict[str, str]   # file_path -> new_content
    author_agent: str = "ImplementationAgent"


@dataclass
class VerificationReport:
    permit: bool
    passed_tests: List[str]
    failed_tests: List[str]
    syntax_errors: List[str]
    broken_imports: List[str]
    version: int
    execution_time_ms: float
    reason: str = ""


@dataclass
class ImpactFrontier:
    modified_files: List[str]
    modified_symbols: List[str]
    direct_dependants: List[str]       # files/symbols directly importing/calling modified code
    transitive_dependants: List[str]   # transitive dependants in G_v
    mapped_tests: List[str]            # tests that exercise impacted code
    semantically_coupled_files: List[str] # files with high Z similarity
    version: int


@dataclass
class DevEvent:
    event_id: str
    timestamp: float
    event_type: str                    # "FILE_EDIT", "TEST_RUN", "PATCH_APPLIED", "AGENT_ACTION"
    target_path: str
    payload: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class DevMetrics:
    file_reads: int = 0
    ast_parses: int = 0
    embedding_calls: int = 0
    test_runs: int = 0
    test_lookups: int = 0
    dependency_traversals: int = 0
    tokens_reconstructed: int = 0
    duplicated_bytes: int = 0
    cpu_time_ms: float = 0.0
    inter_store_marshaling_calls: int = 0
    cross_store_sync_ops: int = 0

    @property
    def repo_tokens_reprocessed(self) -> int:
        """Repository source tokens reprocessed for context reconstruction."""
        return self.tokens_reconstructed

    def record_read(self, tokens: int = 0):
        self.file_reads += 1
        self.tokens_reconstructed += tokens


class DevContextSubstrate(ABC):
    """
    Abstract interface for Software-Engineering Context Substrates.
    """

    @abstractmethod
    def ingest_event(self, event: DevEvent) -> int:
        """Ingests a file edit or test execution, returning new version v."""
        pass

    @abstractmethod
    def apply_patch(self, patch: PatchDiff) -> int:
        """Applies a patch transactionally across S, G, Z, and H."""
        pass

    @abstractmethod
    def context(self, task_description: str, token_budget: int = 2000, version: Optional[int] = None) -> List[FileNode]:
        """Service 1: Assembles relevant context (code, tests, dependencies) under token budget."""
        pass

    @abstractmethod
    def impact(self, modified_files: List[str], version: Optional[int] = None) -> ImpactFrontier:
        """Service 2: Identifies affected symbols, dependants, and mapped tests."""
        pass

    @abstractmethod
    def route(self, event: DevEvent, version: Optional[int] = None) -> List[str]:
        """Service 3: Decides which agent specialists wake upon this change."""
        pass

    @abstractmethod
    def verify(self, patch: PatchDiff, version: Optional[int] = None) -> VerificationReport:
        """Service 4: Invariant verification ensuring patch is safe to commit."""
        pass

    @abstractmethod
    def explain(self, target: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Service 5: Explains why a file/test/patch is currently risky or failing."""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        pass

    @abstractmethod
    def get_metrics(self) -> DevMetrics:
        pass
