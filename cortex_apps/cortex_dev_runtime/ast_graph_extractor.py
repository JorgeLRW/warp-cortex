"""
AST Dependency Graph Extractor (G_v).
=====================================
Analyzes real Python files using Python's standard `ast` module to build
the explicit code topology: imports, class/function definitions, symbol calls,
and test-to-code coverage mappings across warp_cortex.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from cortex_apps.cortex_dev_runtime.dev_runtime_api import (
    CodeSymbol,
    FileNode,
    FileStatus,
    TestNode,
    TestResultStatus,
)


class AstGraphExtractor:
    """
    Parses live Python files into an interconnected dependency graph G_v.
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.files: Dict[str, FileNode] = {}
        self.symbols: Dict[str, CodeSymbol] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)      # file -> imported files
        self.reverse_import_graph: Dict[str, Set[str]] = defaultdict(set) # file -> importing files
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)        # caller symbol -> callee symbols
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)# callee symbol -> callers
        self.test_nodes: Dict[str, TestNode] = {}
        self.symbol_to_tests: Dict[str, Set[str]] = defaultdict(set)  # symbol -> test_ids

    def scan_repository(self, target_subdirs: Optional[List[str]] = None) -> None:
        """Walks target subdirectories and parses all .py files."""
        if target_subdirs is None:
            target_subdirs = ["cortex_core", "cortex_apps", "cortex_validation"]

        for subdir in target_subdirs:
            full_sub = os.path.join(self.root_dir, subdir)
            if not os.path.exists(full_sub):
                continue
            for root, _, files in os.walk(full_sub):
                for f in files:
                    if f.endswith(".py") and not f.startswith("."):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, self.root_dir).replace("\\", "/")
                        self.parse_file(rel_path)

        self._build_test_mappings()
        return self.files

    def parse_file(self, rel_path: str, content: Optional[str] = None) -> Optional[FileNode]:
        """Parses a single file's AST and registers symbols and imports."""
        full_path = os.path.join(self.root_dir, rel_path)
        if content is None:
            if not os.path.exists(full_path):
                return None
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except Exception:
                return None

        token_count = max(10, len(content.split()))

        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError:
            node = FileNode(
                file_path=rel_path,
                symbols={},
                imports=[],
                token_count=token_count,
                status=FileStatus.SYNTAX_ERROR,
                content=content,
            )
            self.files[rel_path] = node
            return node

        file_symbols: Dict[str, CodeSymbol] = {}
        file_imports: List[str] = []

        mod_name = rel_path.replace(".py", "").replace("/", ".").replace("\\", ".")

        # First pass: collect imports and top-level definitions
        for item in tree.body:
            if isinstance(item, ast.Import):
                for alias in item.names:
                    file_imports.append(alias.name)
            elif isinstance(item, ast.ImportFrom):
                if item.module:
                    file_imports.append(item.module)

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym_id = f"{mod_name}::{item.name}"
                doc = ast.get_docstring(item) or ""
                sym = CodeSymbol(
                    symbol_id=sym_id,
                    name=item.name,
                    kind="function",
                    file_path=rel_path,
                    start_line=item.lineno,
                    end_line=item.end_lineno or item.lineno,
                    docstring=doc,
                )
                file_symbols[sym_id] = sym
                self.symbols[sym_id] = sym

            elif isinstance(item, ast.ClassDef):
                cls_id = f"{mod_name}::{item.name}"
                doc = ast.get_docstring(item) or ""
                sym = CodeSymbol(
                    symbol_id=cls_id,
                    name=item.name,
                    kind="class",
                    file_path=rel_path,
                    start_line=item.lineno,
                    end_line=item.end_lineno or item.lineno,
                    docstring=doc,
                )
                file_symbols[cls_id] = sym
                self.symbols[cls_id] = sym

                # Methods inside class
                for class_item in item.body:
                    if isinstance(class_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{cls_id}::{class_item.name}"
                        m_doc = ast.get_docstring(class_item) or ""
                        m_sym = CodeSymbol(
                            symbol_id=method_id,
                            name=f"{item.name}::{class_item.name}",
                            kind="method",
                            file_path=rel_path,
                            start_line=class_item.lineno,
                            end_line=class_item.end_lineno or class_item.lineno,
                            docstring=m_doc,
                        )
                        file_symbols[method_id] = m_sym
                        self.symbols[method_id] = m_sym

        # Second pass: resolve function calls
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                caller_id = f"{mod_name}::{item.name}"
                self._extract_calls(item, caller_id)
            elif isinstance(item, ast.ClassDef):
                cls_id = f"{mod_name}::{item.name}"
                for class_item in item.body:
                    if isinstance(class_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{cls_id}.{class_item.name}"
                        self._extract_calls(class_item, method_id)

        # Update import graph
        self.import_graph[rel_path].clear()
        for imp in file_imports:
            # Map import module to file path if within repo
            target_rel = imp.replace(".", "/") + ".py"
            if target_rel in self.files or os.path.exists(os.path.join(self.root_dir, target_rel)):
                self.import_graph[rel_path].add(target_rel)
                self.reverse_import_graph[target_rel].add(rel_path)

        fnode = FileNode(
            file_path=rel_path,
            symbols=file_symbols,
            imports=file_imports,
            token_count=token_count,
            status=FileStatus.NOMINAL,
            content=content,
        )
        self.files[rel_path] = fnode

        # If test file, create test nodes
        if "test_" in rel_path or "/tests/" in rel_path:
            for sym_id, sym in file_symbols.items():
                if sym.name.startswith("test_") or "Test" in sym.name:
                    test_id = f"{rel_path}::{sym.name}"
                    self.test_nodes[test_id] = TestNode(
                        test_id=test_id,
                        file_path=rel_path,
                        test_name=sym.name,
                        mapped_symbols=[],
                        status=TestResultStatus.UNTESTED,
                    )

        return fnode

    def _extract_calls(self, node: ast.AST, caller_id: str):
        """Extracts called names inside a function node."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                called_name = None
                if isinstance(func, ast.Name):
                    called_name = func.id
                elif isinstance(func, ast.Attribute):
                    called_name = func.attr

                if called_name:
                    if caller_id in self.symbols:
                        self.symbols[caller_id].calls.append(called_name)
                    # Register potential match in call graph
                    for candidate_id in self.symbols:
                        if candidate_id.endswith(f"::{called_name}") or candidate_id.endswith(f".{called_name}"):
                            self.call_graph[caller_id].add(candidate_id)
                            self.reverse_call_graph[candidate_id].add(caller_id)

    def _build_test_mappings(self):
        """Maps test nodes to the production code symbols and files they test."""
        for test_id, tnode in self.test_nodes.items():
            test_file = tnode.file_path
            imported_files = self.import_graph.get(test_file, set())

            # Symbols from imported files called by the test
            tested_symbols = set()
            for imp_file in imported_files:
                if imp_file in self.files:
                    for sym_id, sym in self.files[imp_file].symbols.items():
                        # If test file calls symbol name or imports module
                        tested_symbols.add(sym_id)

            tnode.mapped_symbols = list(tested_symbols)
            for s in tested_symbols:
                self.symbol_to_tests[s].add(test_id)

    def get_transitive_dependants(self, modified_files: List[str]) -> Tuple[Set[str], Set[str]]:
        """Returns (transitive_dependent_files, impacted_tests)."""
        visited_files: Set[str] = set(modified_files)
        queue = list(modified_files)

        while queue:
            curr = queue.pop(0)
            for parent_file in self.reverse_import_graph.get(curr, []):
                if parent_file not in visited_files:
                    visited_files.add(parent_file)
                    queue.append(parent_file)

        impacted_tests: Set[str] = set()
        for f in visited_files:
            # Check direct tests on this file
            for test_id, tnode in self.test_nodes.items():
                if tnode.file_path == f:
                    impacted_tests.add(test_id)
            # Check tests that mapped to symbols in this file
            if f in self.files:
                for sym_id in self.files[f].symbols:
                    for t_id in self.symbol_to_tests.get(sym_id, []):
                        impacted_tests.add(t_id)

        return visited_files, impacted_tests

    @property
    def reverse_dependencies(self) -> Dict[str, Set[str]]:
        return self.reverse_import_graph

    def parse_content(self, rel_path: str, content: str) -> FileNode:
        node = self.parse_file(rel_path, content=content)
        return node or FileNode(file_path=rel_path, content=content)


ASTGraphExtractor = AstGraphExtractor

