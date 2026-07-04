"""AST whitelist validation for LLM-generated code (CLAUDE.md rule #6 / PRD FR-26).

This is the static gate every "custom_code" feature step must pass BEFORE it
is ever dry-run (see src/sandbox/execute.py). No exceptions, including
"trusted" internal use.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_MODULES = {"pandas", "numpy", "math", "re", "datetime"}

DISALLOWED_CALL_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "exit",
    "quit",
}

DISALLOWED_MODULE_PREFIXES = ("os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http")


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_MODULES:
                self.errors.append(f"disallowed import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_MODULES:
            self.errors.append(f"disallowed import: 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALL_NAMES:
            self.errors.append(f"disallowed call: '{node.func.id}(...)'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"disallowed dunder attribute access: '.{node.attr}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith(DISALLOWED_MODULE_PREFIXES) and node.id not in ALLOWED_MODULES:
            self.errors.append(f"disallowed name reference: '{node.id}'")
        self.generic_visit(node)


def validate_code(code: str) -> ValidationResult:
    """Static validation only. This does NOT execute the code — see
    src/sandbox/execute.py for the isolated dry-run step that must follow."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return ValidationResult(valid=False, errors=[f"syntax error: {exc}"])

    visitor = _Visitor()
    visitor.visit(tree)

    has_transform_fn = any(
        isinstance(node, ast.FunctionDef) and node.name == "transform" for node in ast.walk(tree)
    )
    if not has_transform_fn:
        visitor.errors.append("code must define a top-level `def transform(df):` function")

    return ValidationResult(valid=len(visitor.errors) == 0, errors=visitor.errors)
