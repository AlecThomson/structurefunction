"""Every attribute of every class in the package has a docstring."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "structurefunction"


def _attribute_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _is_docstring(node: ast.stmt | None) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def undocumented(path: Path) -> list[str]:
    """``Class.attribute`` names in ``path`` without an attribute docstring."""
    missing = []
    for cls in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(cls, ast.ClassDef):
            continue
        for index, node in enumerate(cls.body):
            name = _attribute_name(node)
            following = cls.body[index + 1] if index + 1 < len(cls.body) else None
            if name is not None and not _is_docstring(following):
                missing.append(f"{cls.name}.{name}")
    return missing


@pytest.mark.parametrize(
    "path",
    sorted(p for p in PACKAGE.rglob("*.py") if p.name != "_version.py"),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_every_attribute_has_a_docstring(path: Path) -> None:
    assert undocumented(path) == []
