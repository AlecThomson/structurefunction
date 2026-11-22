"""Execute the documentation notebooks (spinifex-style self-checking examples)."""

from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nbformat
import pytest
from nbconvert import ScriptExporter

EXAMPLES = sorted((Path(__file__).parents[1] / "docs" / "examples").glob("*.ipynb"))


@pytest.mark.slow_uncompiled
@pytest.mark.parametrize("notebook", EXAMPLES, ids=lambda p: p.name)
def test_notebook_runs(
    notebook: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "treecorr" in notebook.name:
        pytest.importorskip("treecorr")
    monkeypatch.setenv("JUPYTER_PLATFORM_DIRS", "1")
    mpl.use("Agg")
    # Agg cannot show figures and warns; close them instead.
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    script, _ = ScriptExporter().from_notebook_node(
        nbformat.read(notebook, as_version=4)
    )
    path = tmp_path / notebook.with_suffix(".py").name
    path.write_text(script, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(path))
