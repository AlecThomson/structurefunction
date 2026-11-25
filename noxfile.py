#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox"]
# ///
"""Task runner: ``./noxfile.py -s tests`` (or ``lint``, ``docs``, ``build``)."""

from __future__ import annotations

import shutil
from pathlib import Path

import nox

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """Run all pre-commit hooks with prek."""
    session.install("prek")
    session.run("prek", "run", "--all-files", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install("-e.", "--group", "test")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the HTML docs (``-- --serve`` for live reload)."""
    session.install("-e.", "--group", "docs")
    if "--serve" in session.posargs:
        session.install("sphinx-autobuild")
        session.run("sphinx-autobuild", "docs", "docs/_build/html")
    else:
        session.run(
            "sphinx-build", "-n", "-T", "-b", "html", "docs", "docs/_build/html"
        )


@nox.session
def build(session: nox.Session) -> None:
    """Build an sdist and wheel."""
    shutil.rmtree(Path("dist"), ignore_errors=True)
    session.install("build")
    session.run("python", "-m", "build")


if __name__ == "__main__":
    nox.main()
