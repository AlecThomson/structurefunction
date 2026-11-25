"""Sphinx configuration."""

from __future__ import annotations

import importlib.metadata

project = "structurefunction"
author = "Alec Thomson"
copyright = "2021-2026, Alec Thomson"  # noqa: A001
version = release = importlib.metadata.version("structurefunction")

extensions = [
    "autoapi.extension",
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.intersphinx",
    "sphinxarg.ext",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
html_theme = "furo"

autoapi_dirs = ["../structurefunction"]
autoapi_ignore = ["*/_version.py"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "imported-members"]

# Notebooks are executed at build time so the docs always match the code.
nbsphinx_execute = "always"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

always_document_param_types = True
