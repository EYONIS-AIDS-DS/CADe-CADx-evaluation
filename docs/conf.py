# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable without installing it
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "CADe/CADx Evaluation"
copyright = "2024, EYONIS-AIDS-DS contributors"
author = "EYONIS-AIDS-DS"
release = "1.0.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # Pull docstrings from source
    "sphinx.ext.napoleon",       # Google / NumPy docstring style
    "sphinx.ext.viewcode",       # Add [source] links to every symbol
    "sphinx.ext.autosummary",    # Auto-generate summary tables
    "sphinx.ext.intersphinx",    # Cross-reference external projects
    "sphinx_autodoc_typehints",  # Render type annotations in docs
    "myst_parser",               # Allow Markdown pages alongside RST
]

# Napoleon settings — use NumPy style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# autosummary — generate stub files automatically
autosummary_generate = True

# intersphinx mapping to standard libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output options
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

html_context = {
    "display_github": True,
    "github_user": "EYONIS-AIDS-DS",
    "github_repo": "CADe-CADx-evaluation",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# ---------------------------------------------------------------------------
# Source file suffixes
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
