"""
Sphinx configuration for the NetQMPI documentation.

The narrative pages are written in Markdown (MyST); the API reference is
generated with ``autodoc`` from the docstrings in the package itself, so
the reference never drifts away from the code.

Backends are documented without being installed: ``netqasm`` and ``cunqa``
are imported at module level by their adapters, but neither can be
installed in a public CI runner (NetQASM pulls NetSquid from a private,
credentialed index, and CUNQA needs an HPC/SLURM environment). They are
therefore mocked -- see ``autodoc_mock_imports`` below. The Aer and Qoala
adapters import their backends lazily, inside functions, so they need no
mocking to be documented.
"""
from __future__ import annotations

import os
import sys
from datetime import date

# Make the package importable without installing it, so a docs build works
# straight from a checkout.
sys.path.insert(0, os.path.abspath("../.."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "NetQMPI"
author = "F. Javier Cardama"
copyright = f"{date.today().year}, CiTIUS -- Universidade de Santiago de Compostela"


def _package_version() -> str:
    """Read the version from setup.py so it is declared in exactly one place."""
    import re

    setup_py = os.path.join(os.path.dirname(__file__), "..", "..", "setup.py")
    try:
        with open(setup_py, encoding="utf-8") as handle:
            match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", handle.read())
    except OSError:
        return "0.0.0"
    return match.group(1) if match else "0.0.0"


release = _package_version()
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google-style docstrings, as used in the code
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",    # writes .nojekyll for GitHub Pages
    "myst_parser",               # Markdown narrative pages
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",   # architecture and sequence diagrams
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The documentation is English-only for now. ``locale_dirs`` is declared
# up front so a translation can be added later with sphinx-intl without
# restructuring anything:
#
#     pip install sphinx-intl
#     make gettext && sphinx-intl update -p build/gettext -l es
#     sphinx-build -b html -D language=es source build/html/es
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# ---------------------------------------------------------------------------
# MyST (Markdown) configuration
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",      # ::: directives, easier to read than backtick fences
    "deflist",
    "fieldlist",
    "substitution",
    "attrs_inline",
]
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---------------------------------------------------------------------------
# autodoc / napoleon
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autoclass_content = "class"

# Backends that cannot be installed in a public CI runner. Their adapters
# import these at module level; mocking lets autodoc document the adapter
# classes anyway. Keep this list in sync with the module-level imports of
# netqmpi/runtime/adapters/*/.
autodoc_mock_imports = [
    "netqasm",
    "squidasm",
    "netsquid",
    "cunqa",
    "qoala",
    "qiskit",
    "qiskit_aer",
    "numpy",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
# Render `Attributes:` sections as :ivar: fields inside the class body.
# Without this, a property that is also listed under `Attributes:` — as many
# are in this codebase — is documented twice and Sphinx warns about it.
napoleon_use_ivar = True
napoleon_use_rtype = False

autosummary_generate = True

# ---------------------------------------------------------------------------
# intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Unresolvable references would otherwise be reported one by one on every
# build; the backends' own types are not documented here.
nitpicky = False

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"NetQMPI {release}"
# The network-node mark on a dark rounded tile, so it stays legible on both
# the light and the dark tab background browsers use by default -- a flat
# black or white glyph on transparent disappears on one of the two.
html_favicon = "_static/favicon.png"
html_copy_source = False
html_show_sourcelink = False

# The repository ships two logos: black strokes for light backgrounds,
# white strokes for dark ones.
html_theme_options = {
    "light_logo": "logo-dark.svg",
    "dark_logo": "logo-light.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/NetQIR/net-qmpi/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/NetQIR/net-qmpi",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                '8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01'
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95'
                '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 '
                '2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 '
                '2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 '
                '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
                "</path></svg>"
            ),
            "class": "",
        },
    ],
}

# Served from https://netqir.github.io/net-qmpi/
html_baseurl = "https://netqir.github.io/net-qmpi/"
