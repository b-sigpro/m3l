# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
# Full options: https://www.sphinx-doc.org/en/master/usage/configuration.html

from pathlib import Path
import sys

# import m3l

# -- Path setup --------------------------------------------------------------
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path.absolute()))

# -- Project information -----------------------------------------------------
project = "m3l"
author = "National Institute of Advanced Industrial Science And Technology (AIST)"
project_copyright = author
# release = m3l.__version__

html_logo = f"{root_path}/docs/image/logo_m3l.png"
html_favicon = f"{root_path}/docs/image/favicon.ico"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

language = "en"

# -- HTML output -------------------------------------------------------------
html_theme = "shibuya"
html_show_sourcelink = False
html_show_sphinx = False

html_static_path = ["../image"]
html_theme_options = {
    "accent_color": "cyan",
    "nav_links": [
        {"title": "User Guide", "url": "user_guide/index"},
        {"title": "Recipes", "url": "recipes/index"},
        {
            "title": "API Reference",
            "url": "api_reference/index",
            "children": [
                {
                    "title": "Audio",
                    "url": "api_reference/audio",
                    "summary": "m3l.audio",
                },
                {
                    "title": "Audio-Visual",
                    "url": "api_reference/audiovisual",
                    "summary": "m3l.audiovisual",
                },
                {
                    "title": "Audio-Text",
                    "url": "api_reference/audiotext",
                    "summary": "m3l.audiotext",
                },
                {
                    "title": "Common",
                    "url": "api_reference/common",
                    "summary": "m3l.common",
                },
                {
                    "title": "Visual",
                    "url": "api_reference/visual",
                    "summary": "m3l.visual",
                },
                {
                    "title": "Utils",
                    "url": "api_reference/utils",
                    "summary": "m3l.utils",
                },
            ],
        },
        # {"title": "Contribution Guide", "url": "contribution_guide/index"},
    ],
    "github_url": "https://github.com/b-sigpro/m3l",
    "globaltoc_expand_depth": 1,
}
