# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

project = 'BJSS Model'
author = 'Moritz Buerger, Emanuel Jucker, Maximilian Spitaler, Linghang Sun'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',  # Create neat summary tables
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}

html_theme = "sphinx_rtd_theme"

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']