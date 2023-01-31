# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ReSketch AI'
copyright = '2023, Ian Wasser, Robin Steiner'
author = 'Ian Wasser, Robin Steiner'
release = '1.0'

import sys
import sphinx_rtd_theme
sys.path.insert(0, "../src")
sys.path.insert(0, "../src/extras")
sys.path.insert(0, "../src/data")
sys.path.insert(0, "../src/data_statistics")
sys.path.insert(0, "../src/models")
sys.path.insert(0, "../src/optimizers")
sys.path.insert(0, "../src/physics_modules")
sys.path.insert(0, "../src/reproduce_modules")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
