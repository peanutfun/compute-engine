# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CRACE"
copyright = "2025, ETH Zurich"
author = "Lukas Riedel"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_remove_toctrees",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc
autodoc_mock_imports = ["climada"]
autodoc_typehints = "description"

# Napoleon
napoleon_use_param = True
napoleon_preprocess_types = True
napoleon_use_rtype = True

napoleon_type_aliases = {
    "DatasetOrArray": ":py:class:`xarray.Dataset` | :py:class:`xarray.DataArray`",
    "climada.hazard.Hazard": ":py:class:`~climada.hazard.base.Hazard`",
    "climada.Exposures": ":py:class:`~climada.entity.exposures.base.Exposures`",
    "climada.entity.Exposures": ":py:class:`~climada.entity.exposures.base.Exposures`",
    "ArrayLike": "py:class:`numpy.typing.ArrayLike`",
    "numpy.ArrayLike": "py:class:`numpy.typing.ArrayLike`",
}
autodoc_type_aliases = napoleon_type_aliases

# Typehints
# typehints_fully_qualified = False
# typehints_defaults = "braces-after"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "climada": ("https://climada-python.readthedocs.io/en/stable/", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/html/", None),
    "odcgeo": ("https://odc-geo.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
}

# Remove from toctree
remove_from_toctrees = ["api/generated/*", "generated/*"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
