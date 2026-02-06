.. _api:

==========
Public API
==========

.. currentmodule:: crace

.. _api-data-io:

Data IO
-------

.. autosummary::
    :toctree: generated/

    open_dataset
    open_dataarray
    open_datatree


Data Alignment
--------------

.. autosummary::
    :toctree: generated/

    align

.. _api-data-tree:

Data Tree Manipulation
----------------------

.. autosummary::
    :toctree: generated/

    mask_dataset
    split_from_geo
    split_from_groupby
    split_from_groupby_bins
    merge_tree_dset


Data Tree Modification
----------------------

.. autosummary::
    :toctree: generated/

    impact_function
    map_impact_function
    map_aggregate_function
    REGISTRY


Engine
------

.. toctree::
    :maxdepth: 1

    api/engine


Reductions
----------

.. autosummary::
    :toctree: generated/

    at_event
    average_event_impact

.. _api-climada:

CLIMADA Interoperability
------------------------

.. currentmodule:: crace.climada

.. autosummary::
    :toctree: generated/

    hazard_to_dset
    exposure_to_dset
