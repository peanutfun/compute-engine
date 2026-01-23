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


Impact Functions
----------------

.. autosummary::
    :toctree: generated/

    impact_function


Reductions
----------

.. autosummary::
    :toctree: generated/

    at_event
    average_event_impact


Engine
------

.. autosummary::
    :toctree: generated/
    :recursive:

    Engine


.. _api-climada:

CLIMADA Interoperability
------------------------

.. currentmodule:: crace.climada

.. autosummary::
    :toctree: generated/

    hazard_to_dset
    exposure_to_dset
