.. _data-io:

============
Loading Data
============

In CRACE, hazard and exposure data is represented by :py:class:`xarray.Dataset` objects with geospatial encoding.
Xarray provides functions like :py:func:`xarray.open_dataset`, that create such objects from data files.
In many cases, you will have to install a specific backend to parse the file.
See :external+xarray:ref:`io` for details.

CRACE Functions
---------------
To ensure that available geospatial encodings are always loaded from files, CRACE provides its own :ref:`convenience functions <api-data-io>`.
These are simple wrappers around the xarray equivalents with specific default settings.
Most importantly, they enable lazy evaluation with chunked dask arrays (``chunks="auto"``), and ensure that all coordinates are decoded.

Geospatial Information
----------------------
CRACE manages geospatial dataset information with the :external+odcgeo:doc:`odc-geo <intro-geobox>` and :external+rioxarray:ref:`rioxarray <getting_started>` libraries.
Coordinate reference system (CRS) information is stored as attributes to a special ``spatial_ref`` coordinate variable and can evaluated and modified via dedicated accessors.
Technically, CRACE requires a dataset to have indexed spatial coordinates (e.g., ``lat/lon``,  ``x/y``) and a qualified ``spatial_ref`` coordinate variable.
Practically, this means that ``Dataset.odc.geobox`` (see :py:attr:`odc.geo.xr.ODCExtension.geobox`) returns a :py:class:`odc.geo.geobox.GeoBox` object, and not ``None``.
