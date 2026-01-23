.. _data-align:

=============
Aligning Data
=============

When performing n-ary computations, xarray requires datasets to align.
Since coordinates can be attached to dimensions, xarray not only requires that the datasets have compatible dimension sizes (in a ``numpy`` fashion), but also that attached coordinates match.
Together with geospatial information, we use this feature to align hazard and exposure objects.

All alignment operations are conveniently wrapped by the :py:func:`crace.align` function.
Alignment includes all dimensions that are present in both datasets, identified by their name.

Spatial Align
-------------
The exposure is our data of interest.
Therefore, the hazard data is spatially interpolated onto the exposure.
In the most simple form, nearest neighbor matching is performed by xarray.
(Bi-)linear interpolation and other options are provided by the interpolation mechanisms of odc.geo.

.. note::

    Spatial dimensions in the datasets need not have the same names.
    As long as they have names compatible to the `odc.geo` definitions, they can be correctly identified.
    After aligning, the spatial dimensions in the hazard dataset will have received the names of those in the exposure dataset.

Event Align
-----------
Other shared dimensions of the datasets most likely relate to a definition of the events (think of a date and/or time of the event).
Since the hazard data is assumed to be event-wise, and thus of a shorter-term nature than the exposure data, the exposure will be interpolated onto the hazard for these dimensions.
:py:func:`crace.align` lets you choose interpolation methods for individual dimensions to be aligned.

Broadcasting
------------
Dimensions only present in one of the datasets are broadcast according to the xarray rules.
The resulting impact dataset will have the union of all non-aligned dimensions.
The coordinates of these dimensions (if present) will not be modified.
