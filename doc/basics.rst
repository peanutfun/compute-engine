======
Primer
======

Climate risk assessment is conducted by combining three sources of data:

- Hazard: A geospatial dataset defining the hazard intensity at pixels or points.
- Exposure: A geospatial dataset defining exposed goods, assets, people at pixels or points.
- Vulnerability: The ratio of affected exposure as function of the hazard intensity at a particular location.

When combined, CRACE produces and Impact dataset defining the exposure impacted from the hazard.

Workflow
--------

#. :ref:`Loading <data-io>` or generating hazard and exposure datasets.
#. :ref:`Aligning <data-align>` hazard and exposure datasets.
#. *Optional:* Generating a data tree structure for granular assignment of impact functions and reductions.
#. Computing the impact and applying reduction operations.
