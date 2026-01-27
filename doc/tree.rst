.. _data-tree:

====================
Data Tree Operations
====================

After aligning hazard and exposure data, you may run an impact calculation and reduce the result.
However, a more granular approach to impact function application and reductions is required.
For example, vulnerability might vary on a smaller spatial or temporal scale than the scale of the datasets.
Therefore, CRACE offers tools to "split" the exposure dataset into a hierarchical :py:class:`xarray.DataTree` structure.

Importantly, :py:class:`xarray.DataTree` is intended to be a hierarchical structure of datasets.
Each node may store a dataset, and have one parent and multiple children.
The tree automatically aligns child nodes with their parents.
Since we want to split the data, this is a feature we want to avoid.

The CRACE :ref:`split functions <api-data-tree>` work as follows:

- If the target dataset is not a data tree node, place it into a node.
- Split the dataset according to the function.
- Remove the dataset from the target node.
- Place the split datasets into new nodes.
- Register these nodes as children of the target node.

If you want to split along a single coordinate, you may use :py:func:`~crace.split_from_groupby` or :py:func:`~crace.split_from_groupby_bins`, which split a dataset using group-by operations and place the resulting groups in child nodes.

Split With Geometry
-------------------
Spatial groupings (think: countries, regions) are usually to complicated to express in bins along coodinates.
Therefore, CRACE provides :py:func:`~crace.split_from_geo`, which takes a :py:class:`geopandas.GeoDataFrame` as parameter and as base for the splitting.
The data frame needs a valid geometry column and at least one other column to identify group labels.
CRACE will call a group-by operation on the data frame and call :py:func:`~crace.mask_dataset` with the union of grouped geometries on the original dataset.
Each child node of the data tree will then contain the masked dataset for the respective group.

All ``split`` methods have parameters ``inplace`` and ``prune_node``:

If ``inplace`` is set to ``True``, the input node is modified and the functions return ``None``.
If set to ``False``, a shallow copy of the input node is created and the functions returns this copy with the new child nodes attached.
The original node is not modified.

If ``prune_node`` is set to ``True``, the original dataset is removed from the node before attaching the split child nodes.
This allows for "clipped" coordinates in the child nodes, because an empty parent node means that the child nodes need not align with it.

Merging Child Nodes
-------------------
The inverse operation of node splitting is the merging of child nodes into a parent node with :py:func:`~crace.merge_tree_dset`.
