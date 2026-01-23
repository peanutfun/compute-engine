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
