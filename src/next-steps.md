# Next Steps

* Do not store DataTree, this is too inefficient.
* If we do not interpolate, intermediate stores are inefficient.
* Even if hazard AND exposure fit into memory (when sparse), the aligned arrays may not!
  - Need to find a way to guess that. Maybe try to sparsify after aligning?
* `dropna` requires dense arrays, therefore sparse should be applied after aligning
* sanity check for geometries in `split_tree` before grouping/iterating

* Issue in `sparse`: Most efficient way to create sparse xarray from data?
  - Opening with dask and then casting to sparse probably requires to run over data twice, at least all NaNs will be checked explicitly.
  - BUT: Some operations require casting to dense anyway, so we cannot really make use of more efficient loading from disk?
