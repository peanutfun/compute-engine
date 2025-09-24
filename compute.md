# Grid Computation

## Steps

1. ~~Hazard and exposure must be data arrays (unclear which variable to choose otherwise)~~ *Datatree stores Datasets anyways. Use arrays for single-var datasets, and datasets otherwise (users must take care of naming)*
2. Facilitate loading data
   1. Open datasets, but use data array if only a single variable
   2. Maybe load sparse, maybe load dense -> pertain information for saving later!
4. Align non-spatial coordinates:
   1. Select or resample times
5. Reproject hazard
   1. Rename dimensions of reprojected hazard to those of exposure
   2. ~~IF `resampling="nearest"`, apply the impact function BEFORE resampling (should save computation time)~~ *This does not actually seem to save time. Applying the impact function is much cheaper than resampling.*
6. Create hazard data tree:~~This creates a selection, possibly reducing the amout of data~~ *Without dropna, this only masks, but does not drop coordinates*
7. Calculate impact with impact function map
8. Calculate aggregates with aggregate map
   1. Use function map without compulsory default/leaf: If node has no dset, call `merge_tree_dset` on the node, then call the function on it, too. Needs new function map and additional keywords to `map_over_datatree`.
