"""Operations on trees"""

from typing import Any, Callable, Hashable, Mapping

import geopandas as gpd
import numpy as np
import odc.geo.converters
import odc.geo.geom
import odc.geo.xr  # noqa: F401
import xarray as xr

from .impact_funcs import (
    REGISTRY,
    FunctionMap,
    FuncType,
)
from .types import DatasetFunction, DatasetOrArray


def filter_dsets(
    dset: xr.Dataset,
    *dsets: xr.Dataset,
    filterfunc: Callable[[xr.Dataset], bool],
) -> xr.Dataset | None | tuple[xr.Dataset | None, ...]:
    """Return datasets or None if filterfunc returns False"""

    def ds_or_none(ds: xr.Dataset) -> xr.Dataset | None:
        if not filterfunc(ds):
            return None
        return ds

    if not dsets:
        return ds_or_none(dset)

    return tuple(ds_or_none(ds) for ds in (dset,) + dsets)


def map_over_datasets(
    func,
    *args,
    kwargs: Mapping[str, Any] | None = None,
    filterfunc: Callable[[xr.Dataset], bool] = lambda x: bool(x.data_vars),
):
    """map_over_datasets but omit results for which filterfunc is False"""

    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)
        if not isinstance(results, tuple):
            results = (results,)
        return filter_dsets(*results, filterfunc=filterfunc)

    return xr.map_over_datasets(wrapper, *args, kwargs=kwargs)


def tree_is_empty(node: xr.DataTree) -> bool:
    """Check if the node and all children are empty"""
    return all(node.is_empty for node in node.subtree)


def dropna_spatial_dims(data: DatasetOrArray) -> DatasetOrArray:
    """Drop spatial dims where all coordinates are NaN"""
    spatial_dims = data.odc.spatial_dims
    for dim in spatial_dims:
        data = data.dropna(dim, how="all")
    return data


def mask_dataset(
    data: DatasetOrArray,
    geometry: odc.geo.geom.Geometry,
    dropna: bool = False,
    **mask_kwargs,
) -> DatasetOrArray:
    """Mask data using a geometry and possibly drop coordinates without values"""
    data = data.odc.mask(geometry, **mask_kwargs)
    if dropna:
        data = dropna_spatial_dims(data)
    return data


def split_from_groupby_bins(
    node: xr.Dataset | xr.DataTree,
    prune_node: bool = True,
    inplace: bool = False,
    **groupby_bins_kwargs,
):
    splitter = TreeSplitter(tree=node)
    splitter.split_from_groupby_bins(**groupby_bins_kwargs)
    return splitter.result(inplace=inplace, prune_node=prune_node)


def split_from_groupby(
    node: xr.Dataset | xr.DataTree,
    prune_node: bool = True,
    inplace: bool = False,
    **groupby_kwargs,
) -> xr.DataTree | None:
    """Split using groupby"""
    splitter = TreeSplitter(tree=node)
    splitter.split_from_groupby(**groupby_kwargs)
    return splitter.result(inplace=inplace, prune_node=prune_node)


# TODO: Low resolution mode (convert CRS of gdf, not geometry)
def split_from_geo(
    node: xr.Dataset | xr.DataTree,
    gdf: gpd.GeoDataFrame,
    *,
    high_precision: bool = False,
    keep_exterior: bool = False,
    prune_node: bool = True,
    inplace: bool = False,
    groupby_kws: Mapping | None = None,
    mask_kws: Mapping | None = None,
) -> xr.DataTree | None:
    """Split a dataset into subsets and return them as tree leaves"""
    splitter = TreeSplitter(tree=node)
    splitter.split_from_dataframe(
        gdf=gdf,
        keep_exterior=keep_exterior,
        high_precision=high_precision,
        groupby_kws=groupby_kws,
        mask_kws=mask_kws,
    )
    return splitter.result(inplace=inplace, prune_node=prune_node)


class TreeSplitter:
    def __init__(self, tree: xr.DataTree | xr.Dataset):
        if not isinstance(tree, xr.DataTree):
            tree = xr.DataTree(dataset=tree)
        self.tree = tree
        self.child_nodes: list[xr.DataTree] = []

    def split_from_dataframe(
        self,
        gdf: gpd.GeoDataFrame,
        keep_exterior: bool,
        high_precision: bool = False,
        groupby_kws: Mapping[str, Any] | None = None,
        mask_kws: Mapping[str, Any] | None = None,
    ):
        if gdf.empty:
            return
        gdf = gdf.copy(deep=False)
        mask_kws = {"all_touched": False} | (
            dict(mask_kws) if mask_kws is not None else {}
        )
        print(mask_kws)

        # Geospatial data of the dataset
        geobox = self.tree.to_dataset().odc.geobox
        ds_extent = geobox.extent
        crs = geobox.crs
        resolution = np.min(np.abs(geobox.resolution.xy))

        if not high_precision:
            gdf = gdf.to_crs(crs)

        # Infer groupby kwargs
        if groupby_kws is None:
            if len(gdf.columns) > 2:
                raise ValueError(
                    "GeoDataFrame must have exactly one other column than the geomtry "
                    "column to infer groupby key"
                )
            groupby_kws = {"by": gdf.drop(columns=gdf.active_geometry_name).columns[0]}

        # Convert to odc geometries
        odc_geometry_col = "_" + (gdf.active_geometry_name or "geometry") + "_odc"
        gdf[odc_geometry_col] = odc.geo.converters.from_geopandas(gdf.geometry)

        # Interate over groups
        for group, data in gdf.groupby(**groupby_kws):
            # Merge group geometries
            geo_interior = odc.geo.geom.unary_union(data[odc_geometry_col])
            assert geo_interior is not None
            if high_precision:
                geo_interior = geo_interior.to_crs(crs, resolution=resolution)
            if not odc.geo.geom.intersects(geo_interior, ds_extent):
                continue

            self.child_nodes.append(
                xr.DataTree(
                    dataset=mask_dataset(self.tree.dataset, geo_interior, **mask_kws),
                    name=str(group),
                )
            )

        # Maybe return outside
        if keep_exterior:
            geo_exterior = odc.geo.geom.unary_union(gdf[odc_geometry_col])
            assert geo_exterior is not None
            if high_precision:
                geo_exterior = geo_exterior.to_crs(crs, resolution=resolution)
            if not odc.geo.geom.intersects(geo_exterior, ds_extent):
                return

            invert = mask_kws.pop("invert", False)
            self.child_nodes.append(
                xr.DataTree(
                    dataset=mask_dataset(
                        self.tree.dataset,
                        geo_exterior,
                        invert=not invert,
                        **mask_kws,
                    ),
                    name="_exterior",
                )
            )

    def split_from_groupby(self, **groupby_kwargs):
        self.child_nodes = self._nodes_from_dsgroupby(
            self.tree.dataset.groupby(**groupby_kwargs)
        )

    def split_from_groupby_bins(self, **groupby_bins_kwargs):
        self.child_nodes = self._nodes_from_dsgroupby(
            self.tree.dataset.groupby_bins(**groupby_bins_kwargs)
        )

    @staticmethod
    def _nodes_from_dsgroupby(groupby) -> list[xr.DataTree]:
        return [xr.DataTree(ds, name=str(label)) for label, ds in groupby]

    def result(self, inplace: bool, prune_node: bool) -> xr.DataTree | None:
        if not inplace:
            return xr.DataTree.from_dict(
                {
                    "/": xr.DataTree(
                        dataset=self.tree.dataset if not prune_node else None,
                        name=self.tree.name,
                    )
                }
                | {f"/{child.name}": child for child in self.child_nodes}
            )

        if prune_node:
            self.tree.ds = None
        self.tree.children = {child.name: child for child in self.child_nodes}


# TODO: Option: Use closest dsets (need not be hollow)
def merge_tree_dset(
    root: xr.DataTree, drop_subtree: bool = True, overwrite: bool = False
) -> xr.DataTree:
    """Merge the tree leaf datasets into the root node

    Todo
    ----
    Maybe we can just call combine_by_coords on all leaves?
    """
    if root.is_leaf:
        return root
    if root.has_data and not overwrite:
        raise ValueError("Merging would overwrite existing root dataset!")
    if not root.is_hollow:
        raise ValueError("Tree must be hollow for merging!")
    root = root.copy()  # Shallow copy

    # Collect nodes that do not have children, iteratively
    leaf_nodes = []
    for node in root.children.values():
        if not node.is_leaf:
            leaf_nodes.append(merge_tree_dset(node))
        else:
            leaf_nodes.append(node)

    # Merge the leaf datasets and possibly drop them
    if drop_subtree:
        root.children = {}
    root.update(
        xr.merge(
            [node.dataset for node in leaf_nodes],
            join="outer",
            compat="no_conflicts",  # Very slow :(
            # compat="override",  # Error-prone, but MUCH faster!
        )
        # xr.concat(
        #     [node.dataset for node in leaf_nodes], dim="_concat", join="outer"
        # ).sum(dim="_concat", skipna=True, min_count=1)
    )
    return root


def map_impact_function(
    tree: xr.DataTree, func_map: FunctionMap | DatasetFunction
) -> xr.DataTree:
    mapper = TreeMapper(tree=tree, func_map=func_map, registry=REGISTRY)
    mapper.apply(use_parent=True, use_merge=False)
    return mapper.result()


def map_aggregate_function(
    tree: xr.DataTree, func_map: FunctionMap | DatasetFunction
) -> xr.DataTree:
    mapper = TreeMapper(tree=tree, func_map=func_map, registry=None)
    mapper.apply(use_parent=False, use_merge=True)
    return mapper.result()


class TreeMapper:
    """Class for mapping impact function (maps) to trees and returning the result"""

    def __init__(
        self,
        tree: xr.DataTree,
        func_map: DatasetFunction | Mapping[Hashable, DatasetFunction | Hashable],
        registry: Mapping[Hashable, DatasetFunction] | None,
    ):
        """Initialize the mapper"""
        self._tree = tree
        self._func_map = self.function_to_map(func_map)
        self._dsets = {}
        self._registry = registry

    @staticmethod
    def function_to_map(
        func_or_map: DatasetFunction | Mapping[Hashable, DatasetFunction | Hashable],
    ) -> Mapping[Hashable, DatasetFunction | Hashable]:
        """Promote a single impact function to a function map with default entry"""
        if callable(func_or_map):
            return FunctionMap({FuncType.default: func_or_map})
        return func_or_map

    def apply(self, use_parent: bool, use_merge: bool):
        """Apply the function map to the tree"""
        for path, node in self._tree.subtree_with_keys:
            # Find suitable function
            func = self._get_func(node, use_parent=use_parent, use_default=True)
            if func is None:
                self._dsets[path] = None
                continue

            # Merge leafs, if the node does not have data
            if not node.has_data:
                if not use_merge:
                    continue
                node = merge_tree_dset(root=node, drop_subtree=True)

            # Apply function
            self._dsets[path] = func(node.dataset)

    # TODO: What about indicating levels?
    def _get_map_entry(
        self, node: xr.DataTree, use_parent: bool, use_default: bool
    ) -> Callable | Hashable | None:
        """Retrieve the impact function map entry for a specific node"""
        path = node.path
        # Exact path
        if path in self._func_map:
            return self._func_map[path]
        # Name (last path component)
        if node.name is not None and node.name in self._func_map:
            return self._func_map[node.name]
        # Any parent path (but not reverting to default here!)
        if use_parent and not node.is_root:
            entry = self._get_func(
                node.parent, use_parent=use_parent, use_default=False
            )
            if entry is not None:
                return entry
        # Maybe return leaf func
        if node.is_leaf and FuncType.leaf in self._func_map:
            return self._func_map[FuncType.leaf]
        # Return default or None
        if use_default:
            return self._func_map.get(FuncType.default, None)
        return None

    def _get_func(
        self, node: xr.DataTree, use_parent: bool, use_default: bool
    ) -> Callable | None:
        """Retrieve an impact function (or None) for a specific node"""
        entry = self._get_map_entry(
            node=node, use_parent=use_parent, use_default=use_default
        )

        # Try to retrieve registry item
        if entry is not None and not callable(entry):
            if self._registry is not None:
                return self._registry[entry]

            raise ValueError("Entry needs to be a function")

        return entry

    # TODO: Add prune?
    def result(self):
        """Return the new tree (must apply first)"""
        return xr.DataTree.from_dict(self._dsets)
