"""Operations on trees"""

from typing import Mapping, Any, TypeAlias, Callable, Hashable
from pathlib import PurePath
from functools import partial
from typing import overload

import pandas as pd
import xarray as xr
import odc.geo.xr  # noqa: F401
import odc.geo.converters

# from odc.geo.geom import Geometry, intersects
import odc.geo.geom
from odc.geo.geobox import GeoBox
import geopandas as gpd

from .types import DatasetOrArray, DatasetFunction
from .impact_funcs import (
    ImpactFunctionMap,
    FuncDefault,
    FuncLeaf,
    FunctionMap,
    FuncType,
    REGISTRY,
)


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
    node: xr.Dataset | xr.DataTree, prune_node: bool = True, **groupby_bins_kwargs
):
    if not isinstance(node, xr.DataTree):
        node = xr.DataTree(dataset=node)
    child_nodes = _nodes_from_dsgroupby(
        node.dataset.groupby_bins(**groupby_bins_kwargs)
    )

    # Create new tree
    return xr.DataTree.from_dict(
        {
            "/": xr.DataTree(
                dataset=node.dataset if not prune_node else None, name=node.name
            )
        }
        | {f"/{child.name}": child for child in child_nodes}
    )


def split_from_groupby(
    node: xr.Dataset | xr.DataTree, prune_node: bool = True, **groupby_kwargs
) -> xr.DataTree:
    """Split using groupby"""
    if not isinstance(node, xr.DataTree):
        node = xr.DataTree(dataset=node)
    child_nodes = _nodes_from_dsgroupby(node.dataset.groupby(**groupby_kwargs))

    # Create new tree
    return xr.DataTree.from_dict(
        {
            "/": xr.DataTree(
                dataset=node.dataset if not prune_node else None, name=node.name
            )
        }
        | {f"/{child.name}": child for child in child_nodes}
    )


def _nodes_from_dsgroupby(groupby) -> list[xr.DataTree]:
    return [xr.DataTree(ds, name=group) for ds, group in groupby]


def split_from_geo(
    node: xr.Dataset | xr.DataTree,
    gdf: gpd.GeoDataFrame,
    *,
    groupby: str | Mapping[str, Any] = "",
    keep_exterior: bool = False,
    prune_node: bool = True,
    dropna: bool = False,
    **mask_kwargs,
) -> xr.DataTree:
    """Split a dataset into subsets and return them as tree leaves"""
    if not isinstance(node, xr.DataTree):
        node = xr.DataTree(dataset=node)
    if gdf.empty:
        return node
    gdf = gdf.copy(deep=False)

    # Sanitize groupby kwargs
    groupby_kwargs = {}
    if isinstance(groupby, str):
        if groupby == "":
            if len(gdf.columns) > 2:
                raise ValueError(
                    "GeoDataFrame must have exactly one other column than the geomtry "
                    "column to infer groupby key"
                )
            groupby = gdf.drop(columns=gdf.active_geometry_name).columns[0]
        groupby_kwargs["by"] = groupby
    else:
        groupby_kwargs = groupby

    # Convert to odc geometries
    odc_geometry_col = "_" + (gdf.active_geometry_name or "geometry") + "_odc"
    gdf[odc_geometry_col] = odc.geo.converters.from_geopandas(gdf.geometry)

    # Interate over groups
    child_nodes = []
    ds_extent = node.to_dataset().odc.geobox.extent
    for group, data in gdf.groupby(**groupby_kwargs):
        # Merge geometries
        geo_interior = odc.geo.geom.unary_union(data[odc_geometry_col])
        if not odc.geo.geom.intersects(geo_interior, ds_extent):
            continue

        child_nodes.append(
            xr.DataTree(
                dataset=mask_dataset(
                    node.dataset, geo_interior, dropna=dropna, **mask_kwargs
                ),
                name=str(group),
            )
        )

    # Maybe return outside
    if keep_exterior:
        geo_exterior = odc.geo.geom.unary_union(gdf[odc_geometry_col])
        invert = mask_kwargs.pop("invert", False)
        child_nodes.append(
            xr.DataTree(
                dataset=mask_dataset(
                    node.dataset,
                    geo_exterior,
                    invert=not invert,
                    dropna=dropna,
                    **mask_kwargs,
                ),
                name="_exterior",
            )
        )

    # Create new tree
    return xr.DataTree.from_dict(
        {
            "/": xr.DataTree(
                dataset=node.dataset if not prune_node else None, name=node.name
            )
        }
        | {f"/{child.name}": child for child in child_nodes}
    )


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
            [node.dataset for node in leaf_nodes], join="outer", compat="no_conflicts"
        )
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
        if node.name in self._func_map:
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
