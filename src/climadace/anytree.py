"""Operations on trees"""


import geopandas as gpd
import odc.geo.converters
import odc.geo.xr  # noqa: F401
import xarray as xr
from anytree import LevelOrderIter, Node
from odc.geo.geom import Geometry

from .funcs import DatasetOrArray


class DatasetNode(Node):
    """A node with a clearly specified attribute"""

    def __init__(self, *args, dset: xr.Dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.dset = dset


def dropna_spatial_dims(data: DatasetOrArray) -> DatasetOrArray:
    """Drop spatial dims where all coordinates are NaN"""
    spatial_dims = data.odc.spatial_dims
    for dim in spatial_dims:
        data = data.dropna(dim, how="all")
    return data


def mask_dataset(
    data: DatasetOrArray, geometry: Geometry, dropna: bool = False, **mask_kwargs
) -> DatasetOrArray:
    """Mask data using a geometry and drop coordinates without values"""
    data = data.odc.mask(geometry, **mask_kwargs)
    if dropna:
        data = dropna_spatial_dims(data)
    return data


def split_dataset(
    data: xr.Dataset,
    geometry: Geometry,
    # return_inverted: bool = False,
    **mask_kwargs,
) -> xr.Dataset:
    """Split a dataset using a geometry for masking"""
    # invert = mask_kwargs.pop("invert", False)
    data_masked = mask_dataset(data, geometry, **mask_kwargs)

    # if return_inverted:
    #     data_masked_inverted = mask_dataset(
    #         data, geometry, invert=not invert, **mask_kwargs
    #     )
    #     return (data_masked, data_masked_inverted)

    return data_masked


def split_from_geometry():
    pass


def split_from_gdf(
    node: xr.Dataset | DatasetNode,
    gdf: gpd.GeoDataFrame,
    groupy: str,
    keep_exterior: bool = False,
    prune_node: bool = True,
    dropna: bool = False,
    **mask_kwargs,
):
    """Split a dataset into subsets and return them as tree leaves"""
    if not isinstance(node, DatasetNode):
        node = DatasetNode("root", dset=node)

    child_nodes = []
    for group, data in gdf.groupby(groupy):
        # Merge geometries
        geo = odc.geo.geom.unary_union(odc.geo.converters.from_geopandas(data.geometry))
        child_nodes.append(
            DatasetNode(
                group, dset=split_dataset(node.dset, geo, dropna=dropna, **mask_kwargs)
            )
        )

    # Maybe return outside
    if keep_exterior:
        geo = odc.geo.geom.unary_union(odc.geo.converters.from_geopandas(gdf.geometry))
        invert = mask_kwargs.pop("invert", False)
        child_nodes.append(
            DatasetNode(
                "_exterior",
                dset=split_dataset(
                    node.dset, geo, invert=not invert, dropna=dropna, **mask_kwargs
                ),
            )
        )

    # Set child nodes, unset parent
    node.children = child_nodes

    # Maybe remove parent dataset
    if prune_node:
        node.dset = xr.Dataset()

    return node


def merge_tree_dset(root: DatasetNode):
    """Merge the tree leave datasets into the root node

    Todo
    ----
    Maybe we can just call combine_by_coords on all leaves?
    """
    if root.is_leaf:
        return root

    # Collect nodes that do not have children, iteratively
    leaf_nodes = []
    for node in LevelOrderIter(root, maxlevel=2):
        print(node.name)
        if node is root:
            continue
        if not node.is_leaf:
            merge_tree_dset(node)
        else:
            leaf_nodes.append(node)

    # Merge the leaf datasets and drop them
    root.dset = xr.merge([node.dset for node in leaf_nodes], compat="no_conflicts")
    root.children = []
    return root
