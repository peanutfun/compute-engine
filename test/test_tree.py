"""Test functions for tree operations"""

import itertools as it
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.testing as xrt
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros
from shapely.geometry import Point

from climadace.impact_funcs import (
    REGISTRY,
    FuncDefault,
    FuncLeaf,
    FuncType,
    ImpactFunctionMap,
)
from climadace.tree import (
    TreeMapper,
    TreeSplitter,
    dropna_spatial_dims,
    map_aggregate_function,
    map_impact_function,
    map_over_datasets,
    merge_tree_dset,
    split_from_geo,
    merge_by_combine,
)


@pytest.fixture
def dataset():
    return xr.Dataset(
        {"var": (["x", "y"], np.ones((3, 4), dtype="float"))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )


@pytest.fixture
def datatree(dataset):
    return xr.DataTree.from_dict(
        {
            "/": dataset.copy(deep=True),
            "/a": dataset.copy(deep=True),
            "/a/aa": dataset.copy(deep=True),
            "/b": dataset.copy(deep=True),
            "/b/1": dataset.copy(deep=True),
            "/b/2": dataset.copy(deep=True),
            "/b/3": dataset.copy(deep=True),
        }
    )


@pytest.mark.skip("Check for unaligned DataTree nodes")
def test_tree_roots():
    dt = xr.DataTree.from_dict(
        {
            "a": xr.Dataset(
                {"var1": (["x", "y"], np.ones((3, 4), dtype="float"))},
                coords={"x": np.arange(3), "y": np.arange(4)},
            ),
            "b": xr.Dataset(
                {"var2": (["a", "b"], np.ones((3, 4), dtype="float"))},
                coords={"a": np.arange(3) + 5, "b": np.arange(4) + 5},
            ),
            "a/1": xr.Dataset(
                {"var2": (["a", "b"], np.ones((3, 4), dtype="float"))},
                coords={"a": np.arange(3) + 5, "b": np.arange(4) + 5},
            ),  # Works, does not need alignment for different dimensions
            # "a/2": xr.Dataset(
            #     {"var2": (["x", "y"], np.ones((3, 4), dtype="float"))},
            #     coords={"x": np.arange(3) + 5, "y": np.arange(4) + 5},
            # ),  # Does not work, needs alignment
        }
    )


class TestMapOverDatasets:
    @pytest.fixture
    def sliced_tree(self, dataset):
        return xr.DataTree.from_dict(
            {
                "/a": dataset.sel(x=slice(0, 1)),
                "/b/1": dataset.sel(x=slice(2, 3), y=slice(0, 1)),
                "/b/2": dataset.sel(x=slice(2, 3), y=slice(2, 4)),
            }
        )

    @pytest.fixture
    def sliced_tree_zero(self, sliced_tree):
        return xr.DataTree.from_dict(
            {
                "/a": sliced_tree["/a"] * 0,
                "/b/1": sliced_tree["/b/1"] * 0,
                "/b/2": sliced_tree["/b/2"] * 0,
            }
        )

    def test_unary(self, dataset, sliced_tree, sliced_tree_zero):
        dt_unary = map_over_datasets(lambda x: x * xr.zeros_like(dataset), sliced_tree)
        xr.testing.assert_equal(dt_unary, sliced_tree_zero)

    def test_binary(self, sliced_tree, sliced_tree_zero):
        dt_binary = map_over_datasets(lambda x, y: x * y, sliced_tree, sliced_tree_zero)
        xr.testing.assert_equal(dt_binary, sliced_tree_zero)

    def test_tuple_return(self, sliced_tree, sliced_tree_zero):
        dt_eq1, dt_eq2 = map_over_datasets(
            lambda x, y: (x * 0, y + 1), sliced_tree, sliced_tree_zero
        )
        xr.testing.assert_equal(dt_eq1, sliced_tree_zero)
        xr.testing.assert_equal(dt_eq2, sliced_tree)


class TestTreeSplitter:
    @pytest.fixture
    def splitter(self, datatree):
        return TreeSplitter(tree=datatree["/b"])

    def test_init(self, dataset):
        ts = TreeSplitter(tree=dataset)
        assert isinstance(ts.tree, xr.DataTree)

    @pytest.fixture
    def splitter_with_child_nodes(self, splitter, dataset):
        splitter.child_nodes = [
            xr.DataTree(dataset, name="foo"),
            xr.DataTree(dataset, name="bar"),
        ]
        return splitter

    @pytest.mark.parametrize(
        "inplace,prune_node", list(it.product((True, False), repeat=2))
    )
    def test_result(self, splitter_with_child_nodes, dataset, inplace, prune_node):
        result = splitter_with_child_nodes.result(
            inplace=inplace, prune_node=prune_node
        )

        if inplace:
            assert result is None
            result = splitter_with_child_nodes.tree
            assert not result.is_root
        else:
            assert result is not splitter_with_child_nodes.tree
            assert result.is_root

        assert result.children == {
            "foo": xr.DataTree(dataset),
            "bar": xr.DataTree(dataset),
        }
        comparison = dataset
        if prune_node:
            comparison = dataset.drop_vars("var") if inplace else xr.Dataset()
        xr.testing.assert_equal(result.to_dataset(), comparison)


@pytest.fixture
def impf_map():
    return ImpactFunctionMap(
        {
            FuncDefault: lambda x: x * 0,
            FuncLeaf: lambda x: x,
            "/a": lambda x: x * 1.5,
            "1": lambda x: x + 1,
            "/b/3": lambda x: x + 2,
        }
    )


class TestTreeMapper:
    def test_function_to_map(self):
        def func(x):
            return x

        fmap = TreeMapper.function_to_map(func)
        assert isinstance(fmap, dict)
        assert fmap[FuncType.default] is func

    def test_map_to_map(self, impf_map):
        fmap = TreeMapper.function_to_map(impf_map)
        assert fmap is impf_map

    def test_empty_return(self, datatree):
        """Check that returned tree is empty if apply() was not called"""
        tm = TreeMapper(datatree, lambda x: x, None)
        dt = tm.result()
        assert dt.is_leaf
        assert dt.is_root
        assert not dt.has_data

    def test_unchanged_return(self, datatree):
        tm = TreeMapper(datatree, lambda x: x, None)
        tm.apply(False, False)
        dt = tm.result()
        xr.testing.assert_equal(dt, datatree)

    def test_apply(self, datatree, impf_map, dataset):
        tm = TreeMapper(datatree, impf_map, None)
        tm.apply(use_parent=False, use_merge=False)
        tree = tm.result()

        assert tree.isomorphic(datatree)
        xrt.assert_allclose(tree["/a"].to_dataset(), dataset * 1.5)
        xrt.assert_allclose(tree["/a/aa"].to_dataset(), dataset)  # Leaf
        xrt.assert_allclose(tree["/b"].to_dataset(), dataset * 0)
        xrt.assert_allclose(tree["/b/1"].to_dataset(), dataset + 1)
        xrt.assert_allclose(tree["/b/2"].to_dataset(), dataset)  # Leaf
        xrt.assert_allclose(tree["/b/3"].to_dataset(), dataset + 2)

    def test_apply_parent(self, datatree, impf_map, dataset):
        tm = TreeMapper(datatree, impf_map, None)
        tm.apply(use_parent=True, use_merge=False)
        tree = tm.result()

        assert tree.isomorphic(datatree)
        xrt.assert_allclose(tree["/a"].to_dataset(), dataset * 1.5)
        xrt.assert_allclose(tree["/a/aa"].to_dataset(), dataset * 1.5)  # Parent
        xrt.assert_allclose(tree["/b"].to_dataset(), dataset * 0)
        xrt.assert_allclose(tree["/b/1"].to_dataset(), dataset + 1)
        xrt.assert_allclose(tree["/b/2"].to_dataset(), dataset)  # Leaf
        xrt.assert_allclose(tree["/b/3"].to_dataset(), dataset + 2)

    def test_apply_merge(self, dataset):
        dt = xr.DataTree.from_dict(
            {
                "/a": dataset.copy(deep=True),
                "/b/1": dataset.copy(deep=True).sel(x=slice(0, 1)),
                "/b/2": dataset.copy(deep=True).sel(x=slice(2, 3)),
            }
        )
        impf_map = {"/b": lambda x: x}

        tm = TreeMapper(dt, impf_map, None)
        tm.apply(use_parent=False, use_merge=True)
        tree = tm.result()

        xr.testing.assert_equal(
            tree,
            xr.DataTree.from_dict(
                {"/a": None, "/b": dataset, "/b/1": None, "/b/2": None}
            ),
        )

    def test_apply_registry(self, dataset, datatree, impf_map):
        registry = {"foo": impf_map["/b/3"]}
        impf_map["/b/3"] = "foo"
        tm = TreeMapper(datatree, impf_map, registry)
        tm.apply(use_parent=False, use_merge=False)
        tree = tm.result()

        assert tree.isomorphic(datatree)
        xrt.assert_allclose(tree["/a"].to_dataset(), dataset * 1.5)
        xrt.assert_allclose(tree["/a/aa"].to_dataset(), dataset)
        xrt.assert_allclose(tree["/b"].to_dataset(), dataset * 0)
        xrt.assert_allclose(tree["/b/1"].to_dataset(), dataset + 1)
        xrt.assert_allclose(tree["/b/2"].to_dataset(), dataset)
        xrt.assert_allclose(tree["/b/3"].to_dataset(), dataset + 2)

    def test_apply_registry_errors(self, datatree):
        impf_map = {"/a": 1}
        with pytest.raises(ValueError, match="Entry needs to be a function"):
            TreeMapper(datatree, impf_map, None).apply(False, False)
        registry = {"foo": "bar"}
        with pytest.raises(KeyError, match="1"):
            TreeMapper(datatree, impf_map, registry).apply(False, False)
        registry[1] = 1
        with pytest.raises(TypeError, match="object is not callable"):
            TreeMapper(datatree, impf_map, registry).apply(False, False)


def test_map_impact_function(datatree, dataset, impf_map):
    REGISTRY["foo"] = impf_map["/b/3"]
    impf_map["/b/3"] = "foo"
    tree = map_impact_function(datatree, impf_map)

    assert tree.isomorphic(datatree)
    xrt.assert_allclose(tree["/a"].to_dataset(), dataset * 1.5)
    xrt.assert_allclose(tree["/a/aa"].to_dataset(), dataset * 1.5)
    xrt.assert_allclose(tree["/b"].to_dataset(), dataset * 0)
    xrt.assert_allclose(tree["/b/1"].to_dataset(), dataset + 1)
    xrt.assert_allclose(tree["/b/2"].to_dataset(), dataset)
    xrt.assert_allclose(tree["/b/3"].to_dataset(), dataset + 2)


def test_map_aggregate_function(dataset, impf_map):
    datatree = xr.DataTree.from_dict(
        {
            "/a": dataset.copy(deep=True),
            "/b/1": dataset.copy(deep=True).sel(x=slice(0, 1)),
            "/b/2": dataset.copy(deep=True).sel(x=slice(2, 3)),
        }
    )
    impf_map = {"/b": lambda x: x}
    tree = map_aggregate_function(datatree, impf_map)
    xr.testing.assert_equal(
        tree,
        xr.DataTree.from_dict({"/a": None, "/b": dataset, "/b/1": None, "/b/2": None}),
    )


@pytest.mark.parametrize("order", it.permutations(range(3)))
def test_merge_by_combine(dataset, order):
    # Datasets overlap, but there is only one non-NaN value at each coordinate.
    # Permutate to make sure the result is independent from dataset order.
    ds_1 = dataset.copy(deep=True).sel(x=slice(0, 1))
    ds_1["var"].loc[{"x": [1, 1], "y": [1, 2, 3]}] = np.nan
    ds_2 = dataset.copy(deep=True).sel(x=slice(1, 1))
    ds_2["var"].loc[{"y": [0, 3]}] = np.nan
    ds_3 = dataset.copy(deep=True).sel(x=slice(1, 2))
    ds_3["var"].loc[{"x": [1, 1], "y": [0, 1, 2]}] = np.nan

    ds = [ds_1, ds_2, ds_3]
    ds = merge_by_combine(*(ds[idx] for idx in order))
    xr.testing.assert_identical(ds, dataset)


class TestTreeMerge:
    @pytest.fixture
    def datatree(self, dataset):
        return xr.DataTree.from_dict(
            {
                "/a": dataset.sel(x=slice(0, 1)),
                "/b/1": dataset.sel(x=slice(2, 3), y=slice(0, 1)),
                "/b/2": dataset.sel(x=slice(2, 3), y=slice(2, 4)),
            }
        )

    @pytest.mark.parametrize("inplace", (True, False))
    def test_merge_default(self, dataset, datatree, inplace):
        merged = merge_tree_dset(datatree, inplace=inplace)
        if inplace:
            assert merged is None
            merged = datatree
        else:
            assert merged is not datatree
        xrt.assert_identical(merged.to_dataset(), dataset)

    def test_merge_drop_subtree(self, dataset, datatree):
        # Dropping should mean no children
        merged = merge_tree_dset(datatree, drop_subtree=True)
        assert merged.children == {}

        # Alignment error
        with pytest.raises(ValueError) as err:
            merged = merge_tree_dset(datatree, drop_subtree=False)
        assert "Use 'drop-subtree=True'" in str(err)

        # No alignment error
        dt = xr.DataTree.from_dict({
            "/a": dataset.where(dataset["x"] < 1), "/b": dataset
        })
        merged = merge_tree_dset(dt, drop_subtree=False)
        dt.ds = dataset
        xr.testing.assert_identical(merged, dt)

    @pytest.mark.parametrize("inplace", (True, False))
    def test_merge_leaf(self, dataset, inplace):
        dt = xr.DataTree(dataset, name="foo")
        merged = merge_tree_dset(dt, inplace=inplace)

        xr.testing.assert_identical(merged, dt)
        if inplace:
            assert merged is dt
        else:
            assert merged is not dt

    def test_merge_tree_dset_overlap(self, dataset):
        ds_a = dataset.copy(deep=True).sel(x=slice(0, 1))
        ds_a["var"].loc[{"x": 1, "y": 0}] = np.nan  # NOTE: Overlap with NaN is OK!
        ds_b2 = dataset.copy(deep=True).sel(x=slice(1, 3), y=slice(1, 4))
        ds_b2["var"].loc[{"x": 2, "y": 1}] = np.nan  # NOTE: Overlap with NaN is OK!
        dt = xr.DataTree.from_dict(
            {
                "/a": ds_a,
                "/b/1": dataset.sel(x=slice(1, 3), y=slice(0, 1)),
                "/b/2": ds_b2,
            }
        )
        merged = merge_tree_dset(dt)
        assert merged is not dt
        xrt.assert_identical(merged.to_dataset(), dataset)

    def test_merge_tree_dset_errors(self, dataset):
        # Check for hollow tree
        with pytest.raises(ValueError, match="Tree must be hollow") as exc:
            merge_tree_dset(
                xr.DataTree.from_dict({"/a": dataset, "/a/1": dataset, "/a/2": dataset})
            )
            assert "Tree must be hollow" in str(exc.value)

        # Check for overwrite
        with pytest.raises(ValueError, match="Merging would overwrite"):
            merge_tree_dset(
                xr.DataTree.from_dict({"/": dataset, "/1": dataset, "/2": dataset}),
                overwrite=False,
            )


@pytest.fixture
def geo_dataset():
    return xr_zeros(
        GeoBox.from_bbox((0, 0, 3, 4), "EPSG:4326", resolution=1)
    ).to_dataset(name="data")


@pytest.fixture
def geo_series():
    return gpd.GeoSeries(
        [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1), Point(0, 2)]
    ).transform(lambda x: x + 0.5)


@pytest.fixture
def geo_dataframe(geo_series):
    return gpd.GeoDataFrame(
        {
            "cat": [1, 1, 2, 2, 2],
            "geometry": geo_series.buffer(0.5, cap_style="square"),
        },
        crs="EPSG:4326",
    )


def test_dropna_spatial_dims(geo_dataset):
    # Add non-geo coordinate
    ds = xr.concat([geo_dataset, geo_dataset], dim=pd.Index([0, 1], name="z"))

    # Non-geo coordinates (z) are ignored
    ds["data"][1, ...] = np.nan
    xr.testing.assert_equal(dropna_spatial_dims(ds), ds)

    # Geo-coordinates (x) are dropped
    ds["data"][..., 0] = np.nan
    ds["data"][0, 1, 0] = np.nan  # Should not be dropped
    xr.testing.assert_equal(dropna_spatial_dims(ds), ds.sel(longitude=slice(1.5, 2.5)))


# --- split_from_geo --- #


class TestSplitFromGeo:
    @pytest.fixture(autouse=True)
    def assert_split_1_2(self, geo_dataset):
        def assertion(tree, gdf):
            for name, group in gdf.groupby("cat"):
                ds = geo_dataset.copy(deep=True)
                ds["data"][...] = np.nan
                centr = group.geometry.centroid
                for x, y in zip(centr.x, centr.y):
                    ds.loc[{"longitude": x, "latitude": y}] = 0
                node_name = str(name)
                assert node_name in [node.name for node in tree.descendants]
                xrt.assert_equal(tree[node_name].to_dataset(), ds)

        return assertion

    @pytest.mark.parametrize("groupby_kws", [None, {"by": "cat"}])
    def test_split(self, groupby_kws, geo_dataset, geo_dataframe, assert_split_1_2):
        dt = split_from_geo(geo_dataset, geo_dataframe, groupby_kws=groupby_kws)
        assert isinstance(dt, xr.DataTree)
        assert dt.is_hollow
        assert sorted(dict(dt.subtree_with_keys).keys()) == sorted([".", "1", "2"])
        assert not dt.has_data
        assert_split_1_2(dt, geo_dataframe)

    def test_raise_infer_groupby(self, geo_dataset, geo_dataframe):
        gdf = geo_dataframe.copy()
        gdf["col"] = "foo"
        with pytest.raises(ValueError) as exc:
            split_from_geo(geo_dataset, gdf)
        assert "GeoDataFrame must have exactly one other column" in str(exc)

    def test_prune_node(self, geo_dataset, geo_dataframe, assert_split_1_2):
        dt = split_from_geo(geo_dataset, geo_dataframe, prune_node=False)
        assert not dt.is_hollow
        xrt.assert_identical(dt.to_dataset(), geo_dataset)
        assert_split_1_2(dt, geo_dataframe)

    def test_keep_exterior(
        self, geo_dataset, geo_dataframe, geo_series, assert_split_1_2
    ):
        dt = split_from_geo(
            geo_dataset, geo_dataframe, keep_exterior=True
        )
        assert sorted(dict(dt.subtree_with_keys).keys()) == sorted(
            [".", "_exterior", "1", "2"]
        )
        assert_split_1_2(dt, geo_dataframe)
        ext = geo_dataset.copy(deep=True)
        for x, y in zip(geo_series.x, geo_series.y):
            ext.loc[{"longitude": x, "latitude": y}] = np.nan
        xrt.assert_equal(dt["_exterior"].to_dataset(), ext)

    def test_return_unchanged(self, geo_dataset):
        dt = split_from_geo(geo_dataset, gpd.GeoDataFrame())
        xrt.assert_identical(dt, xr.DataTree())
        dt = split_from_geo(geo_dataset, gpd.GeoDataFrame(), prune_node=False)
        xrt.assert_identical(dt, xr.DataTree(geo_dataset))

    def test_pass_groupby_kwargs(self, geo_dataset, geo_dataframe, geo_series):
        geo_dataframe.loc[4, "cat"] = np.nan
        dt = split_from_geo(
            geo_dataset, geo_dataframe, groupby_kws={"by": "cat", "dropna": False}
        )
        assert sorted(dict(dt.subtree_with_keys).keys()) == sorted(
            [".", "1.0", "2.0", "nan"]
        )
        dt_nan = geo_dataset.copy(deep=True)
        dt_nan["data"][...] = np.nan
        dt_nan["data"].loc[
            {"longitude": geo_series.iloc[4].x, "latitude": geo_series.iloc[4].y}
        ] = 0
        xrt.assert_equal(dt["nan"].to_dataset(), dt_nan)
