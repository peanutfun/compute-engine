"""Test functions for tree operations"""

import pytest
import xarray as xr
import numpy as np
import xarray.testing as xrt
import geopandas as gpd
from shapely.geometry import Point
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_zeros

from unittest.mock import patch


from climadace.tree import (
    map_over_datatree,
    merge_tree_dset,
    split_from_geo,
    dropna_spatial_dims,
    map_over_datasets,
)
from climadace.impact_funcs import ImpactFunctionMap, FuncDefault, FuncLeaf


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
    print(list(dt["a"].to_dataset().data_vars.keys()))
    assert False


def test_tree_dset_arithmetic(datatree, dataset):
    dt = xr.DataTree.from_dict(
        {
            "/a": dataset.sel(x=slice(0, 1)),
            "/b/1": dataset.sel(x=slice(2, 3), y=slice(0, 1)),
            "/b/2": dataset.sel(x=slice(2, 3), y=slice(2, 4)),
        }
    )
    dt_new = dt.map_over_datasets(
        lambda x: np.multiply(*xr.align(x, xr.zeros_like(dataset), join="left"))
    )
    print(dt_new)
    assert False


def test_map_over_datatree(datatree, dataset):
    impf = ImpactFunctionMap(
        {
            FuncDefault: lambda x: x * 0,
            FuncLeaf: lambda x: x,
            "/a": lambda x: x * 1.5,
            "1": lambda x: x + 1,
            "/b/3": lambda x: x + 2,
        }
    )
    tree = map_over_datatree(impf, datatree)

    assert tree.isomorphic(datatree)
    xrt.assert_allclose(tree["/a"].to_dataset(), dataset * 1.5)
    xrt.assert_allclose(tree["/a/aa"].to_dataset(), dataset * 1.5)
    xrt.assert_allclose(tree["/b"].to_dataset(), dataset * 0)
    xrt.assert_allclose(tree["/b/1"].to_dataset(), dataset + 1)
    xrt.assert_allclose(tree["/b/2"].to_dataset(), dataset)
    xrt.assert_allclose(tree["/b/3"].to_dataset(), dataset + 2)


def test_merge_tree_dset(dataset):
    dt = xr.DataTree.from_dict(
        {
            "/a": dataset.sel(x=slice(0, 1)),
            "/b/1": dataset.sel(x=slice(2, 3), y=slice(0, 1)),
            "/b/2": dataset.sel(x=slice(2, 3), y=slice(2, 4)),
        }
    )
    merged = merge_tree_dset(dt)
    assert merged is not dt
    xrt.assert_identical(merged.to_dataset(), dataset)

    # Check for hollow tree
    with pytest.raises(RuntimeError) as exc:
        merge_tree_dset(
            xr.DataTree.from_dict({"/a": dataset, "/a/1": dataset, "/a/2": dataset})
        )
        assert "Tree must be hollow" in str(exc.value)

    # Check for quick return
    dt = xr.DataTree.from_dict({"/": dataset})
    with patch("xarray.DataTree.update") as update:
        merged = merge_tree_dset(dt)
        update.assert_not_called()


# --- split_from_geo --- #


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

    @pytest.mark.parametrize("groupby", ["", "cat"])
    def test_split(self, groupby, geo_dataset, geo_dataframe, assert_split_1_2):
        dt = split_from_geo(geo_dataset, geo_dataframe, groupby=groupby)
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
            assert "must have exactly one column" in str(exc.value)

    def test_prune_node(self, geo_dataset, geo_dataframe, assert_split_1_2):
        dt = split_from_geo(geo_dataset, geo_dataframe, groupby="cat", prune_node=False)
        assert not dt.is_hollow
        xrt.assert_identical(dt.to_dataset(), geo_dataset)
        assert_split_1_2(dt, geo_dataframe)

    def test_keep_exterior(
        self, geo_dataset, geo_dataframe, geo_series, assert_split_1_2
    ):
        dt = split_from_geo(
            geo_dataset, geo_dataframe, groupby="cat", keep_exterior=True
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
        xrt.assert_identical(dt, xr.DataTree(geo_dataset))

    def test_pass_groupby_kwargs(self, geo_dataset, geo_dataframe, geo_series):
        geo_dataframe.loc[4, "cat"] = np.nan
        dt = split_from_geo(
            geo_dataset, geo_dataframe, groupby={"by": "cat", "dropna": False}
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
