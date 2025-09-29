"""Tests for impact engine"""

import pytest
import xarray as xr
import numpy as np
import numpy.testing as npt
import pandas as pd

from climadace.tree import map_over_datasets
from climadace.engine import Engine
from climadace.impact_funcs import ImpactFunctionMap, FuncType


@pytest.fixture
def dataset():
    return xr.Dataset(
        {"var": (["x", "y"], np.ones((10, 10), dtype="float"))},
        coords={"x": np.arange(10), "y": np.arange(10)},
    )


@pytest.fixture
def datatree(dataset):
    return xr.DataTree.from_dict(
        {
            "/a": dataset.sel(x=slice(0, 4)),
            "/b/1": dataset.sel(x=slice(5, 9), y=slice(0, 4)),
            "/b/2": dataset.sel(x=slice(5, 9), y=slice(5, 9)),
        }
    )


@pytest.fixture
def hazard(datatree):
    return datatree


@pytest.fixture
def exposure(datatree):
    return map_over_datasets(lambda x: x * 10, datatree)


def test_impf_map(hazard, exposure):
    def impf(x):
        return x + 1

    engine = Engine(hazard, exposure, impf)
    assert engine.impf_map is impf

    impf_map = ImpactFunctionMap({FuncType.leaf: lambda x: x + 2})
    engine.impf_map = impf_map
    assert engine.impf_map is impf_map


def test_impact(hazard, exposure, datatree):
    engine = Engine(hazard, exposure, lambda x: x)
    impact = engine.impact()
    xr.testing.assert_equal(impact, map_over_datasets(lambda x: x * 10, datatree))

    impact2 = engine.impact(lambda x: x * 2)
    xr.testing.assert_equal(impact2, map_over_datasets(lambda x: x * 20, datatree))

    impact3 = engine.impact(
        ImpactFunctionMap(
            {FuncType.leaf: lambda x: x, "1": lambda x: x * 2, "/b/2": lambda x: x * 3}
        )
    )
    result = datatree.copy(deep=True)
    result["/a"] = result["/a"] * 10
    result["/b/1"] = result["/b/1"] * 20
    result["/b/2"] = result["/b/2"] * 30
    xr.testing.assert_equal(impact3, result)


def test_impact_multi_map(hazard, exposure, datatree):
    impf_map = [
        ImpactFunctionMap({FuncType.leaf: lambda x: x}),
        ImpactFunctionMap({FuncType.leaf: lambda x: x * 2}),
        ImpactFunctionMap({FuncType.leaf: lambda x: x * 3}),
    ]
    engine = Engine(hazard, exposure, impf_map)
    impact = engine.impact()
    result = map_over_datasets(
        lambda *dsets: xr.concat(dsets, pd.Index([0, 1, 2], name="impact_function")),
        map_over_datasets(lambda x: x * 10, datatree),
        map_over_datasets(lambda x: x * 20, datatree),
        map_over_datasets(lambda x: x * 30, datatree),
    )
    xr.testing.assert_equal(impact, result)


def test_impact_sampling(hazard, exposure, datatree):
    hazard = map_over_datasets(
        lambda *dsets: xr.concat(dsets, dim=pd.Index([0, 1, 2], name="foo")),
        hazard,
        hazard,
        hazard,
    )
    exposure = map_over_datasets(
        lambda *dsets: xr.concat(dsets, dim=pd.Index(["a", "b", "c"], name="bar")),
        exposure,
        exposure,
        exposure,
    )
    impf_map = [
        ImpactFunctionMap({FuncType.leaf: lambda x: x}),
        ImpactFunctionMap({FuncType.leaf: lambda x: x * 2}),
        ImpactFunctionMap({FuncType.leaf: lambda x: x * 3}),
    ]
    engine = Engine(hazard, exposure, impf_map)
    samples = pd.DataFrame.from_dict(
        {"foo": [0, 2], "bar": ["a", "b"], "impact_function": [1, 2]},
    )
    impact = engine.impact(samples=samples)

    result = map_over_datasets(
        lambda *dsets: xr.concat(dsets, pd.Index([0, 1], name="sample")),
        map_over_datasets(lambda x: x * 20, datatree),
        map_over_datasets(lambda x: x * 30, datatree),
    )
    for path in ("/a", "/b/1", "/b/2"):
        node = result[path]
        node.ds = node.ds.assign_coords(
            foo=xr.DataArray([0, 2], dims=["sample"]),
            bar=xr.DataArray(["a", "b"], dims=["sample"]),
            impact_function=xr.DataArray([1, 2], dims=["sample"]),
        )
    xr.testing.assert_equal(impact, result)
