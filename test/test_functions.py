import numpy as np
import numpy.testing as npt
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from climadace.chunks import (
    is_chunked,
    merge_dicts,
    norm_chunks,
    normed_chunksize,
)


@pytest.fixture
def simple_dataarray():
    data = np.random.rand(4, 4)
    return xr.DataArray(data, coords={"y": np.arange(4), "x": np.arange(4)})


@pytest.fixture
def exposure_dataarray(simple_dataarray):
    simple_dataarray.rio.write_crs("EPSG:4326", inplace=True).chunk(2, 2)
    return simple_dataarray


@pytest.fixture
def hazard_dataarray(simple_dataarray):
    simple_dataarray = simple_dataarray.rename("hazard").chunk((1, 4))
    simple_dataarray.rio.write_crs("EPSG:4326", inplace=True)
    return simple_dataarray


def test_merge_dicts():
    dict1 = {"a": 1, "b": 2}
    dict2 = {"a": 2, "b": 3, "c": 1}
    dict3 = {"b": 4, "c": 2, "d": 1}

    result1 = merge_dicts(dict1, dict2, dict3, default=np.nan, agg=np.nanmax)
    assert result1 == {"a": 2, "b": 4, "c": 2, "d": 1}
    result2 = merge_dicts(dict1, dict2, dict3, default=np.nan, agg=np.nanmin)
    assert result2 == {"a": 1, "b": 2, "c": 1, "d": 1}
    result3 = merge_dicts(dict1, dict2, dict3, default=np.nan, agg=np.min)
    npt.assert_array_equal(list(result3.values()), [1, 2, np.nan, np.nan])


@pytest.mark.skip("Needs to be applied to arrays")
def test_normed_chunksize():
    chunksizes = {"x": (4, 3, 2), "y": (2, 1)}
    result = normed_chunksize(chunksizes)
    assert result == {"x": 4, "y": 2}

    result_min = normed_chunksize(chunksizes, agg=np.min)
    assert result_min == {"x": 2, "y": 1}


def test_is_chunked(simple_dataarray):
    assert not is_chunked(simple_dataarray)
    assert is_chunked(simple_dataarray.chunk("auto"))


def test_is_chunked_dataset(simple_dataarray):
    ds = xr.Dataset(
        {
            "var1": simple_dataarray.copy(),
            "var2": simple_dataarray.copy().chunk({"x": 2}),
        }
    )
    assert is_chunked(ds)


def test_is_chunked_datatree(simple_dataarray):
    dt = xr.DataTree.from_dict(
        {
            "/": simple_dataarray.to_dataset(name="root"),
            "/a": simple_dataarray.to_dataset(name="a"),
        }
    )
    assert not is_chunked(dt)

    dt["/a"] = simple_dataarray.to_dataset(name="a").chunk("auto")
    assert is_chunked(dt)


def test_norm_chunks_no_chunks(simple_dataarray):
    """Check if norm chunks does nothing if the array is not chunked"""
    assert not norm_chunks(simple_dataarray).chunks
    assert not norm_chunks(simple_dataarray, ref=simple_dataarray).chunks


def test_norm_chunks_no_ref(simple_dataarray):
    arr = simple_dataarray.chunk((2, 2))
    result = norm_chunks(arr)
    assert result.chunksizes == arr.unify_chunks().chunksizes


# TODO: Check behavior for datasets with arrays whose chunksizes differ
def test_norm_chunks_with_ref(simple_dataarray):
    arr = simple_dataarray.chunk((2, 2))
    ref = xr.DataArray(
        np.random.rand(2, 4, 4),
        coords={"z": np.arange(2), "y": np.arange(4), "x": np.arange(4)},
    ).chunk((1, 1, 4))
    result = norm_chunks(arr, ref=ref)
    assert dict(result.chunksizes) == {"x": (4,), "y": (2, 2)}
