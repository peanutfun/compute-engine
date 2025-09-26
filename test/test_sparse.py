"""Test functions for sparse operations"""

import pytest
import xarray as xr
import numpy as np
from hypothesis import strategies as st, given, settings, Verbosity
from hypothesis.extra.numpy import array_shapes, arrays

import climadace.sparse
from climadace.sparse import zero_to_nan, SparseArray


@st.composite
def array(draw, max_dims=4, max_side=5, elements=None):
    shape = draw(array_shapes(max_dims=max_dims, max_side=max_side))
    arr = draw(arrays(np.float64, shape=shape, elements=elements))
    coords = {str(idx): np.arange(size) for idx, size in enumerate(shape)}
    return xr.DataArray(arr, coords=coords)


@pytest.fixture
def dset_ones():
    return xr.Dataset(
        {"var": (["x", "y"], np.ones((3, 4), dtype="float"))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )


@pytest.fixture
def dset_zeros():
    ds = xr.Dataset(
        {"var": (["x", "y"], np.zeros((3, 4), dtype="float"))},
        coords={"x": np.arange(3), "y": np.arange(4)},
    )
    ds["var"][0, 0] = np.nan
    return ds


@given(array())
def test_accessor(arr):
    """Test if the accessor exists"""
    assert hasattr(arr, "sp")
    assert not arr.sp.is_sparse
    assert arr.sp.array is None


@given(array())
def test_sparsify(arr):
    """Test sparsification"""
    arr_sp = arr.sp.to_sparse()
    assert not arr.sp.is_sparse
    assert arr_sp.sp.is_sparse
    assert isinstance(arr_sp.sp.array, SparseArray)
    print(arr_sp.sp.array)

    nnz = np.count_nonzero(~np.isnan(zero_to_nan(arr)))
    density = nnz / arr.size
    assert arr_sp.sp.array.density == density

    # Densify
    # arr_ds = arr_sp.sp.to_dense()
    # print(arr_ds)
    # arr_ds = arr_ds.compute()
    # print(arr_ds)
    # assert False
