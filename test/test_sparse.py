"""Test functions for sparse operations"""

import pytest
import xarray as xr
import numpy as np
from hypothesis import strategies as st, given, settings, Verbosity
from hypothesis.extra.numpy import array_shapes, arrays
from sparse import GCXS

import climadace.sparse


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


# -0 is a problem, apparently
# Update: -0 counts as zero accoring to numpy, but not according to sparse
@given(
    # array(elements={"min_value": 0, "max_value": 0}),  # Zeros or NaNs
    array(),
)
# @settings(max_examples=10, deadline=None)
def test_sparsify(arr):
    """Test sparsification"""
    # arr_sp = arr.sp.to_sparse(preprocess=lambda x: np.where(np.isclose(x, 0), 0, x))
    arr_sp = arr.sp.to_sparse()
    assert not arr.sp.is_sparse
    assert arr_sp.sp.is_sparse
    assert isinstance(arr_sp.sp.array, GCXS)

    arr_np = arr.to_numpy()
    # nnz = np.count_nonzero(np.where(np.isclose(arr_np, 0), 0, arr_np))
    nnz = np.count_nonzero(arr_np)
    density = nnz / arr_np.size
    assert arr_sp.sp.array.density == density

    # Densify
    # arr_ds = arr_sp.sp.to_dense()
    # print(arr_ds)
    # arr_ds = arr_ds.compute()
    # print(arr_ds)
    # assert False
