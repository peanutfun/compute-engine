import numpy as np
import pytest
import xarray as xr

from crace.align import rename_spatial_dims


@pytest.fixture
def simple_dataarray():
    data = np.random.rand(4, 4)
    return xr.DataArray(data, coords={"y": np.arange(4), "x": np.arange(4)})


def test_rename_spatial_dims(simple_dataarray):
    # Create a DataArray with dimensions that are known spatial axes
    arr = simple_dataarray.copy()
    arr.rio.set_spatial_dims("x", "y", inplace=True)

    # Create a target with different dimension names
    target = arr.copy()
    target = target.rename({"x": "lon", "y": "lat"})
    target.rio.set_spatial_dims("lon", "lat", inplace=True)

    # Rename arr's spatial dims to match target
    result = rename_spatial_dims(arr, target)

    assert set(result.dims) == {"lon", "lat"}
    assert result.rio.x_dim == target.rio.x_dim
    assert result.rio.y_dim == target.rio.y_dim
