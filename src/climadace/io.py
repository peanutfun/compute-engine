"""Load and store data"""

from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import xarray as xr

from . import sparse
from .types import AnyXarray, CachePolicy
from .funcs import is_chunked, norm_chunks

# DatasetOrArray = TypeVar("DatasetOrArray", xr.Dataset, xr.DataArray)


def open_xr(
    open_f: Callable[..., AnyXarray],
    path: Path | str,
    **kwargs,
) -> AnyXarray:
    """Open any xarray object with the appropriate function"""
    open_kwargs = {
        "chunks": "auto",
        "decode_coords": "all",
    } | kwargs
    return open_f(path, **open_kwargs)


def write_zarr(arr: AnyXarray, path: Path | str, mode: str = "w", **kwargs):
    """Write data array to a Zarr file"""
    # Densify sparse objects
    if arr.sp.is_sparse:
        arr = arr.sp.to_dense()

    # Consistent chunksizes
    arr = norm_chunks(arr)

    # Store and reload
    return arr.to_zarr(path, mode=mode, **kwargs)


def cache_zarr(arr: AnyXarray, path: Path | str, mode: str = "w") -> AnyXarray:
    """Write data array to a Zarr file and reload"""
    open_func = xr.open_dataset
    if isinstance(arr, xr.DataArray):
        open_func = xr.open_dataarray
    elif isinstance(arr, xr.DataTree):
        open_func = xr.open_datatree

    # Store and reload
    path = Path(path).with_suffix(".zarr")
    write_zarr(arr, path, mode, compute=True)
    del arr
    return open_xr(open_func, path, engine="zarr")


def maybe_cache_zarr(
    arr: AnyXarray,
    path: Path | str,
    mode: str = "w",
    *,
    cache_policy: CachePolicy,
    mkdir: bool = True,
) -> AnyXarray:
    """Write data array to Zarr and reload, according to policy"""
    path = Path(path)
    assert isinstance(cache_policy, CachePolicy)

    if cache_policy == CachePolicy.always or (
        cache_policy == CachePolicy.if_chunked and is_chunked(arr)
    ):
        if mkdir:
            path.mkdir(parents=True, exist_ok=True)
        return cache_zarr(arr, path, mode)

    return arr


def maybe_load(
    arr: AnyXarray,
    max_size: int,
    load_sparse: bool,
    assumed_sparsity: float,
    compute_func = lambda x: x.compute(),
    **to_sparse_kwargs,
):
    """Maybe load an xarray object into memory"""
    if not is_chunked(arr):
        return arr

    size = arr.nbytes
    if load_sparse and size * assumed_sparsity < max_size:
        arr = arr.sp.to_sparse(to_sparse_kwargs)
    elif size > max_size:
        return arr
    return compute_func(arr)
