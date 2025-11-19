"""Functions"""

from typing import Callable, Hashable, Mapping, TypeVar

import numpy as np
import odc.geo.xr  # noqa: F401
import rioxarray  # noqa: F401
import xarray as xr
from dask.array import Array as DaskArray

from .tree import map_over_datasets
from .types import AnyXarray

T = TypeVar("T")  # Generic type


def merge_dicts(
    *dicts: Mapping[Hashable, T], agg: Callable[[tuple[T, ...]], T], default: T
) -> dict[Hashable, T]:
    """Merge dict values with an aggregator"""
    merged = dict(dicts[0])
    for new in dicts[1:]:
        for key, value in new.items():
            merged[key] = agg((merged.get(key, default), agg(value)))
    return merged


def unify_chunks(arr: AnyXarray) -> AnyXarray:
    """Unify chunks of an xarray object"""
    if isinstance(arr, xr.DataTree):
        return map_over_datasets(lambda x: x.unify_chunks(), arr)
    return arr.unify_chunks()


def is_chunked(arr: AnyXarray) -> bool:
    """Check if xarray object is chunked"""
    if isinstance(arr, xr.DataArray):
        return isinstance(arr.data, DaskArray)

    if isinstance(arr, xr.Dataset):
        return any((is_chunked(da) for da in arr.data_vars.values()))

    if isinstance(arr, xr.DataTree):
        return any((is_chunked(node.ds) for node in arr.subtree))

    return False


def normed_chunksize(
    arr: AnyXarray,
    agg: Callable[[tuple[int, ...] | int], float] = np.nanmax,
) -> dict[Hashable, int]:
    """Return the normed chunksizes"""
    chunksizes = unify_chunks(arr).chunksizes
    if isinstance(arr, xr.DataTree):
        chunksizes = merge_dicts(*chunksizes.values(), agg=agg, default=np.nan)
    return {dim: int(agg(chunks)) for dim, chunks in chunksizes.items()}


def norm_chunks(
    arr: AnyXarray, ref: xr.DataTree | xr.Dataset | xr.DataArray | None = None
) -> AnyXarray:
    """Rechunk the array to normed chunksizes"""
    # Check if array is chunked at all
    if not is_chunked(arr):
        return arr

    # Adapt chunks
    arr = unify_chunks(arr)
    arr_chunksizes = normed_chunksize(arr)
    if ref is None:
        chunks = arr_chunksizes
    else:
        # No chunks, so array must fit into memory
        if not is_chunked(ref):
            ref_chunks = dict(ref.sizes)

        # Use ref chunks, but filter out dimensions not present in arr
        else:
            ref_chunks = dict(
                filter(
                    lambda item: item[0] in arr_chunksizes,
                    normed_chunksize(ref).items(),
                )
            )

        # Assemble new chunks
        chunks = {}
        for dim in arr_chunksizes:
            if dim in ref_chunks:
                # Do not make chunks smaller than before
                chunks[dim] = np.nanmax((arr_chunksizes[dim], ref_chunks[dim]))
            else:
                chunks[dim] = "auto"

    return unify_chunks(arr.chunk(chunks))
