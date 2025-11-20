"""Functions"""

from typing import Callable, Hashable, Mapping, Sequence, TypeVar

import numpy as np
import odc.geo.xr  # noqa: F401
import rioxarray  # noqa: F401
import xarray as xr
from dask.array import Array as DaskArray

from .tree import map_over_datasets
from .types import AnyXarray

T = TypeVar("T")  # Generic type


def merge_dicts(
    *dicts: Mapping[Hashable, T], agg: Callable[[T | Sequence[T]], T], default: T
) -> dict[Hashable, T]:
    """Merge several dictionaries using an aggregator function.

    For each key appearing in one or more dictionaries, all values are aggregated
    using the provided aggregator. Missing keys are replaced with ``default``.

    Args:
        *dicts (Mapping[Hashable, T]): Dictionaries to merge.
        agg (Callable[[T | Sequence[T]], T]): Aggregation function taking a sequence or
            single value and returning a single merged/aggregated value.
        default (T): Default value used when a key is missing in a dictionary.

    Returns:
        dict[Hashable, T]: A merged dictionary where values are aggregated.
    """
    merged = dict(dicts[0])
    for new in dicts[1:]:
        for key, value in new.items():
            merged[key] = agg((merged.get(key, default), agg(value)))
    return merged


def unify_chunks(arr: AnyXarray) -> AnyXarray:
    """Unify the chunk structure of an xarray object.

    Applies ``xarray.DataArray.unify_chunks`` or ``xarray.Dataset.unify_chunks``.
    For ``DataTree`` inputs, applies unification across all datasets.

    Args:
        arr (AnyXarray): An xarray ``DataArray``, ``Dataset``, or ``DataTree``.

    Returns:
        AnyXarray: A new object with unified chunk layout.

    See also:
        https://docs.xarray.dev/en/stable/generated/xarray.unify_chunks.html#xarray.unify_chunks
    """
    if isinstance(arr, xr.DataTree):
        return map_over_datasets(lambda x: x.unify_chunks(), arr)
    return arr.unify_chunks()


def is_chunked(arr: AnyXarray) -> bool:
    """Determine whether an xarray object is Dask-chunked.

    Args:
        arr (AnyXarray): An xarray ``DataArray``, ``Dataset`` or ``DataTree``.

    Returns:
        bool: True if any stored data uses Dask chunks, otherwise False.
    """
    if isinstance(arr, xr.DataArray):
        return isinstance(arr.data, DaskArray)

    if isinstance(arr, xr.Dataset):
        return any((is_chunked(da) for da in arr.data_vars.values()))

    if isinstance(arr, xr.DataTree):
        return any((is_chunked(node.ds) for node in arr.subtree))

    return False


def normed_chunksize(
    arr: AnyXarray,
    agg: Callable[[Sequence[float | int] | float | int], float] = np.nanmax,
) -> dict[Hashable, int]:
    """Compute normalized chunk sizes for each dimension of an xarray object.

    Chunk sizes are unified and then aggregated along each dimension using
    the provided aggregator (e.g., ``np.nanmax``). For ``DataTree`` inputs,
    chunk sizes from all datasets are merged.

    Args:
        arr (AnyXarray): The input xarray object.
        agg (Callable[..., float], optional): Aggregator applied to dimension chunk
        sizes. Defaults to ``np.nanmax``.

    Returns:
        dict[Hashable, int]: A mapping from dimension names to normalized chunk sizes.
    """
    chunksizes = unify_chunks(arr).chunksizes
    if isinstance(arr, xr.DataTree):
        chunksizes = merge_dicts(*chunksizes.values(), agg=agg, default=np.nan)
    return {dim: int(agg(chunks)) for dim, chunks in chunksizes.items()}


def norm_chunks(
    arr: AnyXarray, ref: xr.DataTree | xr.Dataset | xr.DataArray | None = None
) -> AnyXarray:
    """Rechunk an xarray object to normalized chunk sizes.

    Normalization uses the array's own chunking unless a reference array ``ref`` is
    provided. When a reference is given:

    - If the reference is unchunked, its sizes are used as chunksizes.
    - If the reference is chunked, its normalized chunk sizes are used.
    - Chunks are never made *smaller* than the original array's chunks.

    Args:
        arr (AnyXarray): Input xarray object to rechunk.
        ref (xr.DataTree | xr.Dataset | xr.DataArray | None, optional):
            Reference object whose chunk sizes should be matched when possible.
            Defaults to None.

    Returns:
        AnyXarray: A rechunked xarray object with normalized chunk sizes.
    """
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
