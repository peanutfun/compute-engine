"""Functions"""

from pathlib import Path
from typing import Callable, Mapping, Hashable, TypeVar

import xarray as xr
import odc.geo.xr  # noqa: F401
import rioxarray  # noqa: F401
import numpy as np

from .types import DatasetOrArray, AnyXarray

# from . import sparse

DATA_DIR = Path(__file__).parent.absolute() / "data"
DATA_DIR.mkdir(exist_ok=True)

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
        return arr.map_over_datasets(lambda x: x.unify_chunks()).filter(
            lambda x: bool(x.ds.data_vars)
        )
    return arr.unify_chunks()


def is_chunked(arr: AnyXarray) -> bool:
    """Check if xarray object is chunked"""
    arr = unify_chunks(arr)
    if isinstance(arr, xr.DataTree):
        if any((node.ds.chunks for node in arr.subtree if node.has_data)):
            return True
        return False

    return arr.chunks is not None


def normed_chunksize(
    arr: AnyXarray,
    agg: Callable[[tuple[int, ...] | int], float] = np.nanmax,
) -> dict[Hashable, int]:
    """Return the normed chunksizes"""
    chunksizes = unify_chunks(arr).chunksizes
    if isinstance(arr, xr.DataTree):
        chunksizes = merge_dicts(*chunksizes.values(), agg=agg, default=np.nan)
    return {dim: int(agg(chunks)) for dim, chunks in chunksizes.items()}


def norm_chunks(arr: AnyXarray, ref: AnyXarray | None = None) -> AnyXarray:
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

        chunks = {dim: "auto" for dim in arr_chunksizes.keys()} | ref_chunks

    return unify_chunks(arr.chunk(chunks))


def rename_spatial_dims(
    arr: DatasetOrArray, target: xr.Dataset | xr.DataArray
) -> DatasetOrArray:
    """Rename spatial dimensions in arr like the ones in target"""
    arr = arr.rename({
        arr.rio.x_dim: target.rio.x_dim,
        arr.rio.y_dim: target.rio.y_dim,
    })
    return arr.rio.set_spatial_dims(target.rio.x_dim, target.rio.y_dim)


def align_exposure(
    hazard: xr.Dataset | xr.DataArray,
    exposure: DatasetOrArray,
    method: str | Mapping[Hashable, str] = "nearest",
) -> DatasetOrArray:
    """Align events in exposure"""
    # Find dimensions that are in both datasets (except spatial dims)
    align_dims = set(hazard.dims).intersection(exposure.dims) - {
        exposure.rio.x_dim,
        exposure.rio.y_dim,
    }
    if not align_dims:
        return exposure

    # Use raw array to avoid vectorized indexing (which might cause a copy)
    coords = {dim: hazard[dim].to_numpy() for dim in align_dims}

    # Use .sel or .interp depending on 'method'
    def sel_or_interp(arr, indexer, method):
        if method == "nearest":
            return arr.reindex(indexer, method="nearest")

        return arr.interp(indexer, method=method)

    # Shortcut, use all indexers
    if isinstance(method, str):
        return sel_or_interp(exposure, coords, method)

    # Apply indexers one at a time
    for dim, indexer in coords.items():
        dim_method = method.get(dim)
        if dim_method is None:
            raise ValueError(f"No alignment method given for dim '{dim}'")
        exposure = sel_or_interp(exposure, indexer, dim_method)

    return exposure


# NOTE: If datatree, need to map functions over all datasets!
# TODO: modify tolerance and reproject args!
def reproject_hazard(
    hazard: DatasetOrArray,  # Could be Dataset
    exposure: xr.Dataset | xr.DataArray,  # Could be DataTree
    resampling: str = "nearest",
    align_chunks: bool = True,
    force_reproject: bool = False,
) -> DatasetOrArray:
    """Reproject the hazard"""

    # def wrap(func, _hazard):
    #     if isinstance(_hazard, xr.DataTree):
    #         return xr.map_over_datasets(func, _hazard)
    #     return func(_hazard)

    # Rename dimensions
    hazard = hazard.copy()  # Shallow copy
    hazard = rename_spatial_dims(hazard, exposure)

    # Reproject
    if (
        not force_reproject
        and (hazard.rio.crs == exposure.rio.crs)
        and (resampling == "nearest")
    ):
        res_x, res_y = np.abs(hazard.rio.resolution()) / 2.0
        hazard = hazard.reindex(
            {hazard.rio.x_dim: exposure[exposure.rio.x_dim]},
            method="nearest",
            tolerance=res_x,
        ).reindex(
            {hazard.rio.y_dim: exposure[exposure.rio.y_dim]},
            method="nearest",
            tolerance=res_y,
        )
    else:
        hazard = hazard.odc.reproject(
            exposure.odc.geobox, resampling=resampling, dst_nodata=np.nan
        )

    # Chunking: Use exposure chunk sizes where applicable
    hazard = norm_chunks(hazard, exposure if align_chunks else None)
    return hazard


# def cache_zarr(arr: AnyXarray, path: Path | str, mode: str = "w") -> AnyXarray:
#     """Write data array to a Zarr file"""
#     open_func = xr.open_dataset
#     if isinstance(arr, xr.DataArray):
#         open_func = xr.open_dataarray
#     elif isinstance(arr, xr.DataTree):
#         open_func = xr.open_datatree

#     # Densify sparse objects
#     # arr = arr.sp.to_dense()

#     # Store and reload
#     path = Path(path).with_suffix(".zarr")
#     arr.to_zarr(path, mode=mode)
#     del arr
#     with open_func(path, chunks="auto", decode_coords="all", engine="zarr") as arr:
#         return arr
