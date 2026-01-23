"""Aligning datasets"""

from typing import Any, Hashable, Mapping, TypeVar, overload

import numpy as np
import odc.geo.xr  # noqa: F401
import rioxarray  # noqa: F401
import xarray as xr

from .chunks import norm_chunks
from .types import SPATIAL_DIM, AnyXarray, DatasetOrArray

T = TypeVar("T")  # Generic type


def rename_spatial_dims(
    arr: DatasetOrArray, target: xr.Dataset | xr.DataArray
) -> DatasetOrArray:
    """Rename spatial dimensions in arr like the ones in target"""
    arr = arr.rename(
        {
            arr.rio.x_dim: target.rio.x_dim,
            arr.rio.y_dim: target.rio.y_dim,
        }
    )
    return arr.rio.set_spatial_dims(target.rio.x_dim, target.rio.y_dim)


def align_exposure(
    hazard: xr.Dataset | xr.DataArray,
    exposure: DatasetOrArray,
    method: str | Mapping[Hashable, str] = "nearest",
) -> DatasetOrArray:
    """Align events in exposure"""
    # Find dimensions that are in both datasets (except spatial dims)
    align_dims = (
        set(hazard.dims)
        .intersection(exposure.dims)
        .difference(exposure.odc.spatial_dims or {SPATIAL_DIM})
    )
    if not align_dims:
        return exposure

    # Use raw array to avoid vectorized indexing (which might cause a copy)
    coords = {dim: hazard[dim].to_numpy() for dim in align_dims}

    # Use .sel or .interp depending on 'method'
    def reindex_or_interp(arr, indexer, method):
        if method == "nearest":
            return arr.reindex(indexer, method="nearest", copy=False)

        return arr.interp(indexer, method=method)

    # Shortcut, use all indexers
    if isinstance(method, str):
        return reindex_or_interp(exposure, coords, method)

    # Apply indexers one at a time
    for dim, indexer in coords.items():
        dim_method = method.get(dim, "nearest")
        exposure = reindex_or_interp(exposure, indexer, dim_method)

    return exposure


# NOTE: If datatree, need to map functions over all datasets!
# TODO: modify tolerance and reproject args!
def reproject_hazard(
    hazard: DatasetOrArray,  # Could be Dataset
    exposure: xr.Dataset | xr.DataArray,  # Could be DataTree
    resampling: str = "nearest",
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
            copy=False,
        ).reindex(
            {hazard.rio.y_dim: exposure[exposure.rio.y_dim]},
            method="nearest",
            tolerance=res_y,
            copy=False,
        )
    else:
        hazard = hazard.odc.reproject(
            exposure.odc.geobox, resampling=resampling, dst_nodata=np.nan
        )

    return hazard


@overload
def promote_to_dataset(
    arr: xr.DataArray | xr.Dataset, name: str, force_name: bool = ...
) -> xr.Dataset: ...
@overload
def promote_to_dataset(arr: Any, name: str, force_name: bool = ...) -> Any: ...
def promote_to_dataset(arr, name, force_name: bool = False):
    if isinstance(arr, xr.DataArray):
        if arr.name and not force_name:
            name = str(arr.name)
        arr = arr.to_dataset(name=name, promote_attrs=True)
    return arr


@overload
def promote_to_datatree(arr: AnyXarray, name: str) -> xr.DataTree: ...
@overload
def promote_to_datatree(arr: Any, name: str) -> Any: ...
def promote_to_datatree(arr, name):
    arr = promote_to_dataset(arr, name, force_name=False)
    if isinstance(arr, xr.Dataset):
        arr = xr.DataTree(dataset=arr, name=name)
    return arr


def align(
    hazard: xr.DataArray | xr.Dataset,
    exposure: xr.DataArray | xr.Dataset,
    *,
    method_nonspatial: str | Mapping[Hashable, str] = "nearest",
    method_spatial: str = "nearest",
    force_reproject: bool = False,
    align_chunks: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Align hazard and exposure datasets for impact calculation.

    To ensure alignment in the xarray sense, the spatial dimensions of the hazard
    dataset will be renamed to match those of the exposure dataset. If both datasets
    contain only a single data variable array, this method will ensure that they have
    the same name, defaulting to the name of the exposure data variable or the name
    ``"exposure"``, if the input arrays are unnamed.

    Parameters
    ----------
    hazard
        Geospatial data on the hazard intensity. :py:class:`xarray.DataArray` objects
        will be promoted to :py:class:`xarray.Dataset`.
    exposure
        Geospatial data of exposure. :py:class:`xarray.DataArray` objects
        will be promoted to :py:class:`xarray.Dataset`.
    method_nonspatial
        Alignment method for non-spatial dimensions. Choose a ``method`` setting of
        :py:meth:`~xarray.Dataset.interp`. For ``"nearest"`` (default), the dimension
        coordinates are reindexed with :py:meth:`~xarray.Dataset.reindex`.
    method_spatial
        Alignment method for spatial dimensions (as identified by
        :py:attr:`~odc.geo.xr.ODCExtension.spatial_dims`). Choose a ``resampling``
        parameter from :py:meth:`~odc.geo.xr.ODCExtensionDa.reproject`. For
        ``"nearest"`` (default), the dimension coordinates are reindexed with
        :py:meth:`~xarray.Dataset.reindex`, unless ``force_reproject=True``.
    force_reproject
        If ``True``, always use :py:meth:`~odc.geo.xr.ODCExtensionDa.reproject` for
        aligning spatial coordinates. If ``False`` (default), ``nearest`` interpolation
        is done by reindexing.
    align_chunks
        If ``True``, rechunk the hazard dataset to align its chunks with the exposure
        dataset.

    Returns
    -------
    xarray.Dataset
        Aligned hazard dataset
    xarray.Dataset
        Aligned exposure dataset

    See Also
    --------
    xarray.Dataset.reindex
        Method for ``"nearest"`` alignment for spatial and non-spatial coordinates
    xarray.Dataset.interp
        Method for other alignments of non-spatial coordinates
    odc.geo.xr.ODCExtensionDa.reproject
        Method for other spatial alignments
    """
    aligner = Aligner(
        hazard=hazard,
        exposure=exposure,
        method_nonspatial=method_nonspatial,
        method_spatial=method_spatial,
        force_reproject=force_reproject,
        align_chunks=align_chunks,
    )
    return aligner.hazard, aligner.exposure


class Aligner:
    def __init__(
        self,
        hazard: xr.DataArray | xr.Dataset,
        exposure: xr.DataArray | xr.Dataset,
        *,
        method_nonspatial: str | Mapping[Hashable, str],
        method_spatial: str,
        force_reproject: bool,
        align_chunks: bool,
    ):
        self._hazard = hazard
        self._exposure = exposure

        # Align exposure (ignoring space)
        self._exposure = align_exposure(
            self._hazard, self._exposure, method=method_nonspatial
        )

        # Check 1D in space, assume CLIMADA then
        if self._hazard.odc.spatial_dims is None:
            self._align_1d_climada()

        # Align hazard and exposure
        else:
            self._align_2d(
                method_spatial=method_spatial, force_reproject=force_reproject
            )

        # Align names
        self._exposure, self._hazard = self.align_names(
            self._exposure, self._hazard, "exposure"
        )

        # Chunking
        if align_chunks:
            self._hazard = norm_chunks(self._hazard, self._exposure)

    def _align_2d(
        self,
        method_spatial: str,
        force_reproject: bool,
    ):
        self._hazard = reproject_hazard(
            self._hazard,
            self._exposure,
            resampling=method_spatial,
            force_reproject=force_reproject,
        )

    def _align_1d_climada(self):
        from .climada import align_1d  # noqa: PLC0415

        self._hazard, self._exposure = align_1d(self._hazard, self._exposure)

    @staticmethod
    def align_names(
        arr1: DatasetOrArray, arr2: DatasetOrArray, name_fallback: str
    ) -> tuple[DatasetOrArray, DatasetOrArray]:
        name = name_fallback
        if isinstance(arr1, xr.DataArray):
            if arr1.name is None:
                arr1.name = name
            else:
                name = arr1.name
        elif isinstance(arr1, xr.Dataset) and len(arr1.data_vars) == 1:
            name = next(iter(arr1.data_vars))
        else:
            print("WARNING! Could not align names")

        if isinstance(arr2, xr.DataArray):
            arr2.name = name
        elif isinstance(arr2, xr.Dataset) and len(arr2.data_vars) == 1:
            arr2 = arr2.rename({next(iter(arr2.data_vars)): name})
        else:
            print("WARNING! Could not align names")

        return arr1, arr2

    @property
    def hazard(self) -> xr.Dataset:
        return promote_to_dataset(self._hazard, name="exposure")

    @property
    def exposure(self) -> xr.Dataset:
        return promote_to_dataset(self._exposure, name="exposure")
