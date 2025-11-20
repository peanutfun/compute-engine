"""CLIMADA interoperability module.

This module provides functions to convert CLIMADA `Hazard` and `Exposures`
objects into `xarray.Dataset` representations, enabling interoperability with
xarray-based geospatial and analytical workflows.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Mapping

import numpy as np
import odc.geo  # noqa: F401
import pandas as pd
import rioxarray  # noqa: F401
import sparse as sp
import xarray as xr
from xarray.core import dtypes as xrdtypes

try:
    from climada.entity import Exposures
    from climada.hazard import Hazard
except ImportError as err:
    raise RuntimeError("Please install CLIMADA for using this module!") from err

from .sparse import zero_to_nan
from .types import SPATIAL_DIM, DatasetOrArray

EVENT_DIM = "event"


def drop_none(mapping: Mapping[str, Any]):
    """Remove keys with ``None`` values from a dictionary.

    Args:
        mapping (Mapping[str, Any]): Input dictionary.

    Returns:
        dict[str, Any]: A new dictionary without entries whose values are ``None``.
    """
    return {key: val for key, val in mapping.items() if val is not None}


def set_single_indices(
    dset: DatasetOrArray, coord_names: str | Iterable[str], index_cls=None, **options
) -> DatasetOrArray:
    """Set individual indices for existing coordinates on an xarray object.

    Contrary to `xarray.Dataset.set_xindex``, this does not create a multi-index from
    the coordinates in ``coord_names``, but individual indices.

    Args:
        dset (xr.Dataset | xr.DataArray): The xarray dataset or data array to modify.
        coord_names (str | Iterable[str]): Name or iterable of names of coordinates
            to set as indices.
        index_cls (optional): Custom index class to use for indexing. Defaults to
            ``None``.
        **options: Additional keyword arguments passed to ``xarray.Dataset.set_xindex``.

    Returns:
        xr.Dataset | xr.DataArray: A new dataset or array with the specified coordinates
        indexed.

    See Also:
        https://docs.xarray.dev/en/stable/generated/xarray.Dataset.set_xindex.html#xarray.Dataset.set_xindex
    """
    if isinstance(coord_names, str):
        coord_names = [coord_names]
    for name in coord_names:
        dset = dset.set_xindex(name, index_cls=index_cls, **options)
    return dset


def hazard_to_dset(
    hazard: Hazard, *, to_grid: bool = True, parse_date: bool = True
) -> xr.Dataset:
    """Convert a CLIMADA ``Hazard`` object into an xarray Dataset.

    The resulting dataset represents hazard data in either stacked or gridded form,
    including metadata and coordinate reference system (CRS) information.

    Args:
        hazard (climada.Hazard): A CLIMADA ``Hazard`` instance.
        to_grid (bool, optional): If True, create a 2D grid from the lat/lon
            coordinates. Defaults to True.
        parse_date (bool, optional): If True, convert ordinal date integers to pandas
            Timestamps. Defaults to True.

    Returns:
        xarray.Dataset: An xarray Dataset representation of the hazard object,
        including event and spatial dimensions, metadata, and CRS.
    """

    # 2D variables
    def get_matrix(attr: str):
        matrix = getattr(hazard, attr)
        if matrix.nnz != 0:
            return zero_to_nan(sp.GCXS.from_scipy_sparse(matrix))
        return None

    # 1D variables
    def get_event_array(attr: str):
        arr = getattr(hazard, attr)
        if len(arr) == hazard.size:
            return arr
        return None

    # Build dataset
    matrices = drop_none({attr: get_matrix(attr) for attr in ("intensity", "fraction")})
    event_arrays = drop_none(
        {
            attr: get_event_array(attr)
            for attr in ("event_id", "frequency", "event_name", "date", "orig")
        }
    )
    spatial_arrays = drop_none(
        {
            attr: getattr(hazard.centroids, attr)
            for attr in ("lat", "lon", "on_land", "region_id")
        }
    )
    spatial_arrays["latitude"] = spatial_arrays.pop("lat")
    spatial_arrays["longitude"] = spatial_arrays.pop("lon")
    ds = xr.Dataset(
        {key: ((EVENT_DIM, SPATIAL_DIM), val) for key, val in matrices.items()},
        coords={key: (SPATIAL_DIM, val) for key, val in spatial_arrays.items()}
        | {key: (EVENT_DIM, val) for key, val in event_arrays.items()},
    )
    ds = ds.rio.write_crs(hazard.centroids.crs)

    # Metadata
    ds.attrs = {
        attr: getattr(hazard, attr) for attr in ("haz_type", "units", "frequency_unit")
    }

    # Parse date
    if parse_date and "date" in ds.coords:
        ds = ds.assign_coords(
            {
                "date": (
                    EVENT_DIM,
                    [
                        pd.Timestamp.fromordinal(date)
                        for date in ds["date"].astype("int").to_numpy()
                    ],
                )
            }
        )

    # Set event indexers
    event_indexers = set(event_arrays) - {"frequency"}
    ds = set_single_indices(ds, event_indexers)

    # Set space indexers
    if to_grid:
        ds = ds.set_xindex(["latitude", "longitude"]).unstack(
            SPATIAL_DIM,
            fill_value=defaultdict(
                lambda: xrdtypes.NA, {"on_land": False, "region_id": -1}
            ),
        )
    else:
        ds = set_single_indices(ds, spatial_arrays)

    return ds


def exposure_to_dset(exposure: Exposures, *, to_grid: bool = True) -> xr.Dataset:
    """Convert a CLIMADA ``Exposures`` object into an xarray Dataset.

    The resulting dataset represents exposure data in either stacked or gridded form,
    including metadata and coordinate reference system (CRS) information.

    Args:
        exposure (climada.Exposures): A CLIMADA ``Exposures`` instance.
        to_grid (bool, optional): If True, create a 2D grid from the lat/lon
            coordinates. Defaults to True.

    Returns:
        xarray.Dataset: An xarray Dataset containing exposure variables, coordinates,
        metadata, and CRS information.
    """
    data_vars = drop_none(
        {var: getattr(exposure, var) for var in ("value", "deductible", "cover")}
    )
    coords = drop_none(
        {
            var: getattr(exposure, var)
            for var in ("region_id", "category_id", "latitude", "longitude")
        }
    )
    cols = {
        var: exposure.gdf[var]
        for var in exposure.gdf.columns
        if var.startswith(("centr_", "impf_"))
    }
    # Build dataset
    ds = xr.Dataset(
        {key: (SPATIAL_DIM, val) for key, val in data_vars.items()},
        coords={key: (SPATIAL_DIM, val) for key, val in coords.items()}
        | {key: (SPATIAL_DIM, val) for key, val in cols.items()},
    )
    ds = ds.rio.write_crs(exposure.crs)

    # Metadata
    ds.attrs = {
        attr: getattr(exposure, attr)
        for attr in ("description", "ref_year", "value_unit")
    }

    # Gridding
    if to_grid:
        ds = ds.set_xindex(["latitude", "longitude"]).unstack(
            SPATIAL_DIM,
            fill_value=defaultdict(
                lambda: xrdtypes.NA, {"region_id": -1, "category_id": -1}
            ),
        )
    else:
        ds = set_single_indices(ds, coords)

    return ds


# TODO: Automatically assign centroids if exposure has none
def align_1d(hazard: xr.Dataset, exposure: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """Align one-dimensional hazard and exposure datasets by centroid assignment.

    This function aligns a hazard dataset and an exposure dataset along their
    one-dimensional spatial axis (``loc``), based on centroid indices stored in the
    exposure dataset. It performs the following steps:

    Args:
        hazard (xr.Dataset): A one-dimensional hazard dataset containing spatial
            dimension ``loc`` and a hazard type stored in
            ``hazard.attrs["<haz_type>"]``.
        exposure (xr.Dataset): A one-dimensional exposure dataset containing a
            centroid lookup variable named ``"centr_<haz_type>"``.

    Returns:
        tuple[xr.Dataset, xr.Dataset]: The aligned hazard and exposure datasets, such
        that:

            - hazard and exposure entries with invalid centroid assignments are removed,
            - hazard entries are reordered to match exposure,
            - hazard entries have the same coordinates as exposure (to ensure the
              datasets align).

    Raises:
        AssertionError: If either input dataset is two-dimensional.

    Notes:
        * This function relies on centroid ID variables of the form
          ``centr_<haz_type>`` in the exposure dataset.
    """
    assert hazard.odc.spatial_dims is None, "Hazard must be one-dimensional in space"
    assert exposure.odc.spatial_dims is None, (
        "Exposure must be one-dimensional in space"
    )

    def centroid_ids(exp):
        return exp[f"centr_{hazard.attrs['haz_type']}"].to_numpy()

    exposure = exposure.where(centroid_ids(exposure) >= 0).dropna(
        dim=SPATIAL_DIM, how="all"
    )

    def drop_spatial_indexes(ds: xr.Dataset) -> xr.Dataset:
        indexes = ds.indexes
        idx_to_drop = {
            name: (SPATIAL_DIM, indexes[name].to_numpy())
            for name in indexes.keys()
            if (SPATIAL_DIM in list(indexes.get_all_dims(name).keys()))
        }
        return ds.drop_indexes(idx_to_drop.keys()).assign_coords(idx_to_drop)

    # Drop indexes
    hazard = drop_spatial_indexes(hazard)
    exposure = drop_spatial_indexes(exposure)

    # Alignment
    hazard = hazard.isel({SPATIAL_DIM: centroid_ids(exposure)})

    # Fix intersected coordinates
    # NOTE: Need to assign coordinates because dsets will be misaligned otherwise
    def spatial_coords(ds):
        for name, arr in ds.coords.items():
            if SPATIAL_DIM in arr.dims:
                yield name

    coords = {SPATIAL_DIM: np.arange(exposure.sizes[SPATIAL_DIM])} | {
        name: (SPATIAL_DIM, exposure[name].to_numpy())
        for name in set(spatial_coords(exposure)).intersection(spatial_coords(hazard))
    }
    hazard = hazard.assign_coords(coords)
    exposure = exposure.assign_coords(coords)

    return hazard, exposure
