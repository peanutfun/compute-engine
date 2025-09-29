"""Sparse operations"""

from typing import Callable, Hashable, Mapping

import numpy as np
import sparse
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from .tree import map_over_datasets

AnySparseArray = sparse.GCXS | sparse.COO | sparse.DOK


def zero_to_nan(array: ArrayLike, exact: bool = False) -> np.ndarray:
    """Make zeros to NaNs"""
    # Exact or inexact comparison
    compare_op = np.equal
    if not exact:
        compare_op = np.isclose

    # Compare
    array = np.where(compare_op(array, 0), np.nan, array)
    return array


class SparseArray(sparse.GCXS):
    """Specialization of a the sparse array with fixed, non-default fill value"""

    fill_value = np.nan

    def __init__(self, *args, **kwargs):
        """Initialize"""
        kwargs["fill_value"] = self.fill_value
        super().__init__(*args, **kwargs)


def to_sparse(
    data: xr.DataArray,
    array_type: type[AnySparseArray] = SparseArray,
    dtype: DTypeLike | None = None,
    preprocess: Callable[[ArrayLike], ArrayLike] = zero_to_nan,
    fill_value=np.nan,
) -> xr.DataArray:
    """Make sparse"""
    if dtype is None:
        dtype = data.dtype
    # nan_to_num_kwargs = {
    #     "copy": False,
    #     "nan": 0.0,
    #     "posinf": np.inf,
    #     "neginf": -np.inf,
    # } | nan_to_num_kwargs
    return xr.apply_ufunc(
        # lambda x: array_type.from_numpy(
        #     zero_to_nan(x.astype(dtype), exact=False), fill_value=np.nan
        # ),
        lambda x: array_type.from_numpy(
            np.asanyarray(preprocess(x)).astype(dtype, copy=False),
            fill_value=fill_value,
        ),
        data,
        dask="parallelized",
        output_dtypes=[dtype],
    )


def to_dense(
    data: xr.DataArray,
    dtype: DTypeLike | None = None,
    chunks: Mapping[Hashable, int] | str | None = "auto",
) -> xr.DataArray:
    """Make dense"""
    if dtype is None:
        dtype = data.dtype
    if chunks is not None:
        data = data.chunk(chunks)
    return xr.apply_ufunc(
        lambda x: x.todense(),
        data,
        dask="parallelized",
        output_dtypes=[dtype],
    )


@xr.register_dataarray_accessor("sp")
class SparseArrAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @property
    def is_sparse(self) -> bool:
        """Return if the data is stored in a sparse format"""
        return isinstance(self._obj.data, AnySparseArray)

    @property
    def array(self) -> AnySparseArray | None:
        """Return the sparse array, if the object is sparse"""
        if not self.is_sparse:
            return None
        return self._obj.data

    def to_sparse(self, override: bool = False, **to_sparse_kwargs) -> xr.DataArray:
        """Sparsify the object. Does nothing if it is already sparse"""
        if self.is_sparse and not override:
            return self._obj
        return to_sparse(data=self._obj, **to_sparse_kwargs)

    def to_dense(self) -> xr.DataArray:
        """Densify the object. Does nothing if it is already dense"""
        if not self.is_sparse:
            return self._obj
        return to_dense(data=self._obj)


@xr.register_dataset_accessor("sp")
class SparseSetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @property
    def sparsity(self) -> dict[Hashable, bool]:
        """Return which variables are stored in sparse format"""
        return {var: arr.sp.is_sparse for var, arr in self._obj.data_vars.items()}

    @property
    def is_sparse(self) -> bool:
        """True if all data variables are sparse"""
        return all(self.sparsity.values())

    def to_sparse(self, override: bool = False, **to_sparse_kwargs) -> xr.Dataset:
        """Sparsify the object"""
        return self._obj.map(
            lambda da: da.sp.to_sparse(override=override, **to_sparse_kwargs)
        )

    def to_dense(self) -> xr.Dataset:
        """Densify the object"""
        return self._obj.map(lambda da: da.sp.to_dense())


@xr.register_datatree_accessor("sp")
class SparseTreeAccessor:
    def __init__(self, xarray_obj: xr.DataTree):
        self._obj = xarray_obj

    @property
    def sparsity(self) -> dict[str, dict[Hashable, bool]]:
        """Return which variables are stored in sparse format"""
        return {
            path: node.to_dataset().sp.sparsity
            for path, node in self._obj.subtree_with_keys
        }

    @property
    def is_sparse(self) -> bool:
        """True if all data variables are sparse"""
        return any(node.to_dataset().sp.is_sparse for node in self._obj.subtree)

    def to_sparse(self, override: bool = False, **to_sparse_kwargs) -> xr.DataTree:
        """Sparsify the object"""
        return map_over_datasets(
            lambda ds: ds.sp.to_sparse(override=override, **to_sparse_kwargs), self._obj
        )

    def to_dense(self) -> xr.DataTree:
        """Densify the object"""
        return map_over_datasets(lambda ds: ds.sp.to_dense(), self._obj)
