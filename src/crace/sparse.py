"""Sparse operations"""

from functools import partial
from typing import Callable, Hashable, Mapping

import numpy as np
import sparse as sp
import xarray as xr
from numpy.typing import ArrayLike, DTypeLike

from .chunks import is_chunked
from .tree import map_over_datasets
from .types import AnyXarray

AnySparseArray = sp.GCXS | sp.COO | sp.DOK

MEMORY_BUDGET = 100_000_000  # 100 MB

# --- Sparse interoperability --- #


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    return sp.elemwise(
        partial(np.isclose, rtol=rtol, atol=atol, equal_nan=equal_nan), a, b
    )


def interp(x, xp, fp, left=None, right=None, period=None):
    # Densify sparse interpolants
    if isinstance(xp, sp.SparseArray):
        xp = xp.todense()
    if isinstance(fp, sp.SparseArray):
        fp = fp.todense()

    def interp_func(xx):
        return np.interp(xx, xp, fp, left=left, right=right, period=period)

    # Shortcut for dense arrays
    if not isinstance(x, sp.SparseArray):
        return interp_func(x)

    # Define output type
    out_kwargs = {}
    out_type = sp.COO
    if isinstance(x, sp.GCXS):
        out_type = sp.GCXS
        out_kwargs["compressed_axes"] = x.compressed_axes
    elif isinstance(x, sp.DOK):
        out_type = sp.DOK

    # Perform interpolation on sparse object
    arr = sp.as_coo(x)
    data = interp_func(arr.data)
    fill_value = interp_func(arr.fill_value)
    return sp.COO(
        data=data, coords=arr.coords, shape=arr.shape, fill_value=fill_value, prune=True
    ).asformat(out_type, **out_kwargs)


if not hasattr(sp, "interp"):
    sp.interp = interp
if not hasattr(sp, "isclose"):
    sp.isclose = isclose

# --- #


def zero_to_nan(array: ArrayLike, exact: bool = False) -> np.ndarray:
    """Make zeros to NaNs"""
    # Exact or inexact comparison
    compare_op = np.equal
    if not exact:
        compare_op = np.isclose

    # Compare
    array = np.where(compare_op(array, 0), np.nan, array)
    return array


# class NaNArray(sp.GCXS):
#     """Specialization of a the sparse array with fixed, non-default fill value"""

#     fill_value = np.nan

#     def __init__(self, *args, **kwargs):
#         """Initialize"""
#         kwargs["fill_value"] = self.fill_value
#         super().__init__(*args, **kwargs)


def to_sparse(
    data: xr.DataArray,
    array_type: type[AnySparseArray] = sp.GCXS,
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

    def to_dense(self, **to_dense_kwargs) -> xr.DataArray:
        """Densify the object. Does nothing if it is already dense"""
        if not self.is_sparse:
            return self._obj
        return to_dense(data=self._obj, **to_dense_kwargs)

    @property
    def nbytes_dense(self) -> int:
        try:
            itemsize = self._obj.dtype.itemsize
        except AttributeError:
            itemsize = 8  # Default for numerical 64-bit values
        return int(self._obj.size * itemsize)

    def load(self, max_nbytes: int = MEMORY_BUDGET, **to_sparse_kwargs) -> xr.DataArray:
        return sparse_load(self._obj, max_nbytes=max_nbytes, **to_sparse_kwargs)

    def maybe_densify(
        self, max_nbytes: int = MEMORY_BUDGET, chunks=None
    ) -> xr.DataArray:
        return maybe_densify(self._obj, max_nbytes=max_nbytes, chunks=chunks)


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
        if sparcity := self.sparsity:
            return all(sparcity.values())
        return False

    def to_sparse(self, override: bool = False, **to_sparse_kwargs) -> xr.Dataset:
        """Sparsify the object"""
        return self._obj.map(
            lambda da: da.sp.to_sparse(override=override, **to_sparse_kwargs)
        )

    def to_dense(self, **to_dense_kwargs) -> xr.Dataset:
        """Densify the object"""
        return self._obj.map(lambda da: da.sp.to_dense(**to_dense_kwargs))

    @property
    def nbytes_dense(self) -> int:
        return sum(arr.sp.nbytes_dense for arr in self._obj.data_vars.values())

    def load(self, max_nbytes: int = MEMORY_BUDGET, **to_sparse_kwargs) -> xr.Dataset:
        return sparse_load(self._obj, max_nbytes=max_nbytes, **to_sparse_kwargs)

    def maybe_densify(self, max_nbytes: int = MEMORY_BUDGET, chunks=None) -> xr.Dataset:
        return maybe_densify(self._obj, max_nbytes=max_nbytes, chunks=chunks)


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

    def to_dense(self, **to_dense_kwargs) -> xr.DataTree:
        """Densify the object"""
        return map_over_datasets(
            lambda ds: ds.sp.to_dense(**to_dense_kwargs), self._obj
        )

    @property
    def nbytes_dense(self) -> int:
        return sum(node.ds.sp.nbytes_dense for node in self._obj.subtree)

    def load(self, max_nbytes: int = MEMORY_BUDGET, **to_sparse_kwargs) -> xr.DataTree:
        return sparse_load(self._obj, max_nbytes=max_nbytes, **to_sparse_kwargs)

    def maybe_densify(
        self, max_nbytes: int = MEMORY_BUDGET, chunks=None
    ) -> xr.DataTree:
        return maybe_densify(self._obj, max_nbytes=max_nbytes, chunks=chunks)


def sparse_load(obj: AnyXarray, max_nbytes: int, **to_sparse_kwargs) -> AnyXarray:
    """Load an xarray object into memory, either sparse or dense"""
    if not is_chunked(obj):
        return obj
    if obj.sp.nbytes_dense > max_nbytes:
        obj = obj.sp.to_sparse(**to_sparse_kwargs)

    return obj.compute()


def maybe_densify(obj: AnyXarray, max_nbytes: int, chunks) -> AnyXarray:
    """Maybe densify a sparse object based on maximum memory"""
    if not obj.sp.is_sparse or obj.sp.nbytes_dense > max_nbytes:
        return obj

    print("Returning dense")
    return obj.sp.to_dense(chunks=chunks)
