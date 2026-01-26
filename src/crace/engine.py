"""Compute Engine"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Hashable, Sequence

import pandas as pd
import xarray as xr

from .chunks import is_chunked
from .impact_funcs import FuncType
from .io import maybe_cache_zarr
from .tree import (
    map_aggregate_function,
    map_impact_function,
    map_over_datasets,
    tree_is_empty,
)
from .types import AnyXarray, CachePolicy, DatasetFunction, DatasetOrArray

BASE_DIR = Path("~/Desktop/ImpactEngine").expanduser()


def get_workdir(create: bool = False):
    """Create a new workdir based on the time"""
    workdir = BASE_DIR / datetime.now().isoformat()
    if create:
        workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def compute_impact(
    hazard: DatasetOrArray,
    exposure: xr.Dataset | xr.DataArray,
    impact_func: Callable[[DatasetOrArray], DatasetOrArray],
) -> xr.Dataset | xr.DataArray:
    """The kernel impact computation"""
    return impact_func(hazard) * exposure


def derive_event_dims(
    arr: xr.Dataset | xr.DataArray,
) -> dict[str, Hashable | None]:
    """Derive event dimensions"""

    def find_time_dim(dims: Sequence[Hashable]) -> Hashable | None:
        for dim in dims:
            if hasattr(arr[dim], "dt"):
                return dim
        return None

    def find_event_dim(dims: Sequence[Hashable]) -> Hashable | None:
        event_dim = None
        for dim in dims:
            if dim == "event":
                return dim
            if hasattr(arr[dim], "dt"):
                event_dim = dim
            elif "event" in str(dim):
                event_dim = dim
        return event_dim

    # Search for appropriate dimensions
    non_spatial_dims = set(arr.dims) - {arr.rio.x_dim, arr.rio.y_dim}
    return {
        "time": find_time_dim(sorted(non_spatial_dims)),
        "event": find_event_dim(sorted(non_spatial_dims)),
    }


class EventAlignment(Enum):
    select = auto()
    interpolate = auto()


class Reprojection(Enum):
    resample = auto()
    reproject = auto()


@dataclass
class Strategy:
    event_alignment: EventAlignment
    reprojection: Reprojection


@dataclass
class EnginePaths:
    base_dir: Path
    hazard: Path
    exposure: Path
    impact: Path
    aggregates: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path | str, suffix: str = ".zarr"):
        base_dir = Path(base_dir)
        return cls(
            base_dir=base_dir,
            hazard=(base_dir / "hazard").with_suffix(suffix),
            exposure=(base_dir / "exposure").with_suffix(suffix),
            impact=(base_dir / "impact").with_suffix(suffix),
            aggregates=(base_dir / "aggregates").with_suffix(suffix),
        )


def tree_divide(dset: xr.Dataset, tree: xr.DataTree):
    return xr.DataTree.from_dict(
        {node.path: xr.align(dset, node.ds, join="right")[0] for node in tree.leaves}
    )


ImpactFuncSpecSingle = DatasetFunction | Mapping[str | FuncType, DatasetFunction | str]
ImpactFuncSpec = ImpactFuncSpecSingle | Sequence[ImpactFuncSpecSingle]


# TODO: Add new engine for unsequa/sampling
class Engine:
    """Facility for computing impacts from hazard, exposure, and vulnerability.

    This assumes that hazard and exposure have been aligned.

    Parameters
    ----------
    hazard
        The hazard dataset. Will be split into a data tree that is isomorphic to the
        exposure.
    exposure
        The exposure dataset or data tree. If it is a dataset, it will be promoted
        to a data tree with a single node.
    impf_map : ImpactFuncSpec
        The impact function definition. If it is a mapping, it must be compatible
        with the exposure data tree. If it is a single function, this function
        applied to all nodes of the data tree.

    Attributes
    ----------
    hazard : xarray.DataTree
        The hazard intensity data. Isomorphic to :py:attr:`exposure`.
    exposure : xarray.DataTree
        The exposure value data. Isomorphic to :py:attr:`hazard`.
    impf_map : ImpactFuncSpec
        The impact function definition. Modifying this attribute will reset the stored
        impact. Calling :py:meth:`impact` will then cause the impact to be recomputed.
    """

    def __init__(
        self,
        hazard: xr.Dataset,
        exposure: xr.Dataset | xr.DataTree,
        impf_map: ImpactFuncSpec,
        *,
        workdir: Path | None = None,
        cache_policy: CachePolicy = CachePolicy.never,
    ):
        # Create workdir if needed
        self._cache_policy = cache_policy
        self._workdir = workdir if workdir is not None else get_workdir(create=False)
        self._paths = EnginePaths.from_base_dir(self._workdir)

        self.hazard = self._maybe_cache(hazard, self._paths.hazard)
        self.exposure = self._maybe_cache(exposure, self._paths.exposure)

        # Build aligned DataTrees
        if isinstance(self.exposure, xr.Dataset):
            self.exposure = xr.DataTree(dataset=self.exposure)
        self.hazard = tree_divide(self.hazard, self.exposure)

        self.impf_map = impf_map
        self._impact = xr.DataTree()
        self._aggregates = xr.DataTree()

    def _maybe_cache(self, data: AnyXarray, path: Path):
        """Maybe cache data"""
        return maybe_cache_zarr(arr=data, path=path, cache_policy=self._cache_policy)

    @property
    def impf_map(self) -> ImpactFuncSpec:
        """The impact function definition.

        Modifying this attribute will reset the stored impact. Calling :py:meth:`impact`
        will then cause the impact to be recomputed.
        """
        return self._impf_map

    @impf_map.setter
    def impf_map(self, value: ImpactFuncSpec):
        self._impf_map = value
        self._impact = xr.DataTree()  # Resets impact

    def _compute_single_impact(
        self,
        impf_map: ImpactFuncSpec,
    ) -> xr.DataTree:
        damage = map_impact_function(func_map=impf_map, tree=self.hazard)
        return map_over_datasets(lambda dmg, exp: dmg * exp, damage, self.exposure)

    def _compute_impact(self) -> xr.DataTree:
        # Single function case
        if isinstance(self._impf_map, Mapping) or callable(self._impf_map):
            return self._compute_single_impact(impf_map=self._impf_map)

        # Multi-function case
        impacts = (
            self._compute_single_impact(impf_map=impf_map)
            for impf_map in self._impf_map
        )
        return map_over_datasets(
            lambda *dsets: xr.concat(
                dsets,
                dim=pd.Index(list(range(len(self._impf_map))), name="impact_function"),
            ),
            *impacts,
        )

        # TODO: Use binary operator for trees once this is fixed:
        #       https://github.com/pydata/xarray/issues/10013
        # assert damage.isomorphic(self.exposure)
        # return xr.DataTree.from_dict(
        #     {
        #         node.path: (damage[node.path].ds * node.ds)
        #         for node in self.exposure.leaves
        #     }
        # )

    # TODO: .sel will probably not work as expected because the tree nodes do not share
    #       coordinates! Need to call .sel on each dataset individually.
    def _sample_impact(self, samples: pd.DataFrame) -> xr.DataTree:
        impact_samples = (
            self._impact.sel(**sample) for _, sample in samples.iterrows()
        )
        return map_over_datasets(
            lambda *dsets: xr.concat(dsets, dim=pd.Index(samples.index, name="sample")),
            *impact_samples,
        )

    def impact(
        self,
        impf_map=None,
        *,
        samples: pd.DataFrame | None = None,
        compute: bool = False,
    ) -> xr.DataTree:
        """Compute and return the impact.

        If an impact was already computed, and no other parameters are given, the
        existing impact is returned. Otherwise, the impact is re-computed.

        Notes
        -----
        If either :py:attr:`hazard` or :py:attr:`exposure` contain chunked data, the
        returned impact will be chunked. By default, the dask arrays will **not** be
        computed. Use ``compute`` to control this behavior or call
        :py:meth:`xarray.DataTree.compute` on the retured tree.

        Parameters
        ----------
        impf_map
            The impact function specification. If not ``None``, the impact will be
            recomputed.
        samples
            Specifications for subsampling the impact data structure. Each row of the
            data frame will be interpreted as a sample. Each column name and associated
            value will be passed to :py:meth:`xarray.Dataset.sel` as keyword arguments.
            The values of the index will be used to concatenate the resulting selection
            along a new coordinate called ``"sample"``.
        compute
            If ``True``, call :py:meth:`xarray.DataTree.compute` before returning, but
            only if the data is chunked.

        Returns
        -------
        impact : xarray.DataTree
            The impact calculated from the input hazard, exposure, and impact functions.
            The tree is isomorphic to :py:attr:`exposure`.
        """
        if impf_map is not None:
            self.impf_map = impf_map
        if tree_is_empty(self._impact):
            self._impact = self._compute_impact()
        if samples is not None:
            self._impact = self._sample_impact(samples)

        self._maybe_cache(self._impact, self._paths.impact)
        if compute and is_chunked(self._impact):
            self._impact = self._impact.compute()
        return self._impact

    def aggregate(
        self, aggregate, *aggregates, compute=True
    ) -> xr.DataTree | tuple[xr.DataTree, ...]:
        def map_and_compute(aggregate_func):
            result = map_aggregate_function(tree=self._impact, func_map=aggregate_func)
            # ).sp.to_dense()
            if compute and is_chunked(result):
                result = result.compute()
            return result

        def preprocess():
            self.impact(samples=None)
            if aggregates:
                # Load sparse impact because we compute multiple aggregates
                self._impact = self._impact.sp.to_sparse()
                if is_chunked(self._impact):
                    self._impact.compute()

        preprocess()
        if not aggregates:
            result = map_and_compute(aggregate)
        else:
            result = tuple(
                map_and_compute(aggregate_func=agg) for agg in (aggregate,) + aggregates
            )
        return result

    # def _impact_fill_zero(self):
    #     if tree_is_empty(self._impact):
    #         return

    #     for node in self._impact.subtree:
    #         for da in node.ds.data_vars.values():
    #             if (arr := da.sp.array) is not None and np.isnan(arr.fill_value):
    #                 arr.fill_value = 0.0

    # def _impact_fill_nan(self):
    #     if tree_is_empty(self._impact):
    #         return

    #     for node in self._impact.subtree:
    #         for da in node.ds.data_vars.values():
    #             if (arr := da.sp.array) is not None and arr.fill_value == 0.0:
    #                 arr.fill_value = np.nan
