"""Compute Engine"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence, overload

import pandas as pd
import xarray as xr

from . import funcs
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


@overload
def promote_to_dataset(
    arr: xr.DataArray | xr.Dataset, name: str, force_name: bool
) -> xr.Dataset: ...
@overload
def promote_to_dataset(arr: Any, name: str, force_name: bool) -> Any: ...
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


def align(
    hazard: xr.DataArray | xr.Dataset, exposure: xr.DataArray | xr.Dataset
) -> tuple[xr.Dataset, xr.DataTree]:
    """Align hazard and exposure with default settings"""
    aligner = Aligner(hazard=hazard, exposure=exposure)
    return aligner.get_hazard(), aligner.get_exposure()


class Aligner:
    def __init__(
        self,
        hazard: xr.DataArray | xr.Dataset,
        exposure: xr.DataArray | xr.Dataset,
    ):
        # Align hazard and exposure
        exposure = funcs.align_exposure(hazard, exposure)
        hazard = funcs.reproject_hazard(hazard, exposure)
        exposure, hazard = self.align_names(exposure, hazard, "exposure")
        self._exposure = promote_to_datatree(exposure, name="exposure")
        self._hazard = promote_to_dataset(hazard, name="exposure")

    @staticmethod
    def align_names(
        arr1: DatasetOrArray, arr2: DatasetOrArray, name_fallback
    ) -> tuple[DatasetOrArray, DatasetOrArray]:
        if isinstance(arr1, xr.DataArray):
            if arr1.name is None:
                arr1.name = name_fallback
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

    def get_hazard(self) -> xr.Dataset:
        return self._hazard

    def get_exposure(self) -> xr.DataTree:
        return self._exposure


ImpactFuncSpecSingle = DatasetFunction | Mapping[str | FuncType, DatasetFunction | str]
ImpactFuncSpec = ImpactFuncSpecSingle | Sequence[ImpactFuncSpecSingle]


# TODO: Add new engine for unsequa/sampling
class Engine:
    # hazard: xr.Dataset
    # exposure: xr.DataTree
    # impf_map: ImpactFunctionMap | list[ImpactFunctionMap] = InitVar()

    # output_dir : Path

    # event_dims: dict[str, Hashable] = field(init=False)
    # _impact: xr.DataTree = field(init=False, default_factory=xr.DataTree)
    # _aggregates: xr.DataTree = field(init=False, default_factory=xr.DataTree)
    def __init__(
        self,
        hazard: xr.DataTree | xr.Dataset,  # Makes sense? Tree MUST align with exposure
        exposure: xr.DataTree,
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
        if isinstance(self.hazard, xr.Dataset):
            self.hazard = tree_divide(self.hazard, self.exposure)
        self.impf_map = impf_map
        self._aggregates = xr.DataTree()

    @classmethod
    def from_aligner(
        cls,
        aligner: Aligner,
        impf_map,
        *,
        workdir=None,
        cache_policy=CachePolicy.never,
    ):
        """Create from aligner"""
        return cls(
            hazard=aligner.get_hazard(),
            exposure=aligner.get_exposure(),
            impf_map=impf_map,
            workdir=workdir,
            cache_policy=cache_policy,
        )

    def _maybe_cache(self, data: AnyXarray, path: Path):
        """Maybe cache data"""
        return maybe_cache_zarr(arr=data, path=path, cache_policy=self._cache_policy)

    @property
    def impf_map(self) -> ImpactFuncSpec:
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
        if impf_map is not None:
            self.impf_map = impf_map
        if tree_is_empty(self._impact):
            self._impact = self._compute_impact()
        if samples is not None:
            self._impact = self._sample_impact(samples)

        self._maybe_cache(self._impact, self._paths.impact)
        if compute:
            self._impact = self._impact.compute()
        return self._impact

    def aggregate(
        self, aggregate, *aggregates, compute=True
    ) -> xr.DataTree | tuple[xr.DataTree, ...]:
        def map_and_compute(aggregate_func):
            result = map_aggregate_function(
                tree=self._impact, func_map=aggregate_func
            ).sp.to_dense()
            if compute:
                result = result.compute()
            return result

        self.impact(samples=None)  # Assert impact object exists
        if not aggregates:
            return map_and_compute(aggregate)

        # Load sparse impact because we compute multiple aggregates
        self._impact = self._impact.sp.to_sparse().compute()
        return tuple(
            map_and_compute(aggregate_func=agg) for agg in (aggregate,) + aggregates
        )
