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
from .impact_funcs import ImpactFunctionMap
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
def promote_to_dataset(arr: Any, name: str, force_name: bool) -> Any: ...
@overload
def promote_to_dataset(
    arr: xr.DataArray | xr.Dataset, name: str, force_name: bool
) -> xr.Dataset: ...
def promote_to_dataset(arr, name, force_name: bool = False):
    if isinstance(arr, xr.DataArray):
        if arr.name and not force_name:
            name = str(arr.name)
        arr = arr.to_dataset(name=name, promote_attrs=True)
    return arr


@overload
def promote_to_datatree(arr: Any, name: str) -> Any: ...
@overload
def promote_to_datatree(arr: AnyXarray, name: str) -> xr.DataTree: ...
def promote_to_datatree(arr, name):
    arr = promote_to_dataset(arr, name, force_name=False)
    if isinstance(arr, xr.Dataset):
        arr = xr.DataTree(dataset=arr, name=name)
    return arr


# @dataclass
# class EngineOld:
#     # Class input
#     hazard: xr.DataArray
#     exposure: xr.DataArray
#     impact_func: Callable
#     data_dir: InitVar[Path | str] = Path("~/Desktop/ImpactEngine").expanduser()

#     # Set by __post_init__
#     output_dir: Path = field(init=False)
#     event_dims: dict[str, Hashable | None] = field(init=False)

#     # Set later
#     impact: xr.DataArray = field(init=False, default_factory=lambda: xr.DataArray())
#     aggregates: xr.Dataset = field(init=False, default_factory=lambda: xr.Dataset())

#     def __post_init__(self, data_dir):
#         """Initialize"""
#         # Create output directory
#         data_dir = Path(data_dir)
#         data_dir.mkdir(exist_ok=True)
#         self.output_dir = data_dir / datetime.now().isoformat()
#         self.output_dir.mkdir()

#         # Detect dimensions
#         self.event_dims = derive_event_dims(self.hazard)

#     def reproject_hazard(self, cache_result: bool = True) -> xr.DataArray:
#         """Reproject the hazard onto the exposure"""
#         self.hazard = funcs.reproject_hazard(
#             self.hazard, self.exposure, align_chunks=True
#         )
#         if cache_result:
#             self.hazard = funcs.cache_zarr(self.hazard, self.output_dir / "hazard")
#         return self.hazard

#     def compute(self, cache_result: bool = True):
#         """Compute impact"""
#         self.impact = compute_impact(
#             hazard=self.hazard, exposure=self.exposure, impact_func=self.impact_func
#         )
#         if cache_result:
#             self.impact = funcs.cache_zarr(self.impact, self.output_dir / "impact")
#         return self.impact

#     def aggregate(self, cache_result: bool = True):
#         """Compute aggregates"""
#         if not self.impact.data_vars:
#             self.compute(cache_result=False)
#         at_event = self.impact.sum(dim=[self.impact.rio.x_dim, self.impact.rio.y_dim])
#         self.aggregates = xr.Dataset({"at_event": at_event})

#         if self.event_dims["event"] is not None:
#             self.aggregates["average_impact"] = at_event.mean(
#                 dim=self.event_dims["event"]
#             )
#         if self.event_dims["time"] is not None:
#             self.aggregates["average_annual_impact"] = at_event.groupby(
#                 at_event[self.event_dims["time"]].dt.year
#             ).mean()

#         if cache_result:
#             self.aggregates = funcs.cache_zarr(
#                 self.aggregates, self.output_dir / "aggregates"
#             )
#         return self.aggregates


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


# class Engine:
#     # Class input
#     # impact_func: Callable
#     # data_dir: InitVar[Path | str] = Path("~/Desktop/ImpactEngine").expanduser()

#     # Set by __post_init__
#     # output_dir: Path = field(init=False)
#     # event_dims: dict[str, Hashable | None] = field(init=False)

#     def __init__(
#         self,
#         hazard: xr.Dataset | xr.DataArray,
#         # NOTE: Could also be Dataset with multiple, in this case the variables have to
#         #       match. Generally: Load dataset, but use array if only a single variable
#         exposure: xr.DataTree | xr.Dataset | xr.DataArray,
#         impf_map: ImpactFunctionMap | None = None,
#         *,
#         align_exposure: bool = True,
#         reproject_hazard: bool = True,
#         data_dir: Path | str = Path(
#             "~/Desktop/ImpactEngine"
#         ).expanduser()  # Make this a configuration setting
#     ):
#         """Initialize"""

#         # Create output directory
#         data_dir = Path(data_dir)
#         data_dir.mkdir(exist_ok=True)
#         output_dir = data_dir / datetime.now().isoformat()
#         self.paths = EnginePaths.from_base_dir(output_dir)
#         self.paths.base_dir.mkdir()

#         # Detect dimensions
#         self.hazard = hazard
#         self.event_dims = derive_event_dims(self.hazard)
#         self.impf_map = impf_map

#         # Align exposure
#         self.exposure = exposure

#         # Output variables
#         self._impact = None
#         self.aggregates = xr.DataTree()

#         if align_exposure:
#             self.align_exposure()
#             self.exposure = self._maybe_cache(self.exposure, self.paths.exposure)
#         if reproject_hazard:
#             self.reproject_hazard()
#             self.hazard = self._maybe_cache(self.hazard, self.paths.hazard)

#     # TODO: What to do when exposure is tree?
#     def align_exposure(self):
#         """Align events in exposure"""
#         self.exposure = funcs.align_exposure(self.hazard, self.exposure)

#     def reproject_hazard(self):
#         """Reproject the hazard onto the exposure"""
#         self.hazard = funcs.reproject_hazard(self.hazard, self.exposure)

#     @property
#     def impact(self) -> xr.DataTree:
#         """Return or compute the impact"""
#         if self._impact is None:
#             self._impact = self.compute()
#             self._impact = self._maybe_cache(self._impact, self.paths.impact)
#         return self._impact

#     def _maybe_cache(self, data: AnyXarray, path: Path) -> AnyXarray:
#         """Decide if data should be cached. Return the cached or non-cached data"""
#         if funcs.is_chunked(data):
#             data = cache_zarr(arr=data, path=path)
#         return data

#     def compute(self, impf_map: ImpactFunctionMap | None = None) -> xr.DataTree:
#         """Compute impact"""
#         self.hazard = promote_to_dataset(self.hazard)
#         self.exposure = promote_to_datatree(self.exposure, "exposure")

#         # Use default impf_map
#         if impf_map is None:
#             if self.impf_map is None:
#                 raise RuntimeError("No impact function map specified!")
#             impf_map = self.impf_map

#         # Compute
#         impact = map_over_datatree(impf_map, self.hazard) * self.exposure
#         return impact

#     def aggregate(self, cache_result: bool = True):
#         """Compute aggregates"""
#         if not self.impact.data_vars:
#             self.compute(cache_result=False)
#         at_event = self.impact.sum(dim=[self.impact.rio.x_dim, self.impact.rio.y_dim])
#         self.aggregates = xr.Dataset({"at_event": at_event})

#         if self.event_dims["event"] is not None:
#             self.aggregates["average_impact"] = at_event.mean(
#                 dim=self.event_dims["event"]
#             )
#         if self.event_dims["time"] is not None:
#             self.aggregates["average_annual_impact"] = at_event.groupby(
#                 at_event[self.event_dims["time"]].dt.year
#             ).mean()


#         if cache_result:
#             self.aggregates = funcs.cache_zarr(
#                 self.aggregates, self.output_dir / "aggregates"
#             )
#         return self.aggregates
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


# NOTE: For applying impf to hazard, it's better if it is also a tree
# TODO: Add aggregates
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
        impf_map: ImpactFunctionMap | DatasetFunction | Sequence[ImpactFunctionMap],
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
    def impf_map(
        self,
    ) -> ImpactFunctionMap | DatasetFunction | Sequence[ImpactFunctionMap]:
        return self._impf_map

    @impf_map.setter
    def impf_map(
        self, value: ImpactFunctionMap | DatasetFunction | Sequence[ImpactFunctionMap]
    ):
        self._impf_map = value
        self._impact = xr.DataTree()  # Resets impact

    def _compute_single_impact(
        self,
        impf_map: ImpactFunctionMap | DatasetFunction | Sequence[ImpactFunctionMap],
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
        #     {node.path: damage[node.path].ds * node.ds for node in self.exposure.leaves}
        # )

    def _sample_impact(self, samples: pd.DataFrame) -> xr.DataTree:
        impact_samples = (
            self._impact.sel(**sample) for _, sample in samples.iterrows()
        )
        return map_over_datasets(
            lambda *dsets: xr.concat(dsets, dim=pd.Index(samples.index, name="sample")),
            *impact_samples,
        )

    # NOTE: Unnecessary work if impact is computed eagerly!
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
        self, aggregate, *aggregates
    ) -> xr.DataTree | tuple[xr.DataTree, ...]:
        impact = self.impact(samples=None)
        if not aggregates:
            return map_aggregate_function(tree=impact, func_map=aggregate)

        return tuple(
            map_aggregate_function(tree=impact, func_map=agg)
            for agg in (aggregate,) + aggregates
        )
