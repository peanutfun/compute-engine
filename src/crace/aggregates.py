"""Aggregation functions"""

from abc import ABC, abstractmethod
from typing import Hashable

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .types import DatasetOrArray

# from .tree import DatasetNode


class Aggregate(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def apply(
        self, impact: xr.Dataset, aggregates: xr.Dataset, **kwargs
    ) -> xr.Dataset: ...

    # NOTE: Need kwargs for more complicated functions!
    def __call__(
        self,
        impact: xr.Dataset,
        aggregates: xr.Dataset,
        force_compute: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        if not force_compute and self.name in aggregates:
            return aggregates[self.name].to_dataset()
        return self.apply(impact=impact, aggregates=aggregates, **kwargs)


# class AtEventOLD(Aggregate):
#     @staticmethod
#     def leaf_op(data: DatasetOrArray) -> DatasetOrArray:
#         """The leaf operation"""
#         return data.sum(dim=[data.rio.x_dim, data.rio.y_dim])

#     @staticmethod
#     def merge_op(nodes: Sequence[DatasetOrArray]) -> DatasetOrArray:
#         """How leaves are merged into their parent"""
#         return sum(nodes, start=xr.zeros_like(nodes[0], chunks="auto"))


class AtEvent(Aggregate):
    # @property
    # def name(self) -> Hashable:
    #     return "at_event"

    @property
    def name(self) -> str:
        return "at_event"

    def apply(self, impact: xr.Dataset, aggregates: xr.Dataset) -> xr.Dataset:
        return impact.sum(dim=[impact.rio.x_dim, impact.rio.y_dim])


# at_event = AtEvent()


class AverageAnnualImpact(Aggregate):
    def __init__(self, agg_func=np.mean, time_dim="time"):
        self.agg_func = agg_func
        self.time_dim = time_dim

    @property
    def name(self) -> Hashable:
        return "average_annual_impact"

    def apply(self, impact: xr.Dataset, aggregates: xr.Dataset, **kwargs) -> xr.Dataset:
        time_dim = kwargs.pop("time_dim", self.time_dim)
        agg_func = kwargs.pop("agg_func", self.agg_func)
        event_impact = AtEvent()(impact=impact, aggregates=aggregates)
        return event_impact.groupby(event_impact[time_dim].dt.year).reduce(agg_func)


class AverageEventImpact(Aggregate):
    def __init__(self, agg_func=np.mean, event_dim="event"):
        self.agg_func = agg_func
        self.event_dim = event_dim

    @property
    def name(self) -> Hashable:
        return "average_event_impact"

    def apply(self, impact: xr.Dataset, aggregates: xr.Dataset, **kwargs) -> xr.Dataset:
        event_dim = kwargs.pop("event_dim", self.event_dim)
        agg_func = kwargs.pop("agg_func", self.agg_func)
        event_impact = AtEvent()(impact=impact, aggregates=aggregates)
        return event_impact.reduce(agg_func, dim=event_dim)


def at_event(arr: DatasetOrArray) -> DatasetOrArray:
    """Return the total impact for each event"""
    return arr.sum(dim=[arr.rio.x_dim, arr.rio.y_dim])


# def average_annual_impact(
#     arr: xr.DataArray,
#     time_dim: str = "time",
#     agg_func: Callable[[np.ndarray], np.ndarray] = np.mean,
# ) -> xr.DataArray:
#     """Compute the average annual impact"""
#     event_impact = at_event(arr)
#     return event_impact.groupby(event_impact[time_dim].dt.year).reduce(agg_func)


def average_event_impact(
    arr: DatasetOrArray,
    event_dim: str = "event",
    weights: str | ArrayLike | None = None,
) -> DatasetOrArray:
    """Compute the average impact per event"""
    event_impact = at_event(arr)
    if isinstance(weights, str):
        weights = arr[weights]
    return event_impact.reduce(np.average, weights=weights, dim=event_dim)
