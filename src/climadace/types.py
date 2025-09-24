"""Type definitions"""

from typing import TypeVar
from enum import Enum, auto
from collections.abc import Callable

import xarray as xr

AnyXarray = TypeVar("AnyXarray", xr.DataTree, xr.Dataset, xr.DataArray)
DatasetOrArray = TypeVar("DatasetOrArray", xr.Dataset, xr.DataArray)

class CachePolicy(Enum):
    always = auto()
    if_chunked = auto()
    never = auto()

DatasetFunction = Callable[[xr.Dataset], xr.Dataset]
