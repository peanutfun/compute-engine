"""CLIMADA-CE"""

from contextlib import contextmanager

from . import sparse  # noqa: F401
from dask.distributed import Client

from .aggregates import at_event, average_event_impact
from .engine import Aligner, Engine, align
from .funcs import align_exposure, reproject_hazard
from .impact_funcs import (
    REGISTRY,
    FuncType,
    ImpactFunctionMap,
    ImpactFunctionRegistry,
    impact_function,
)
from .io import open_dataarray, open_dataset, open_datatree
from .tree import (
    dropna_spatial_dims,
    map_aggregate_function,
    map_impact_function,
    map_over_datasets,
    mask_dataset,
    merge_tree_dset,
    split_from_geo,
    split_from_groupby,
    split_from_groupby_bins,
)
from .types import CachePolicy


@contextmanager
def dask_client(n_workers, threads_per_worker, memory_limit, *args, **kwargs):
    """Create a context with a ``dask.distributed.Client``.

    This is a lightweight wrapper and intended to expose only the most important
    parameters to end users.

    Parameters
    ----------
    n_workers : int
        Number of parallel processes to launch.
    threads_per_worker : int
        Compute threads launched by each worker.
    memory_limit : str
        Memory limit for each process. Example: 4 GB can be expressed as ``4000M`` or
        ``4G``.
    args, kwargs
        Additional (keyword) arguments passed to the ``dask.distributed.Client``
        constructor.

    Example
    -------
    >>> with dask_client(n_workers=2, threads_per_worker=2, memory_limit="4G"):
    ...     xr.open_dataset("data.nc", chunks="auto").median()
    """
    # Yield the client with the arguments, and close it afterwards
    client = Client(
        *args,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        **kwargs,
    )
    try:
        yield client
    finally:
        client.close()
