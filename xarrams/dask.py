"""Dask integration utilities for xarray datasets.

Provides task-graph diagnostics and intermediate result caching to
improve performance when working with large, lazy datasets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from carlee_tools.types_carlee_tools import PathLike


def dask_diagnostics(ds: xr.Dataset) -> None:
    """Print per-variable chunk size and memory usage for a dask-backed Dataset.

    Args:
        ds: Dataset whose data variables may be backed by dask arrays.
    """
    for var_name in ds.data_vars:
        var = ds[var_name]
        if hasattr(var.data, "chunksize"):
            chunks = var.data.chunksize
            n_chunks = var.data.npartitions
            bytes_per = np.prod(chunks) * var.dtype.itemsize
            mb_per = bytes_per / 1e6
            total_mb = mb_per * n_chunks
            print(
                f"{var_name:20s}: {chunks} | {n_chunks:6d} chunks |"
                f" {mb_per:6.2f} MB/chunk | {total_mb:8.1f} MB total"
            )


def reload_intermediate(
    ds: Union[xr.Dataset, xr.DataArray],
    cache_dir: PathLike,
    cache_name: Optional[str] = None,
    force: bool = False,
    auto_analyze: bool = True,
    min_graph_depth: int = 5,
    min_tasks: int = 1000,
    verbose: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """Cache a lazy dataset to disk and reload it, breaking up large task graphs.

    Useful when dask scheduler overhead becomes a bottleneck due to deeply
    chained operations or very large task graphs.

    Args:
        ds: Lazy (dask-backed) Dataset or DataArray.
        cache_dir: Directory for ``.zarr`` cache stores.
        cache_name: Basename for the cache file.  If ``None``, a name is
            generated from a hash of the dataset structure.
        force: Always cache, even if analysis suggests it won't help.
        auto_analyze: Inspect the task graph before deciding whether to cache.
        min_graph_depth: Minimum estimated graph depth to trigger caching.
        min_tasks: Minimum task count to trigger caching.
        verbose: Print diagnostic information.

    Returns:
        The reloaded Dataset or DataArray (eagerly computed and read back).
    """
    cache_dir = Path(cache_dir)

    if not cache_name:
        hash_str = hashlib.md5(
            str(ds.dims).encode() + str(sorted(ds.data_vars)).encode()
        ).hexdigest()[:8]
        cache_path = cache_dir / f"cache_{hash_str}.zarr"
    else:
        cache_path = (cache_dir / cache_name).with_suffix(".zarr")

    has_dask = False
    if isinstance(ds, xr.Dataset):
        has_dask = any(hasattr(var.data, "dask") for var in ds.data_vars.values())
    elif isinstance(ds, xr.DataArray):
        has_dask = hasattr(ds.data, "dask")

    if not has_dask:
        if verbose:
            print("Data already computed, skipping cache")
        return ds

    should_cache = force
    if auto_analyze and not force:
        try:
            if isinstance(ds, xr.Dataset):
                graph = ds.__dask_graph__()
            else:
                graph = ds.data.__dask_graph__()

            n_tasks = len(graph)

            if hasattr(graph, "dependencies"):
                dependencies = graph.dependencies
                max_depth = 0

                def get_depth(key: object, visited: set | None = None) -> int:
                    if visited is None:
                        visited = set()
                    if key in visited:
                        return 0
                    visited.add(key)
                    if key not in dependencies:
                        return 1
                    deps = dependencies[key]
                    if not deps:
                        return 1
                    return 1 + max(
                        (get_depth(d, visited.copy()) for d in deps), default=0
                    )

                sample_keys = list(dependencies.keys())[:10]
                depths = [get_depth(k) for k in sample_keys]
                max_depth = max(depths) if depths else 0
            else:
                max_depth = min(n_tasks // 100, 10)

            should_cache = (max_depth >= min_graph_depth) or (n_tasks >= min_tasks)

            if verbose:
                print(f"Task graph analysis: {n_tasks} tasks, estimated depth ~{max_depth}")
                print(f"Caching {'recommended' if should_cache else 'not recommended'}")

        except Exception as e:
            if verbose:
                print(f"Could not analyze graph ({e}), proceeding with cache")
            should_cache = True

    if not should_cache:
        if verbose:
            print("Skipping cache (use force=True to override)")
        return ds

    if cache_path.exists():
        if verbose:
            print(f"Loading from existing cache: {cache_path}")
        return xr.open_zarr(cache_path)

    if verbose:
        print(f"Computing and caching to: {cache_path}")

    ds_computed = ds.compute()
    ds_computed.to_zarr(cache_path, mode="w")

    ds_cached = xr.open_zarr(cache_path)
    if verbose:
        print("Cached and reloaded successfully")

    return ds_cached
