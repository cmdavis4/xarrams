"""Miscellaneous utilities for RAMS grid generation, time handling, and log parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from carlee_tools.types_carlee_tools import PathLike


def get_z_levels(
    deltaz: float,
    dzrat: float,
    dzmax: float,
    nnzp: Optional[int] = None,
    max_height: Optional[float] = None,
) -> np.ndarray:
    """Generate RAMS vertical grid levels using a stretched-grid algorithm.

    Starting from a sub-ground level at ``-deltaz/2``, each successive level
    is spaced by ``min(deltaz * dzrat, dzmax)`` until *nnzp* levels are
    reached or *max_height* is exceeded.

    Args:
        deltaz: Initial vertical spacing (m).
        dzrat: Grid stretch ratio applied at each level.
        dzmax: Maximum allowed spacing (m).
        nnzp: Target number of vertical levels (mutually exclusive with *max_height*).
        max_height: Target domain top (m) (mutually exclusive with *nnzp*).

    Returns:
        1-D array of z-coordinate values (m).

    Raises:
        ValueError: If neither *nnzp* nor *max_height* is provided.
    """
    if not nnzp and not max_height:
        raise ValueError("Must pass one of nnzp or max_height")

    heights: list[float] = [-deltaz / 2, deltaz / 2]

    def need_more() -> bool:
        if nnzp:
            return len(heights) <= nnzp
        return heights[-1] < max_height  # type: ignore[operator]

    while need_more():
        deltaz = min(deltaz * dzrat, dzmax)
        heights.append(heights[-1] + deltaz)

    return np.array(heights)


def to_t_minutes(
    time_values: Union[np.ndarray, xr.DataArray, list],
    start_time: "np.datetime64 | xr.DataArray",
) -> Union[np.ndarray, xr.DataArray, list[int]]:
    """Convert time values to minutes elapsed since *start_time*.

    Args:
        time_values: Array-like of datetime64 values.
        start_time: Reference start time.

    Returns:
        Minutes elapsed, in the same container type as the input.
    """
    if hasattr(start_time, "to_numpy"):
        start_time = start_time.to_numpy()
    if isinstance(time_values, xr.DataArray):
        return (time_values - start_time).dt.total_seconds() // 60
    if isinstance(time_values, np.ndarray):
        return (time_values - start_time) / np.timedelta64(1, "m") // 1
    return [int((x - start_time) / np.timedelta64(1, "m")) for x in time_values]


def with_t_minutes_coord(
    ds: xr.Dataset,
    start_time: Optional["np.datetime64"] = None,
) -> xr.Dataset:
    """Add a ``t_minutes`` coordinate to *ds* measuring minutes since *start_time*.

    Args:
        ds: Dataset with a ``time`` coordinate.
        start_time: Reference time.  Defaults to the earliest time in *ds*.

    Returns:
        Dataset with a new ``t_minutes`` coordinate.
    """
    start_time = start_time if start_time is not None else ds["time"].min().values
    return ds.assign_coords(t_minutes=to_t_minutes(ds["time"], start_time=start_time))


def parse_rams_stdout_walltimes(
    rams_stdout_path: PathLike,
    plot: bool = True,
) -> tuple[list[float], list[float]]:
    """Extract per-timestep walltime data from a RAMS stdout log file.

    Args:
        rams_stdout_path: Path to the RAMS stdout file.
        plot: If ``True``, create a walltime-vs-simulation-time plot.

    Returns:
        Tuple of ``(simulation_times, walltimes)`` lists (first entry
        dropped as it is typically an outlier).
    """
    sim_times: list[float] = []
    walltimes: list[float] = []
    with Path(rams_stdout_path).open("r") as f:
        for line in f:
            match = re.search(
                r"Timestep.*Sim time\(sec\)=\s*([0-9\.]+).*Wall time\(sec\)=\s*([0-9\.]+)",
                line,
            )
            if match:
                sim_times.append(float(match.group(1)))
                walltimes.append(float(match.group(2)))

    # Drop the first entry — the first timestep is always an outlier
    sim_times = sim_times[1:]
    walltimes = walltimes[1:]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(sim_times, walltimes)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Walltime per timestep (s)")

    return sim_times, walltimes
