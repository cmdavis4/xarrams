"""File I/O for reading and parsing RAMS model output.

Handles reading RAMS HDF5 output files into xarray Datasets, parsing
filenames for datetime and grid information, and mapping phony HDF5
dimensions to meaningful coordinate names using header files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional, Union
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm

from carlee_tools.types_carlee_tools import PathLike, DatetimeLike
from carlee_tools.dt import dt_to_str, str_to_dt

from .constants import (
    CM1_TO_RAMS_VARIABLE_NAMES,
    HEADER_NAME_DIMENSION_DICT,
    RAMS_FILENAME_DT_STRFTIME_FORMAT,
    RAMS_FILENAME_DT_REGEX,
    RAMS_VARIABLES_DF,
    ureg,
)


def get_datetime_from_rams_filename(
    rams_output_filepath: PathLike,
    filename_datetime_regex: str = RAMS_FILENAME_DT_REGEX,
) -> "datetime.datetime":
    """Extract the datetime from a RAMS output filename.

    Args:
        rams_output_filepath: Path to a RAMS output file (e.g. ``a-A-2020-01-01-120000-g1.h5``).
        filename_datetime_regex: Regex matching the datetime portion of the filename.

    Returns:
        Parsed datetime object.

    Raises:
        ValueError: If the filename does not contain a recognizable datetime.
    """
    import datetime  # noqa: F811 – local for type hint above

    name = Path(rams_output_filepath).name
    match = re.search(filename_datetime_regex, name)
    if not match:
        raise ValueError(
            f"Unable to parse datetime from filepath {rams_output_filepath}"
        )
    return str_to_dt(match.group(0))


def to_ramsin_parameters(dt_like: DatetimeLike):
    # Coerce to a pd.Timestamp (accepts datetime, np.datetime64, and pd.Timestamp)
    ts = pd.Timestamp(dt_like)
    return {
        "IYEAR1": str(ts.year).zfill(4),
        "IMONTH1": str(ts.month).zfill(2),
        "IDATE1": str(ts.day).zfill(2),
        "ITIME1": str(ts.hour).zfill(2) + str(ts.minute).zfill(2),
    }


def get_grid_number(rams_output_filepath: PathLike) -> int:
    """Extract the grid number from a RAMS output filename.

    Args:
        rams_output_filepath: Path to a RAMS output file.

    Returns:
        Integer grid number (e.g. ``1`` for ``g1``).

    Raises:
        ValueError: If the grid number cannot be parsed.
    """
    match = re.search(r"g([1-9]+)\.h5", Path(rams_output_filepath).name)
    if not match:
        raise ValueError(
            f"Unable to parse grid number from filepath {rams_output_filepath}"
        )
    return int(match.group(1))


def to_rams_output_filename(
    this_dt: DatetimeLike, lite: bool = False, grid: int = 1, header=False
) -> str:
    """Build a RAMS output filename for a given datetime, file type, and grid.

    Args:
        this_dt: Datetime of the output file.
        lite: If ``True``, generate a lite-file (``a-L-...``) name; otherwise analysis (``a-A-...``).
        grid: Grid number.

    Returns:
        Filename string (e.g. ``a-A-2020-01-01-120000-g1.h5``).
    """
    tail = "head.txt" if header else f"g{grid}.h5"
    return (
        f"a-{'L' if lite else 'A'}-{dt_to_str(this_dt, date_format=RAMS_FILENAME_DT_STRFTIME_FORMAT)}-{tail}"
    )


def to_header_filepath(rams_output_filepath: PathLike) -> Path:
    """Convert a RAMS data-file path to its corresponding header-file path.

    Args:
        rams_output_filepath: Path to a RAMS ``.h5`` data file.

    Returns:
        Path to the matching ``*-head.txt`` header file.
    """
    p = Path(rams_output_filepath)
    return p.with_name(re.sub(r"g[1-9]+\.h5", "head.txt", p.name))


def get_rams_dimension_values(
    header_filepath: PathLike,
    grid_number: int = 1,
) -> dict[str, list[float]]:
    """Read coordinate values (x, y, z) from a RAMS header file.

    Args:
        header_filepath: Path to a RAMS ``*-head.txt`` file.
        grid_number: Grid number to look up in the header.

    Returns:
        Dictionary mapping dimension names (``"x"``, ``"y"``, ``"z"``) to lists of
        coordinate values.
    """
    dimension_vals: dict[str, list[float]] = {}
    with Path(header_filepath).open("r") as f:
        header_name_dimension_dict = {
            k.format(grid_number=str(grid_number).zfill(2)): v
            for k, v in HEADER_NAME_DIMENSION_DICT.items()
        }
        while header_name_dimension_dict:
            for line in f:
                line = line.strip()
                if line in header_name_dimension_dict:
                    this_header_name = line
                    break
            n_levels = int(next(f).strip())
            levels = [float(next(f).strip()) for _ in range(n_levels)]
            dimension_vals[header_name_dimension_dict[this_header_name]] = (
                levels
            )
            header_name_dimension_dict.pop(this_header_name)
    return dimension_vals


_RAMS_DIM_TO_STANDARD: dict[str, str] = {"nx": "x", "ny": "y", "nz": "z"}
"""RAMS Fortran dimension labels (from rams_variables.csv) → header standard names."""


def infer_rams_dimensions(
    single_time_rams_ds: xr.Dataset,
    grid_number: int = 1,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Infer the mapping from phony dimensions to real names using known variable shapes.

    For each data variable in the dataset, looks up its expected RAMS dimension
    spec in :data:`RAMS_VARIABLES_DF` (Fortran order, e.g. ``"nx,ny,nz"``),
    reverses it to the on-disk/numpy axis order, and pairs each phony dim
    positionally with the corresponding standard name. Votes are aggregated
    across every recognized variable; the most common vote wins so a single
    bad CSV entry can't poison the mapping.

    This is robust when ``nx == ny`` (length-based matching is ambiguous there)
    and is independent of the order in which variables appear in the file —
    important for lite files, where ``LITE_VARS`` can change which variable
    h5netcdf encounters first and therefore which ``phony_dim_N`` ends up
    referring to which physical axis.

    Args:
        single_time_rams_ds: A single-timestep RAMS dataset (must have ``encoding["source"]``).
        grid_number: Grid number used to look up coordinate values in the header file.

    Returns:
        ``(dim_names_mapping, dimension_values)`` where *dim_names_mapping* maps
        dataset (phony) dimension names to standard names (``x``, ``y``, ``z``)
        and *dimension_values* maps standard names to coordinate arrays read
        from the header.

    Raises:
        ValueError: If no recognized variables are present or if ``x``/``y``/``z``
            cannot all be located.
    """
    from collections import Counter

    header_filepath = to_header_filepath(single_time_rams_ds.encoding["source"])
    dimension_vals = get_rams_dimension_values(
        header_filepath, grid_number=grid_number
    )

    known_dims: dict[str, str] = (
        RAMS_VARIABLES_DF.dropna(subset=["dimensions"])
        .set_index("name")["dimensions"]
        .to_dict()
    )

    # Tally per-phony-dim votes for each standard name across all recognized vars.
    votes: dict[str, Counter] = {}
    n_recognized = 0
    for var_name in single_time_rams_ds.data_vars:
        spec = known_dims.get(str(var_name))
        if not spec:
            continue
        rams_dims = [d.strip() for d in spec.split(",")]
        if not all(d in _RAMS_DIM_TO_STANDARD for d in rams_dims):
            # Variable references a dim we don't have header values for
            # (e.g. ``np``, ``nzg``, ``nkr``); skip — those phony dims are
            # handled by ``keep_unknown_dims`` downstream.
            on_disk_order = list(reversed(rams_dims))
        else:
            on_disk_order = list(reversed(rams_dims))
        ds_dims = single_time_rams_ds[var_name].dims
        if len(ds_dims) != len(on_disk_order):
            continue
        n_recognized += 1
        for ds_dim, rams_dim in zip(ds_dims, on_disk_order, strict=True):
            standard = _RAMS_DIM_TO_STANDARD.get(rams_dim)
            if standard is None:
                continue
            votes.setdefault(ds_dim, Counter())[standard] += 1

    if n_recognized == 0:
        raise ValueError(
            "Cannot infer dimension names: no variables in this dataset are"
            " present in RAMS_VARIABLES_DF."
        )

    # Resolve votes: take the majority winner for each phony dim. Conflicts are
    # tolerated (e.g. RCO2P is mis-listed in the CSV as ``nz,nx,ny``) but warn.
    dim_names_mapping: dict[str, str] = {}
    for ds_dim, counter in votes.items():
        winner, winner_count = counter.most_common(1)[0]
        total = sum(counter.values())
        if winner_count != total:
            print(
                f"Warning: conflicting dim inference for {ds_dim}:"
                f" {dict(counter)}. Using majority winner {winner!r}."
            )
        dim_names_mapping[ds_dim] = winner

    # Sanity check: every standard name should be assigned to exactly one phony dim.
    assigned = list(dim_names_mapping.values())
    expected = set(_RAMS_DIM_TO_STANDARD.values())
    missing = expected - set(assigned)
    if missing:
        raise ValueError(
            f"Could not infer phony dim mapping for {sorted(missing)}; no"
            " recognized variables in the dataset reference these dimensions."
        )
    duplicates = [name for name in expected if assigned.count(name) > 1]
    if duplicates:
        raise ValueError(
            f"Inferred dimension mapping is inconsistent: {sorted(duplicates)}"
            f" assigned to multiple phony dims. Mapping: {dim_names_mapping}"
        )

    return dim_names_mapping, dimension_vals


def fill_rams_output_dimensions(
    ds: xr.Dataset,
    dimension_names_mapping: dict[str, str],
    dimension_values: dict[str, list[float]],
) -> xr.Dataset:
    """Rename phony dimensions and assign coordinate values from a header file.

    Args:
        ds: xarray Dataset with RAMS output data.
        dimension_names_mapping: Mapping from current (phony) dim names to standard names.
        dimension_values: Mapping from standard dim names to coordinate value lists.

    Returns:
        Dataset with proper dimension names and coordinate values.

    Raises:
        ValueError: If dimension lengths don't match between dataset and header.
    """
    try:
        ds = ds.rename_dims(dimension_names_mapping).assign_coords(
            dimension_values
        )
    except ValueError:
        print(
            "Mismatch between dimension lengths in dataset and header;\nPassed"
            f" dimension dict: {dimension_names_mapping}\nDimension sizes in"
            f" dataset: {ds.dims}\nDimension lengths from header:"
            f" { {k: len(v) for k, v in dimension_values.items()} }"
        )
        raise
    return ds


def read_rams_output(
    input_filenames: list[PathLike],
    fill_dim_names: bool = True,
    dim_names: Optional[dict[str, str]] = None,
    keep_unknown_dims: bool = False,
    drop_vars: Optional[list[str]] = None,
    keep_vars: Optional[list[str]] = None,
    preprocess: Optional[Callable[..., xr.Dataset]] = None,
    time_dim_name: str = "time",
    parallel: bool = True,
    chunks: Union[str, dict[str, int]] = "auto",
    concatenate: bool = True,
    silent: bool = False,
    open_dataset_kwargs: Optional[dict[str, Any]] = None,
    filename_datetime_regex: str = RAMS_FILENAME_DT_REGEX,
    units: bool = False,
) -> Union[xr.Dataset, list[xr.Dataset]]:
    """Read one or more RAMS HDF5 output files into an xarray Dataset.

    Handles dimension renaming, coordinate assignment from header files,
    time coordinate construction from filenames, and optional unit attachment.

    Args:
        input_filenames: Paths to RAMS ``.h5`` output files.
        fill_dim_names: Whether to rename phony dimensions to real names.
        dim_names: Explicit dimension name mapping. If ``None``, the phony-dim
            mapping is inferred per-file via ``infer_rams_dimensions`` (vote-based
            on known variable shapes) for both analysis and lite files.
        keep_unknown_dims: If ``False``, drop variables with unrecognized
            phony dimensions after renaming.
        drop_vars: Variables to exclude. Mutually exclusive with *keep_vars*.
        keep_vars: Variables to keep (all others dropped). Mutually exclusive with *drop_vars*.
        preprocess: Callable applied to each single-timestep dataset before concatenation.
        time_dim_name: Name for the time dimension.
        parallel: Use dask for parallel reading (requires dask).
        chunks: Chunk specification passed to ``xr.open_mfdataset``.
        concatenate: Whether to concatenate files along the time dimension.
        silent: Suppress progress output.
        open_dataset_kwargs: Extra keyword arguments for ``xr.open_dataset``.
        filename_datetime_regex: Regex for extracting datetimes from filenames.
        units: Attach pint units from the RAMS variable table.

    Returns:
        An xarray Dataset (or list of Datasets if ``concatenate=False``).

    Raises:
        ValueError: If both *drop_vars* and *keep_vars* are provided.
    """
    drop_vars = drop_vars or []
    keep_vars = keep_vars or []
    open_dataset_kwargs = open_dataset_kwargs or {}

    # Convert to list in case it's a generator
    input_filenames = list(input_filenames)

    if drop_vars and keep_vars:
        raise ValueError("Cannot pass both drop_vars and keep_vars")

    # When the caller hasn't pinned an explicit phony-dim mapping, leave
    # ``dim_names`` as None so ``infer_rams_dimensions`` runs per-file. The old
    # behavior forced a hardcoded positional map (RAMS_ANALYSIS_FILE_DIMENSIONS_DICT)
    # for analysis (a-A) files, which assumed phony_dim_2 was always the vertical
    # axis. That assumption breaks whenever surface/canopy/soil variables (e.g.
    # CAN_RVAP, SFCWATER_DEPTH) introduce a small patch/soil dimension and shift
    # the phony-dim numbering, mislabeling the length-2 patch dim as ``z``.
    # Vote-based inference handles this (and the nx == ny ambiguity) correctly.

    def maybe_print(x: str) -> None:
        if not silent:
            print(x)

    if parallel:
        try:
            import dask  # noqa: F401
        except ImportError:
            print(
                "dask must be installed to use the `parallel` option; falling"
                " back to serial"
            )
            parallel = False

    input_filenames = [Path(x) for x in input_filenames]
    input_datetimes = []
    for fpath in input_filenames:
        time = get_datetime_from_rams_filename(
            fpath, filename_datetime_regex=filename_datetime_regex
        )
        if not time:
            raise ValueError(
                f"File {fpath.name} does not contain a valid timestamp in the"
                " filename"
            )
        input_datetimes.append(time)

    if keep_vars:
        print("Determining drop_vars from keep_vars...")
        present_vars = xr.open_dataset(input_filenames[0]).data_vars
        drop_vars = [x for x in present_vars if x not in keep_vars]

    def _sanitized_preprocess(ds: xr.Dataset) -> xr.Dataset:
        if fill_dim_names:
            if dim_names:
                _dimension_names = dim_names
                dimension_values = get_rams_dimension_values(
                    header_filepath=to_header_filepath(ds.encoding["source"]),
                    grid_number=get_grid_number(ds.encoding["source"]),
                )
            else:
                _dimension_names, dimension_values = infer_rams_dimensions(
                    single_time_rams_ds=ds,
                    grid_number=get_grid_number(ds.encoding["source"]),
                )
            ds = fill_rams_output_dimensions(
                ds=ds,
                dimension_names_mapping=_dimension_names,
                dimension_values=dimension_values,
            )
        if not keep_unknown_dims:
            phony_dims = [dim for dim in ds.dims if dim.startswith("phony_")]
            if phony_dims:
                vars_with_phony_dims = [
                    var
                    for var in ds.data_vars
                    if any(pd in ds[var].dims for pd in phony_dims)
                ]
                ds = ds.drop_vars(vars_with_phony_dims)
        if preprocess:
            ds = preprocess(ds)
        return ds

    if parallel:
        maybe_print(
            f"Reading and concatenating {len(input_filenames)} individual"
            " timestep outputs..."
        )
        from contextlib import nullcontext

        from dask.diagnostics import ProgressBar

        open_ds_context_manager = nullcontext if silent else ProgressBar
        with open_ds_context_manager():
            default_open_mfdataset_kwargs = {
                "combine": "nested",
                "phony_dims": "sort",
                "engine": "h5netcdf",
                "parallel": True,
                "chunks": chunks,
            }
            ds = xr.open_mfdataset(
                input_filenames,
                concat_dim=time_dim_name,
                preprocess=_sanitized_preprocess,
                drop_variables=drop_vars,
                **(default_open_mfdataset_kwargs | open_dataset_kwargs),
            )
    else:
        maybe_print(
            f"Reading {len(input_filenames)} individual timestep outputs..."
        )
        to_concat: list[xr.Dataset] = []
        wrapped_to_read = (
            tqdm(input_filenames) if not silent else input_filenames
        )
        for ds_path in wrapped_to_read:
            ds = xr.open_dataset(
                ds_path,
                drop_variables=drop_vars,
                engine="h5netcdf",
                phony_dims="sort",
                **open_dataset_kwargs,
            )
            to_concat.append(ds)
        if len(to_concat) > 1:
            if concatenate:
                maybe_print("Concatenating along time...")
                ds = xr.concat(to_concat, dim=time_dim_name)
            else:
                ds = to_concat  # type: ignore[assignment]
        else:
            ds = to_concat[0]

    source_files = [str(f) for f in input_filenames]
    ds = ds.assign_coords(**{
        time_dim_name: input_datetimes,
        "source_file": (time_dim_name, source_files),
    })
    ds = ds.sortby(time_dim_name)

    if parallel:
        ds = ds.unify_chunks()

    if units:
        ds = ds.pint.quantify(
            RAMS_VARIABLES_DF.set_index("name")["units"].to_dict(),
            unit_registry=ureg,
        )

    rams_attrs_dicts = RAMS_VARIABLES_DF.set_index("name").to_dict(
        orient="index"
    )
    for var in ds.data_vars:
        ds[var] = ds[var].assign_attrs(rams_attrs_dicts.get(var, {}))

    return ds


CM1_DEFAULT_START_DATETIME: str = "2000-01-01 00:00:00"
"""Default anchor for converting CM1's seconds-since-start to absolute datetimes."""

_CM1_SCALAR_GRID_RENAME: dict[str, str] = {"xh": "x", "yh": "y", "zh": "z"}
"""CM1 scalar-grid dimensions renamed to match RAMS conventions. Flux-grid
dimensions (``xf``/``yf``/``zf``) are left as-is."""


def read_cm1_output(
    input_filenames: list[PathLike],
    drop_vars: Optional[list[str]] = None,
    keep_vars: Optional[list[str]] = None,
    preprocess: Optional[Callable[..., xr.Dataset]] = None,
    time_dim_name: str = "time",
    parallel: bool = True,
    chunks: Union[str, dict[str, int]] = "auto",
    concatenate: bool = True,
    silent: bool = False,
    open_dataset_kwargs: Optional[dict[str, Any]] = None,
) -> Union[xr.Dataset, list[xr.Dataset]]:
    """Read one or more CM1 netCDF output files into an xarray Dataset.

    CM1 output already carries a ``time`` dimension (seconds since the start
    of the simulation) and proper spatial coordinates: ``xh``/``yh``/``zh``
    on the scalar grid and ``xf``/``yf``/``zf`` on the flux grid. This
    function concatenates files along ``time``, converts the raw seconds to
    absolute datetimes anchored at *start_datetime*, and optionally renames
    the scalar-grid dims to ``x``/``y``/``z`` to match the convention used
    by :func:`read_rams_output`. Flux-grid dims are left untouched.

    Args:
        input_filenames: Paths to CM1 ``cm1out_*.nc`` output files.
        start_datetime: Absolute datetime corresponding to t=0 in the simulation.
        rename_dims: If True, rename ``xh``/``yh``/``zh`` → ``x``/``y``/``z``.
        drop_vars: Variables to exclude. Mutually exclusive with *keep_vars*.
        keep_vars: Variables to keep (all others dropped). Mutually exclusive with *drop_vars*.
        preprocess: Callable applied to each single-file dataset before concatenation.
            Receives the dataset *after* any rename, so callers should refer to
            ``x``/``y``/``z`` when ``rename_dims=True``.
        time_dim_name: Name of the time dimension in CM1 output.
        parallel: Use dask for parallel reading (requires dask).
        chunks: Chunk specification passed to ``xr.open_mfdataset``.
        concatenate: Whether to concatenate files along the time dimension.
        silent: Suppress progress output.
        open_dataset_kwargs: Extra keyword arguments for ``xr.open_dataset``.

    Returns:
        An xarray Dataset (or list of Datasets if ``concatenate=False`` and
        more than one file was passed).

    Raises:
        ValueError: If both *drop_vars* and *keep_vars* are provided.
    """
    drop_vars = drop_vars or []
    keep_vars = keep_vars or []
    open_dataset_kwargs = open_dataset_kwargs or {}

    input_filenames = [Path(x) for x in list(input_filenames)]

    if drop_vars and keep_vars:
        raise ValueError("Cannot pass both drop_vars and keep_vars")

    def maybe_print(x: str) -> None:
        if not silent:
            print(x)

    if parallel:
        try:
            import dask  # noqa: F401
        except ImportError:
            print(
                "dask must be installed to use the `parallel` option; falling"
                " back to serial"
            )
            parallel = False

    if keep_vars:
        maybe_print("Determining drop_vars from keep_vars...")
        present_vars = xr.open_dataset(input_filenames[0]).data_vars
        drop_vars = [x for x in present_vars if x not in keep_vars]

    def _sanitized_preprocess(ds: xr.Dataset) -> xr.Dataset:
        if preprocess:
            ds = preprocess(ds)
        return ds

    if parallel:
        maybe_print(
            f"Reading and concatenating {len(input_filenames)} individual"
            " timestep outputs..."
        )
        from contextlib import nullcontext

        from dask.diagnostics import ProgressBar

        base_open_dataset_kwargs = {
            "parallel": True,
            "engine": "h5netcdf",
            "data_vars": "all",
            "combine": "nested",
        }
        open_ds_context_manager = nullcontext if silent else ProgressBar
        with open_ds_context_manager():
            ds = xr.open_mfdataset(
                input_filenames,
                concat_dim=time_dim_name,
                preprocess=_sanitized_preprocess,
                drop_variables=drop_vars,
                chunks=chunks,
                **(base_open_dataset_kwargs | open_dataset_kwargs),
            )
    else:
        maybe_print(
            f"Reading {len(input_filenames)} individual timestep outputs..."
        )
        to_concat: list[xr.Dataset] = []
        wrapped_to_read = (
            tqdm(input_filenames) if not silent else input_filenames
        )
        for ds_path in wrapped_to_read:
            ds = xr.open_dataset(
                ds_path,
                drop_variables=drop_vars,
                engine="h5netcdf",
                **open_dataset_kwargs,
            )
            ds = _sanitized_preprocess(ds)
            to_concat.append(ds)
        if len(to_concat) > 1:
            if concatenate:
                maybe_print("Concatenating along time...")
                ds = xr.concat(to_concat, dim=time_dim_name)
            else:
                return to_concat
        else:
            ds = to_concat[0]

    # Tag each timestep with its source file when there's one timestep per file
    # (the CM1 default); skip otherwise rather than guess the mapping.
    if len(input_filenames) == ds.sizes.get(time_dim_name, -1):
        ds = ds.assign_coords({
            "source_file": (time_dim_name, [str(f) for f in input_filenames]),
        })

    ds = ds.sortby(time_dim_name)

    if parallel:
        ds = ds.unify_chunks()

    return ds


# def rename_cm1_to_rams_vars(
#     ds: xr.Dataset,
#     mapping: Optional[dict[str, str]] = None,
# ) -> xr.Dataset:
#     """Rename CM1 data variables in *ds* to their RAMS counterparts.

#     Only variables present in both *ds* and *mapping* are renamed; everything
#     else passes through untouched, so it's safe to call on a CM1 dataset that
#     contains variables outside the mapping (or on a partially-renamed dataset).

#     The default mapping (:data:`CM1_TO_RAMS_VARIABLE_NAMES`) covers winds
#     interpolated to scalar points (``uinterp``/``vinterp``/``winterp`` →
#     ``UC``/``VC``/``WC``), potential temperature, water vapor, hydrometeor
#     mixing ratios and number concentrations, TKE, and surface precipitation
#     rate. Variables without a clean correspondence are intentionally omitted.

#     Args:
#         ds: Dataset with CM1 variable names.
#         mapping: Mapping from CM1 → RAMS names. Defaults to
#             :data:`CM1_TO_RAMS_VARIABLE_NAMES`. Pass a custom mapping to
#             override (e.g. for a different microphysics scheme).

#     Returns:
#         Dataset with applicable variables renamed.
#     """
#     mapping = mapping if mapping is not None else CM1_TO_RAMS_VARIABLE_NAMES
#     applicable = {cm1: rams for cm1, rams in mapping.items() if cm1 in ds.data_vars}
#     if not applicable:
#         return ds
#     return ds.rename(applicable)
