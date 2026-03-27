"""File I/O for reading and parsing RAMS model output.

Handles reading RAMS HDF5 output files into xarray Datasets, parsing
filenames for datetime and grid information, and mapping phony HDF5
dimensions to meaningful coordinate names using header files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

from carlee_tools.types_carlee_tools import PathLike
from carlee_tools.utils import dt_to_str, str_to_dt

from .constants import (
    HEADER_NAME_DIMENSION_DICT,
    RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
    RAMS_DT_FORMAT,
    RAMS_FILENAME_DATETIME_REGEX,
    RAMS_VARIABLES_DF,
    ureg,
)


def get_datetime(
    rams_output_filepath: PathLike,
    filename_datetime_regex: str = RAMS_FILENAME_DATETIME_REGEX,
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
        raise ValueError(f"Unable to parse datetime from filepath {rams_output_filepath}")
    return str_to_dt(match.group(0))


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
        raise ValueError(f"Unable to parse grid number from filepath {rams_output_filepath}")
    return int(match.group(1))


def to_rams_output_filename(
    this_dt: "datetime.datetime",
    lite: bool = False,
    grid: int = 1,
) -> str:
    """Build a RAMS output filename for a given datetime, file type, and grid.

    Args:
        this_dt: Datetime of the output file.
        lite: If ``True``, generate a lite-file (``a-L-...``) name; otherwise analysis (``a-A-...``).
        grid: Grid number.

    Returns:
        Filename string (e.g. ``a-A-2020-01-01-120000-g1.h5``).
    """
    return f"a-{'L' if lite else 'A'}-{dt_to_str(this_dt, date_format=RAMS_DT_FORMAT)}-g{grid}.h5"


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
            dimension_vals[header_name_dimension_dict[this_header_name]] = levels
            header_name_dimension_dict.pop(this_header_name)
    return dimension_vals


def infer_rams_dimensions(
    single_time_rams_ds: xr.Dataset,
    grid_number: int = 1,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """Infer the mapping from phony dimensions to real names using the header file.

    Works by matching dimension lengths between the dataset and the header file.
    Requires that no two grid dimensions share the same length.

    Args:
        single_time_rams_ds: A single-timestep RAMS dataset (must have ``encoding["source"]``).
        grid_number: Grid number.

    Returns:
        A tuple of ``(dim_names_mapping, dimension_values)`` where *dim_names_mapping*
        maps dataset dimension names to standard names and *dimension_values* maps
        standard names to coordinate arrays.

    Raises:
        ValueError: If dimensions cannot be uniquely matched by length.
    """
    header_filepath = to_header_filepath(single_time_rams_ds.encoding["source"])
    dimension_vals = get_rams_dimension_values(header_filepath, grid_number=grid_number)

    header_dimension_lengths = {k: len(v) for k, v in dimension_vals.items()}
    if len(header_dimension_lengths.values()) != len(set(header_dimension_lengths.values())):
        raise ValueError(
            "Cannot determine dimension mapping when dimensions have identical lengths."
        )

    ds_dimension_lengths = {
        dim: len(single_time_rams_ds[dim]) for dim in single_time_rams_ds.dims
    }
    dim_names_mapping: dict[str, str] = {}
    for header_dim_name, header_dim_length in header_dimension_lengths.items():
        ds_dims_matching_length = [
            ds_dim
            for ds_dim, ds_dim_length in ds_dimension_lengths.items()
            if ds_dim_length == header_dim_length
        ]
        if len(ds_dims_matching_length) > 1:
            raise ValueError(
                "Multiple dimensions of same length in dataset; cannot infer dimension names and values"
            )
        if len(ds_dims_matching_length) < 1:
            raise ValueError(
                f"No dimensions of length {header_dim_length} found in dataset; this shouldn't happen"
            )
        dim_names_mapping[ds_dims_matching_length[0]] = header_dim_name

    assert len(dim_names_mapping) == len(HEADER_NAME_DIMENSION_DICT)
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
        ds = ds.rename_dims(dimension_names_mapping).assign_coords(dimension_values)
    except ValueError:
        print(
            "Mismatch between dimension lengths in dataset and header;\n"
            f"Passed dimension dict: {dimension_names_mapping}\n"
            f"Dimension sizes in dataset: {ds.dims}\n"
            f"Dimension lengths from header: { {k: len(v) for k, v in dimension_values.items()} }"
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
    filename_datetime_regex: str = RAMS_FILENAME_DATETIME_REGEX,
    units: bool = False,
) -> Union[xr.Dataset, list[xr.Dataset]]:
    """Read one or more RAMS HDF5 output files into an xarray Dataset.

    Handles dimension renaming, coordinate assignment from header files,
    time coordinate construction from filenames, and optional unit attachment.

    Args:
        input_filenames: Paths to RAMS ``.h5`` output files.
        fill_dim_names: Whether to rename phony dimensions to real names.
        dim_names: Explicit dimension name mapping. If ``None``, inferred
            automatically for analysis files or from the header for lite files.
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

    lite = any(Path(x).name.startswith("a-L") for x in input_filenames)
    if not lite and not dim_names:
        dim_names = RAMS_ANALYSIS_FILE_DIMENSIONS_DICT

    if drop_vars and keep_vars:
        raise ValueError("Cannot pass both drop_vars and keep_vars")

    lite = Path(list(input_filenames)[0]).name.startswith("a-L")
    if not dim_names and not lite:
        dim_names = RAMS_ANALYSIS_FILE_DIMENSIONS_DICT

    def maybe_print(x: str) -> None:
        if not silent:
            print(x)

    if parallel:
        try:
            import dask  # noqa: F401
        except ImportError:
            print("dask must be installed to use the `parallel` option; falling back to serial")
            parallel = False

    input_filenames = [Path(x) for x in input_filenames]
    input_datetimes = []
    for fpath in input_filenames:
        time = get_datetime(fpath, filename_datetime_regex=filename_datetime_regex)
        if not time:
            raise ValueError(
                f"File {fpath.name} does not contain a valid timestamp in the filename"
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
            f"Reading and concatenating {len(input_filenames)} individual timestep outputs..."
        )
        from contextlib import nullcontext

        from dask.diagnostics import ProgressBar

        open_ds_context_manager = nullcontext if silent else ProgressBar
        with open_ds_context_manager():
            ds = xr.open_mfdataset(
                input_filenames,
                concat_dim=time_dim_name,
                combine="nested",
                preprocess=_sanitized_preprocess,
                phony_dims="sort",
                engine="h5netcdf",
                drop_variables=drop_vars,
                parallel=True,
                chunks=chunks,
                **open_dataset_kwargs,
            )
    else:
        maybe_print(f"Reading {len(input_filenames)} individual timestep outputs...")
        to_concat: list[xr.Dataset] = []
        wrapped_to_read = tqdm(input_filenames) if not silent else input_filenames
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

    ds = ds.assign_coords(**{time_dim_name: input_datetimes})
    ds = ds.sortby(time_dim_name)

    if parallel:
        ds = ds.unify_chunks()

    if units:
        ds = ds.pint.quantify(
            RAMS_VARIABLES_DF.set_index("name")["units"].to_dict(), unit_registry=ureg
        )

    rams_attrs_dicts = RAMS_VARIABLES_DF.set_index("name").to_dict(orient="index")
    for var in ds.data_vars:
        ds[var] = ds[var].assign_attrs(rams_attrs_dicts.get(var, {}))

    return ds
