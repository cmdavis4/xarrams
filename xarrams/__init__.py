"""xarrams: Utilities for working with RAMS atmospheric model output.

Provides functions for reading, processing, and analyzing Regional Atmospheric
Modeling System (RAMS) output files.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("xarrams")
except PackageNotFoundError:
    __version__ = "1.0.0"  # fallback for editable / non-installed usage

# --- Constants ---------------------------------------------------------------
from .constants import (
    DEFAULT_BSR_VARIABLES,
    HYDROMETEOR_SPECIES_FULL_NAMES,
    RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
    RAMS_DT_FORMAT,
    RAMS_DT_STRFTIME_STR,
    RAMS_FILENAME_DATETIME_REGEX,
    RAMS_VARIABLES_DF,
    SOUNDING_NAMELIST_VARIABLES,
    ureg,
)

# --- File I/O ----------------------------------------------------------------
from .io import (
    fill_rams_output_dimensions,
    get_datetime,
    get_grid_number,
    get_rams_dimension_values,
    infer_rams_dimensions,
    read_rams_output,
    to_header_filepath,
    to_rams_output_filename,
)

# --- RAMS execution ----------------------------------------------------------
from .execution import (
    generate_ramsin,
    ramsin_str,
    run_rams,
)

# --- Calculations ------------------------------------------------------------
from .calculations import (
    calculate_bsr_variables,
    calculate_derived_variables,
    calculate_thermodynamic_variables,
)

# --- Soundings ---------------------------------------------------------------
from .soundings import (
    format_sounding_field_ramsin_str,
    plot_sounding,
    with_updated_sounding_fields,
    wk84_sounding,
    write_rams_formatted_sounding,
)

# --- Utilities ---------------------------------------------------------------
from .utils import (
    get_z_levels,
    parse_rams_stdout_walltimes,
    to_t_minutes,
    with_t_minutes_coord,
)

# --- Dask integration --------------------------------------------------------
from .dask import dask_diagnostics, reload_intermediate

__all__ = [
    # Constants
    "DEFAULT_BSR_VARIABLES",
    "HYDROMETEOR_SPECIES_FULL_NAMES",
    "RAMS_ANALYSIS_FILE_DIMENSIONS_DICT",
    "RAMS_DT_FORMAT",
    "RAMS_DT_STRFTIME_STR",
    "RAMS_FILENAME_DATETIME_REGEX",
    "RAMS_VARIABLES_DF",
    "SOUNDING_NAMELIST_VARIABLES",
    "ureg",
    # File I/O
    "fill_rams_output_dimensions",
    "get_datetime",
    "get_grid_number",
    "get_rams_dimension_values",
    "infer_rams_dimensions",
    "read_rams_output",
    "to_header_filepath",
    "to_rams_output_filename",
    # Execution
    "generate_ramsin",
    "ramsin_str",
    "run_rams",
    # Calculations
    "calculate_bsr_variables",
    "calculate_derived_variables",
    "calculate_thermodynamic_variables",
    # Soundings
    "format_sounding_field_ramsin_str",
    "plot_sounding",
    "with_updated_sounding_fields",
    "wk84_sounding",
    "write_rams_formatted_sounding",
    # Utilities
    "get_z_levels",
    "parse_rams_stdout_walltimes",
    "to_t_minutes",
    "with_t_minutes_coord",
    # Dask
    "dask_diagnostics",
    "reload_intermediate",
]
