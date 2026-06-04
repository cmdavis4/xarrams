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
    CM1_TO_RAMS_VARIABLE_NAMES,
    DEFAULT_BSR_VARIABLES,
    HYDROMETEOR_SPECIES_FULL_NAMES,
    RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
    RAMS_FILENAME_DT_STRFTIME_FORMAT,
    RAMS_FILENAME_DT_REGEX,
    RAMS_VARIABLES_DF,
    SOUNDING_NAMELIST_VARIABLES,
    ureg,
)

# --- File I/O ----------------------------------------------------------------
from .io import (
    fill_rams_output_dimensions,
    get_datetime_from_rams_filename,
    get_grid_number,
    get_rams_dimension_values,
    infer_rams_dimensions,
    read_cm1_output,
    read_rams_output,
    to_header_filepath,
    to_rams_output_filename,
)

# --- RAMS execution ----------------------------------------------------------
from .execution import (
    generate_ramsin,
    ramsin_str,
    run_rams,
    build_rams_directory_structure,
    setup_history_restart,
    stdout_dir_for,
    render_rams_submit,
    render_cm1_submit,
    write_rams_submit_script,
)

# --- CM1 ---------------------------------------------------------------------
from . import cm1

# --- Build ---------------------------------------------------------------------
from . import build

# --- Calculations ------------------------------------------------------------
from .calculations import (
    calculate_bsr_variables,
    calculate_derived_variables,
    calculate_thermodynamic_variables,
)

# --- Soundings ---------------------------------------------------------------
from .soundings import (
    format_sounding_field_ramsin_str,
    plot_sounding_skewt,
    with_updated_sounding_fields,
    wk84_sounding,
    write_rams_formatted_sounding,
    to_sounding_df,
    calculate_sounding_derived_vars,
    plot_sounding_diagnostics,
    plot_base_state_diagnostics,
)

# --- Utilities ---------------------------------------------------------------
from .utils import (
    get_z_levels,
    parse_rams_stdout_walltimes,
    to_t_minutes,
    with_t_minutes_coord,
    check_rams_run_statuses,
)

# --- Dask integration --------------------------------------------------------
from .dask import dask_diagnostics, reload_intermediate

__all__ = [
    # Constants
    "CM1_TO_RAMS_VARIABLE_NAMES",
    "DEFAULT_BSR_VARIABLES",
    "HYDROMETEOR_SPECIES_FULL_NAMES",
    "RAMS_ANALYSIS_FILE_DIMENSIONS_DICT",
    "RAMS_FILENAME_DT_STRFTIME_FORMAT",
    "RAMS_FILENAME_DT_REGEX",
    "RAMS_VARIABLES_DF",
    "SOUNDING_NAMELIST_VARIABLES",
    "ureg",
    # File I/O
    "fill_rams_output_dimensions",
    "get_datetime_from_rams_filename",
    "get_grid_number",
    "get_rams_dimension_values",
    "infer_rams_dimensions",
    "read_cm1_output",
    "read_rams_output",
    "to_header_filepath",
    "to_rams_output_filename",
    # Execution
    "generate_ramsin",
    "ramsin_str",
    "run_rams",
    "setup_history_restart",
    "stdout_dir_for",
    "render_rams_submit",
    "render_cm1_submit",
    "write_rams_submit_script",
    "build_rams_directory_structure",
    # Calculations
    "calculate_bsr_variables",
    "calculate_derived_variables",
    "calculate_thermodynamic_variables",
    # Soundings
    "format_sounding_field_ramsin_str",
    "plot_sounding_skewt",
    "with_updated_sounding_fields",
    "wk84_sounding",
    "write_rams_formatted_sounding",
    "to_sounding_df",
    "calculate_sounding_derived_vars",
    "plot_sounding_diagnostics",
    "plot_base_state_diagnostics",
    # Utilities
    "get_z_levels",
    "parse_rams_stdout_walltimes",
    "to_t_minutes",
    "with_t_minutes_coord",
    "check_rams_run_statuses",
    # Dask
    "dask_diagnostics",
    "reload_intermediate",
    # Submodules
    "cm1",
    "build",
]
