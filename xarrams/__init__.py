"""
xarrams: Utilities for working with RAMS atmospheric model output.

Provides functions for reading, processing, and analyzing Regional Atmospheric
Modeling System (RAMS) output files.
"""

__version__ = "0.1.0"

from .core import (
    # File I/O
    read_rams_output,
    get_datetime,
    get_grid_number,
    to_header_filepath,
    to_rams_output_filename,

    # RAMS execution
    run_rams,
    run_rams_for_ramsin,
    generate_ramsin,

    # Data processing
    calculate_thermodynamic_variables,
    calculate_derived_variables,
    calculate_bsr_variables,
    fill_rams_output_dimensions,
    infer_rams_dimensions,
    get_rams_dimension_values,

    # Soundings
    plot_sounding,
    wk84_sounding,
    write_rams_formatted_sounding,

    # Utilities
    get_z_levels,
    ramsin_str,
    with_t_minutes_coord,
    to_t_minutes,
    reload_intermediate,

    # Constants
    RAMS_DT_FORMAT,
    RAMS_DT_STRFTIME_STR,
    RAMS_FILENAME_DATETIME_REGEX,
    RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
    RAMS_VARIABLES_DF,
    HYDROMETEOR_SPECIES_FULL_NAMES,
    SOUNDING_NAMELIST_VARIABLES,
    DEFAULT_BSR_VARIABLES,
    ureg,
)

__all__ = [
    # Functions
    "read_rams_output",
    "get_datetime",
    "get_grid_number",
    "calculate_thermodynamic_variables",
    "calculate_derived_variables",
    "calculate_bsr_variables",
    "plot_sounding",
    "wk84_sounding",
    "run_rams",
    "generate_ramsin",
    "to_rams_output_filename",
    "to_header_filepath",
    "fill_rams_output_dimensions",
    "get_z_levels",
    "with_t_minutes_coord",
    # Constants
    "RAMS_DT_FORMAT",
    "RAMS_DT_STRFTIME_STR",
    "RAMS_VARIABLES_DF",
    "ureg",
]
