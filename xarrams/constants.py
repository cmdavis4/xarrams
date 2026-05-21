"""Constants, data loading, and unit configuration for xarrams.

This module centralizes all RAMS-related constants, regex patterns,
variable metadata, and the pint unit registry used throughout the package.
"""

from pathlib import Path

import pandas as pd
from pint import UnitRegistry
import pint_xarray

from metpy.units import units

# ---------------------------------------------------------------------------
# Datetime format
# ---------------------------------------------------------------------------

RAMS_FILENAME_DT_STRFTIME_FORMAT: str = r"%Y-%m-%d-%H%M%S"
"""strftime/strptime format string for RAMS filenames (e.g. ``2020-01-01-120000``)."""

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

RAMS_FILENAME_DT_REGEX: str = r"[0-9]{4}\-[0-9]{2}\-[0-9]{2}\-[0-9]{6}"
"""Regex that matches the datetime portion of a RAMS output filename."""

# ---------------------------------------------------------------------------
# Dimension mappings
# ---------------------------------------------------------------------------

RAMS_ANALYSIS_FILE_DIMENSIONS_DICT: dict[str, str] = {
    "phony_dim_0": "y",
    "phony_dim_1": "x",
    "phony_dim_2": "z",
    "phony_dim_3": "p",
    "phony_dim_4": "kppz",
}
"""Mapping from HDF5 phony dimension names to real names for analysis files."""

HEADER_NAME_DIMENSION_DICT: dict[str, str] = {
    "__ztn{grid_number}": "z",
    "__ytn{grid_number}": "y",
    "__xtn{grid_number}": "x",
}
"""Mapping from header-file dimension names (with grid number placeholder) to standard names."""

# ---------------------------------------------------------------------------
# Variable lists
# ---------------------------------------------------------------------------

DEFAULT_BSR_VARIABLES: list[str] = ["THETA", "UC", "VC", "THETA_v", "THETA_rho", "P"]
"""Default variables for which to compute base-state-relative perturbations."""

SOUNDING_NAMELIST_VARIABLES: list[str] = ["PS", "TS", "RTS", "US", "VS"]
"""Variables that make up an initial sounding in a RAMSIN namelist."""

HYDROMETEOR_SPECIES_FULL_NAMES: dict[str, str] = {
    "PP": "pristine ice",
    "SP": "snow",
    "AP": "aggregates",
    "HP": "hail",
    "GP": "graupel",
    "CP": "cloud",
    "DP": "drizzle",
    "RP": "rain",
}
"""Human-readable names for each RAMS hydrometeor species abbreviation."""

# ---------------------------------------------------------------------------
# RAMS variable metadata (loaded from CSV)
# ---------------------------------------------------------------------------

RAMS_VARIABLES_DF: pd.DataFrame = pd.read_csv(
    Path(__file__).parent / "data" / "rams_variables.csv"
)
RAMS_VARIABLES_DF["units"] = RAMS_VARIABLES_DF["units"].str.replace("#", "1")
RAMS_VARIABLES_DF = RAMS_VARIABLES_DF.drop(RAMS_VARIABLES_DF.columns[0], axis=1)

# ---------------------------------------------------------------------------
# Pint unit registry
# ---------------------------------------------------------------------------

ureg: UnitRegistry = UnitRegistry()
"""Pint UnitRegistry with custom RAMS unit definitions loaded."""

ureg.load_definitions(str(Path(__file__).parent / "data" / "rams_pint_units.txt"))
pint_xarray.setup_registry(ureg)


# Physical constants with values used by RAMS
# ================================

C_p = 1004.0 * units("J/kg/K")  # J/(kg·K)
R_d = 287.0 * units("J/kg/K")  # J/(kg·K)
p0 = 1000.0 * units("hPa")  # hPa
reps = 461.5 / 287.0  # = R_v / R_d, ≈ 1.608

