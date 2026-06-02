"""CM1 simulation execution and namelist configuration generation.

Provides functions for generating CM1 ``namelist.input`` files from
templates, analogous to the RAMSIN handling in :mod:`xarrams.execution`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

from carlee_tools.types_carlee_tools import PathLike

from .constants import CM1_TO_RAMS_DIM_MAPPINGS, CM1_TO_RAMS_VARIABLE_NAMES
from .io import CM1_DEFAULT_START_DATETIME


def generate_cm1_namelist(
    output_dir: PathLike,
    namelist_template_path: PathLike,
    parameters: dict[str, str],
) -> Path:
    """Generate a CM1 ``namelist.input`` file from a template.

    Reads *namelist_template_path*, replaces the values of the specified
    *parameters* in-place, and writes the result to
    ``{cm1_run_dir}/namelist.input.{namelist_name}`` (CM1 itself requires
    the file to be named ``namelist.input`` exactly at runtime, so the
    caller is responsible for renaming or symlinking to that name).

    Each replacement targets a single line of the form
    ``<name> = <value>[,] [! comment]``, preserving any trailing comma and
    inline comment on the line.

    Args:
        namelist_name: Name used in the output filename
            (``namelist.input.{namelist_name}``).
        cm1_run_dir: Directory where the generated namelist is written.
        namelist_template_path: Path to the template ``namelist.input`` file.
        parameters: Parameter names mapped to their replacement values.
            Values are written verbatim — include quotes for strings
            (e.g. ``"'string'"``) and Fortran logicals as ``".true."`` or
            ``".false."``.

    Returns:
        Path to the generated namelist file.

    Raises:
        ValueError: If a parameter name is not found in the template.
    """
    output_dir = Path(output_dir)
    namelist_template_path = Path(namelist_template_path)

    namelist = namelist_template_path.read_text()

    for parameter_name, parameter_value in parameters.items():
        parameter_regex = r"(^\s*{}\s*=\s*)[^!,\n]*?(\s*)(?=,|!|$)".format(
            re.escape(parameter_name)
        )
        namelist, n_subs = re.subn(
            parameter_regex,
            lambda m, v=parameter_value: f"{m.group(1)}{v}{m.group(2)}",
            namelist,
            count=1,
            flags=re.MULTILINE,
        )
        if n_subs == 0:
            raise ValueError(
                f"Field {parameter_name} not found in template namelist.input"
            )

    output_dir = output_dir / "namelist.input"
    output_dir.write_text(namelist)
    return output_dir


def _calculate_ramslike_derived_variables(cm1_ds):
    cm1_ds = cm1_ds.copy()
    cm1_ds["RTP"] = (
        cm1_ds["RV"]
        + cm1_ds["RCP"]
        + cm1_ds["RRP"]
        + cm1_ds["RPP"]
        + cm1_ds["RSP"]
        + cm1_ds["RGP"]
    )
    return cm1_ds


def coerce_to_ramslike(
    cm1_ds, start_datetime=CM1_DEFAULT_START_DATETIME, time_dim_name="time"
):
    # First limit to the variables we can map directly, and that are present
    present_coercables = [
        x for x in CM1_TO_RAMS_VARIABLE_NAMES.keys() if x in cm1_ds.data_vars
    ]
    cm1_ds = cm1_ds[present_coercables]
    # Rename dimensions
    cm1_ds = cm1_ds.rename(CM1_TO_RAMS_DIM_MAPPINGS)
    # Rename data variables
    cm1_ds = cm1_ds.rename(
        {k: v for k, v in CM1_TO_RAMS_VARIABLE_NAMES.items() if k in present_coercables}
    )
    # Calculate other base RAMS variables that we can get directly from these
    cm1_ds = _calculate_ramslike_derived_variables(cm1_ds)
    # Fix the time coordinate
    start_ts = pd.Timestamp(start_datetime)
    seconds = np.asarray(cm1_ds[time_dim_name].values, dtype="float64")
    absolute_times = start_ts + pd.to_timedelta(seconds, unit="s")
    cm1_ds = cm1_ds.assign_coords({time_dim_name: absolute_times})

    return cm1_ds
