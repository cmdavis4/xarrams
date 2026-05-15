"""CM1 simulation execution and namelist configuration generation.

Provides functions for generating CM1 ``namelist.input`` files from
templates, analogous to the RAMSIN handling in :mod:`xarrams.execution`.
"""

from __future__ import annotations

import re
from pathlib import Path

from carlee_tools.types_carlee_tools import PathLike


def generate_cm1_namelist(
    namelist_name: str,
    cm1_run_dir: PathLike,
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
    cm1_run_dir = Path(cm1_run_dir)
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

    namelist_path = cm1_run_dir / f"namelist.input.{namelist_name}"
    namelist_path.write_text(namelist)
    return namelist_path
