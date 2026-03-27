"""RAMS simulation execution and RAMSIN configuration generation.

Provides functions for generating RAMSIN configuration files from templates
and running RAMS simulations (serial or MPI-parallel).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

from carlee_tools.types_carlee_tools import PathLike

# ---------------------------------------------------------------------------
# Command templates
# ---------------------------------------------------------------------------

RAMS_SERIAL_COMMAND_TEMPLATE: str = "{rams_executable_path} -f {ramsin_path}"
"""Shell command template for running RAMS in serial mode."""

RAMS_MPIEXEC_COMMAND_TEMPLATE: str = (
    "mpiexec -machinefile {machsfile_path} -np {n_cores}"
    " {rams_executable_path} -f {ramsin_path}"
)
"""Shell command template for running RAMS via MPI.

The ``mpiexec`` binary is expected to be on ``$PATH``.  Override this
module-level constant if a specific path is required.
"""


def ramsin_str(s: object) -> str:
    """Wrap a value in single quotes for use in a RAMSIN namelist.

    Args:
        s: Value to wrap.

    Returns:
        The value as a single-quoted string (e.g. ``"'value'"``).
    """
    return f"'{s}'"


def generate_ramsin(
    ramsin_name: str,
    parameters: dict[str, str],
    rams_input_dir: Optional[PathLike],
    rams_output_dir: Optional[PathLike],
    ramsin_dir: PathLike,
    ramsin_template_path: PathLike,
) -> str:
    """Generate a RAMSIN configuration file from a template.

    Reads *ramsin_template_path*, replaces the values of the specified
    *parameters*, sets I/O directory prefixes, and writes the result to
    ``{ramsin_dir}/RAMSIN.{ramsin_name}``.

    Args:
        ramsin_name: Name used in the output filename (``RAMSIN.{ramsin_name}``).
        parameters: Parameter names mapped to their replacement values.
            Values are written verbatim — include quotes where needed.
        rams_input_dir: Base directory for RAMS input files.  Sets
            ``TOPFILES``, ``SFCFILES``, ``SSTFPFX``, and ``NDVIFPFX``
            unless overridden in *parameters*.
        rams_output_dir: Base directory for RAMS output files.  Sets
            ``AFILEPREF`` unless overridden in *parameters*.
        ramsin_dir: Directory where the generated RAMSIN is written.
        ramsin_template_path: Path to the template RAMSIN file.

    Returns:
        The full text of the generated RAMSIN.

    Raises:
        ValueError: If a parameter name is not found in the template.
    """
    parameters = dict(parameters)

    rams_input_dir = Path(rams_input_dir) if rams_input_dir is not None else None
    rams_output_dir = Path(rams_output_dir) if rams_output_dir is not None else None
    ramsin_dir = Path(ramsin_dir)
    ramsin_template_path = Path(ramsin_template_path)

    ramsin = ramsin_template_path.read_text()

    input_dir_sub_suffixes = {
        "TOPFILES": "toph",
        "SFCFILES": "sfch",
        "SSTFPFX": "ssth",
        "NDVIFPFX": "ndh",
    }
    output_dir_sub_suffixes = {"AFILEPREF": "a"}

    for param_name, suffix in input_dir_sub_suffixes.items():
        if param_name not in parameters and rams_input_dir is not None:
            parameters[param_name] = f"'{rams_input_dir / suffix}'"
    for param_name, suffix in output_dir_sub_suffixes.items():
        if param_name not in parameters and rams_output_dir is not None:
            parameters[param_name] = f"'{rams_output_dir / suffix}'"

    for parameter_name, parameter_value in parameters.items():
        parameter_regex = r"(^\s*{}\s*\=\s*).*?(\n[^\n\!]*[\=\$])".format(parameter_name)
        replacement_regex = r"\g<1>{},\g<2>".format(parameter_value)
        ramsin, n_subs = re.subn(
            parameter_regex,
            replacement_regex,
            ramsin,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
        )
        if n_subs == 0:
            raise ValueError(f"Field {parameter_name} not found in template RAMSIN")

    (ramsin_dir / f"RAMSIN.{ramsin_name}").write_text(ramsin)
    return ramsin


def run_rams(
    ramsin_path: PathLike,
    stdout_path: PathLike,
    rams_executable_path: PathLike,
    machsfile_path: Optional[PathLike] = None,
    log_command: bool = True,
    log_ramsin: bool = True,
    dry_run: bool = False,
    asynchronous: bool = True,
    verbose: bool = True,
) -> Union[bool, subprocess.Popen, subprocess.CompletedProcess]:  # type: ignore[type-arg]
    """Run a RAMS simulation.

    Launches the RAMS executable for a single RAMSIN configuration,
    either in serial or MPI-parallel mode.  Optionally logs the command
    and RAMSIN contents to the stdout capture file.

    Args:
        ramsin_path: Path to the RAMSIN configuration file.
        stdout_path: File where RAMS stdout is captured.
        rams_executable_path: Path to the RAMS executable.
        machsfile_path: Machine file for MPI parallel execution.
            If ``None``, RAMS is run in serial.
        log_command: Prepend the executed command (and executable checksum)
            to the stdout file.
        log_ramsin: Prepend the RAMSIN contents to the stdout file.
        dry_run: If ``True``, skip execution and return ``True``.
        asynchronous: If ``True``, return a :class:`subprocess.Popen` handle
            immediately; otherwise block until completion.
        verbose: Print the command before execution.

    Returns:
        ``True`` for dry runs, a :class:`subprocess.Popen` for async runs,
        or a :class:`subprocess.CompletedProcess` for synchronous runs.

    Raises:
        ValueError: If the resolved RAMSIN path exceeds 256 characters
            (a RAMS limitation).
    """
    if len(str(Path(ramsin_path).resolve())) > 256:
        raise ValueError("RAMS cannot handle ramsin paths longer than 256 characters")

    rams_executable_path = str(Path(rams_executable_path).resolve())

    if not machsfile_path:
        command = RAMS_SERIAL_COMMAND_TEMPLATE.format(
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )
    else:
        with Path(machsfile_path).open("r") as f:
            nodelist = f.readlines()
        n_cores = sum(int(s.split(":")[1]) for s in nodelist)
        command = RAMS_MPIEXEC_COMMAND_TEMPLATE.format(
            machsfile_path=str(Path(machsfile_path).resolve()),
            n_cores=n_cores,
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )

    write_mode = "w"

    if log_command:
        with Path(rams_executable_path).open("rb") as rams_exe_f:
            rams_checksum = hashlib.md5(rams_exe_f.read()).hexdigest()
        with Path(stdout_path).open(write_mode) as stdout_f:
            hashes = "#" * 47
            stdout_f.write(f"{hashes}\nRAMS CHECKSUM: {rams_checksum}\n{hashes}\n\n")
            stdout_f.write(
                "##############################\n         BEGIN"
                " COMMAND\n##############################\n"
            )
            stdout_f.write(f"{command} > {stdout_path}")
            stdout_f.write(
                "\n##############################\n         END"
                " COMMAND\n##############################\n\n"
            )
        write_mode = "a"

    if log_ramsin:
        with Path(stdout_path).open(write_mode) as stdout_f:
            with Path(ramsin_path).open("r") as ramsin_f:
                stdout_f.write(
                    "##############################\n         BEGIN"
                    " RAMSIN\n##############################\n"
                )
                stdout_f.write(ramsin_f.read())
                stdout_f.write(
                    "\n##############################\n         END"
                    " RAMSIN\n##############################\n\n"
                )
        write_mode = "a"

    if verbose:
        print(f"{command} > {stdout_path}")

    if dry_run:
        return True

    with Path(stdout_path).open(write_mode) as stdout_f:
        if asynchronous:
            return subprocess.Popen(
                command.split(" "), stdout=stdout_f, start_new_session=True
            )
        else:
            return subprocess.run(command.split(" "), stdout=stdout_f)
