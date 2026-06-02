"""RAMS simulation execution and RAMSIN configuration generation.

Provides functions for generating RAMSIN configuration files from templates
and running RAMS simulations (serial or MPI-parallel).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import jinja2
from pandas import Timestamp, Timedelta

from carlee_tools.types_carlee_tools import PathLike

from .utils import dt_to_rams_output_filenames, head_to_data_filename

_RAMS_SUBMIT_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "template_slurm_rams_submission.sh"
)

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


def build_rams_directory_structure(
    base_dir: PathLike, input=True, output=True, derived=True, stdout=True
):
    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    if input:
        (base_dir / "input").mkdir(exist_ok=True, parents=False)
    if output:
        (base_dir / "output").mkdir(exist_ok=True, parents=False)
    if derived:
        (base_dir / "derived").mkdir(exist_ok=True, parents=False)
    if stdout:
        (base_dir / "stdout").mkdir(exist_ok=True, parents=False)

    return base_dir


def generate_ramsin(
    ramsin_name: str,
    rams_run_dir: PathLike,
    ramsin_template_path: PathLike,
    parameters: dict[str, str],
    rams_input_dir: Optional[PathLike] = None,
    rams_output_dir: Optional[PathLike] = None,
):
    rams_run_dir = Path(rams_run_dir)
    ramsin = _generate_ramsin_text(
        run_dir=rams_run_dir,
        ramsin_template_path=ramsin_template_path,
        parameters=parameters,
        rams_input_dir=rams_input_dir,
        rams_output_dir=rams_output_dir,
    )
    ramsin_path = rams_run_dir / f"RAMSIN.{ramsin_name}"
    ramsin_path.write_text(ramsin)
    return ramsin_path


def _generate_ramsin_text(
    run_dir: PathLike,
    ramsin_template_path: PathLike,
    parameters: dict[str, str],
    rams_input_dir: Optional[PathLike] = None,
    rams_output_dir: Optional[PathLike] = None,
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

    run_dir = Path(run_dir)
    # Defaults are written as relative paths so the generated RAMSIN is
    # portable. RAMS resolves these relative to its cwd, which the slurm
    # submission script and run_rams both set to rams_run_dir.
    rams_input_dir = (
        Path(rams_input_dir) if rams_input_dir is not None else Path("input")
    )
    rams_output_dir = (
        Path(rams_output_dir) if rams_output_dir is not None else Path("output")
    )
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
        parameter_regex = r"(^\s*{}\s*\=\s*).*?(\n[^\n\!]*[\=\$])".format(
            parameter_name
        )
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

    return ramsin


def run_rams(
    ramsin_path: PathLike,
    stdout_path: PathLike,
    rams_executable_path: PathLike,
    machsfile_path: Optional[PathLike] = None,
    cwd: Optional[PathLike] = None,
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

    Note:
        For SLURM-scheduled runs, prefer ``dry_run=True`` to only stage
        the RAMSIN/directory structure; the checksum and RAMSIN preamble
        are emitted at job start by the submission script generated by
        ``ps.templates.render_rams_submit`` (so the checksum reflects
        the binary that actually runs, not the one present at submit).

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
    # cwd is what relative paths in the RAMSIN and on the command line resolve
    # against. Defaults to the RAMSIN's parent directory (which is the run
    # dir under the standard layout written by generate_ramsin).
    cwd = Path(cwd).resolve() if cwd is not None else Path(ramsin_path).resolve().parent
    # Resolve relative file paths against cwd for Python-side reads. The
    # command line still passes the original (possibly relative) form so the
    # subprocess sees paths the way they're written in the user's submit.
    ramsin_path_abs = (cwd / ramsin_path).resolve()

    if len(str(ramsin_path_abs)) > 256:
        raise ValueError("RAMS cannot handle ramsin paths longer than 256 characters")

    rams_executable_path = str(Path(rams_executable_path).resolve())

    if not machsfile_path:
        command = RAMS_SERIAL_COMMAND_TEMPLATE.format(
            rams_executable_path=rams_executable_path,
            ramsin_path=str(ramsin_path),
        )
    else:
        with (cwd / machsfile_path).open("r") as f:
            nodelist = f.readlines()
        n_cores = sum(int(s.split(":")[1]) for s in nodelist)
        command = RAMS_MPIEXEC_COMMAND_TEMPLATE.format(
            machsfile_path=str(machsfile_path),
            n_cores=n_cores,
            rams_executable_path=rams_executable_path,
            ramsin_path=str(ramsin_path),
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
            with ramsin_path_abs.open("r") as ramsin_f:
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
                command.split(" "), stdout=stdout_f, start_new_session=True, cwd=cwd
            )
        else:
            return subprocess.run(command.split(" "), stdout=stdout_f, cwd=cwd)


def stdout_dir_for(base_dir: PathLike, ramsin_name: str) -> Path:
    """Return the conventional per-runtype stdout directory.

    Convention: ``{base_dir}/stdout/{ramsin_name}/``. This is the
    directory the SLURM template writes ``<datetime>.stdout`` etc. into.
    """
    return Path(base_dir) / "stdout" / ramsin_name


_TIMEUNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_ramsin_field(ramsin_text: str, name: str) -> Optional[str]:
    """Return the raw RAMSIN value for *name* (sans quotes), or ``None``."""
    match = re.search(
        rf"^\s*{name}\s*=\s*(.*?)\s*,",
        ramsin_text,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    return match.group(1).strip().strip("'\"")


def _require_ramsin_field(ramsin_text: str, name: str) -> str:
    value = _parse_ramsin_field(ramsin_text, name)
    if value is None:
        raise ValueError(f"RAMSIN field {name} not found")
    return value


def _initial_run_end_datetime(ramsin_text: str) -> _dt.datetime:
    """Compute the end datetime of a run from its RAMSIN's start + TIMMAX."""
    iyear = int(_require_ramsin_field(ramsin_text, "IYEAR1"))
    imonth = int(_require_ramsin_field(ramsin_text, "IMONTH1"))
    idate = int(_require_ramsin_field(ramsin_text, "IDATE1"))
    itime = int(_require_ramsin_field(ramsin_text, "ITIME1"))  # HHMM
    timmax = float(_require_ramsin_field(ramsin_text, "TIMMAX"))
    timeunit = (_parse_ramsin_field(ramsin_text, "TIMEUNIT") or "h").lower()
    if timeunit not in _TIMEUNIT_SECONDS:
        raise ValueError(f"Unrecognized TIMEUNIT {timeunit!r}")

    start = _dt.datetime(iyear, imonth, idate, itime // 100, itime % 100)
    return start + _dt.timedelta(seconds=timmax * _TIMEUNIT_SECONDS[timeunit])


# def setup_history_restart(
#     base_dir: PathLike,
#     extension: _dt.timedelta,
#     history_file_head_path: Optional[PathLike] = None,
#     ramsin_template_path: Optional[PathLike] = None,
#     initial_ramsin_name: str = "initial",
#     history_ramsin_name: str = "history",
#     parameters: Optional[dict[str, str]] = None,
#     copy_history_seed: bool = True,
#     seed_dir: Optional[PathLike] = None,
# ) -> Path:
#     """Generate a HISTORY-restart RAMSIN, optionally copying the seed files.

#     Produces ``RAMSIN.{history_ramsin_name}`` alongside the initial RAMSIN
#     in *base_dir*, with ``RUNTYPE='HISTORY'`` and ``HFILIN`` pointed at
#     *history_file_head_path``, and ``TIMMAX`` extended by *extension* past
#     the initial run's end time.

#     When the history run's output directory coincides with the directory
#     containing the seed history file, a running restart could overwrite
#     its own HFILIN. To protect against that, when ``copy_history_seed``
#     is ``True`` (default) and that overlap is detected, the head file and
#     all matching grid ``-g*.h5`` files are copied to *seed_dir* (default
#     ``{base_dir}/restart_seeds/{history_ramsin_name}``) and ``HFILIN`` is
#     pointed at the copy.

#     Args:
#         base_dir: Run directory (the same one passed to
#             :func:`build_rams_directory_structure`).
#         extension: How much further the history run should simulate past
#             the initial run's end. Added to the initial RAMSIN's
#             ``TIMMAX`` (in its ``TIMEUNIT``) to set the new ``TIMMAX``;
#             the start time is unchanged so TIMMAX remains measured from
#             ``IYEAR1``/``IMONTH1``/``IDATE1``/``ITIME1``.
#         history_file_head_path: Path to the seed file's ``...-head.txt``.
#             If ``None``, inferred from the initial run's end datetime
#             (start + initial ``TIMMAX``) under ``{base_dir}/output``.
#         ramsin_template_path: RAMSIN to use as the template. Defaults to
#             ``{base_dir}/RAMSIN.{initial_ramsin_name}``.
#         initial_ramsin_name: Suffix of the initial run's RAMSIN. Used to
#             locate the default template and to compute end time / new
#             TIMMAX.
#         history_ramsin_name: Suffix used for the generated RAMSIN.
#         parameters: Additional RAMSIN overrides merged on top of the
#             ``RUNTYPE``/``HFILIN``/``TIMMAX`` defaults set here.
#         copy_history_seed: If ``True``, copy the seed files when the
#             history run would write to the directory holding them.
#         seed_dir: Destination for the copy when one is made.

#     Returns:
#         Path to the generated ``RAMSIN.{history_ramsin_name}``.
#     """
#     base_dir = Path(base_dir)
#     parameters = dict(parameters) if parameters else {}

#     if ramsin_template_path is None:
#         ramsin_template_path = base_dir / f"RAMSIN.{initial_ramsin_name}"

#     initial_text = (base_dir / f"RAMSIN.{initial_ramsin_name}").read_text()
#     initial_timmax = float(_require_ramsin_field(initial_text, "TIMMAX"))
#     timeunit = (_parse_ramsin_field(initial_text, "TIMEUNIT") or "h").lower()
#     if timeunit not in _TIMEUNIT_SECONDS:
#         raise ValueError(f"Unrecognized TIMEUNIT {timeunit!r}")
#     new_timmax = (
#         initial_timmax + extension.total_seconds() / _TIMEUNIT_SECONDS[timeunit]
#     )

#     if history_file_head_path is None:
#         end_dt = _initial_run_end_datetime(initial_text)
#         initial_output_dir = (base_dir / "output").resolve()
#         datestamp = end_dt.strftime("%Y-%m-%d-%H%M%S")
#         history_file_head_path = initial_output_dir / f"a-A-{datestamp}-head.txt"
#         print(f"Inferred HFILIN as {str(history_file_head_path)}")

#     history_file_head_path = Path(history_file_head_path).resolve()

#     # The history run's output dir is wherever its AFILEPREF resolves to.
#     # generate_ramsin defaults AFILEPREF to "output/a" (relative to cwd =
#     # base_dir), so by default the history output dir is base_dir/output.
#     if "AFILEPREF" in parameters:
#         afilepref = parameters["AFILEPREF"].strip().strip(",").strip().strip("'\"")
#         history_output_dir = (base_dir / afilepref).resolve().parent
#     else:
#         history_output_dir = (base_dir / "output").resolve()

#     seed_overlaps_output = history_file_head_path.parent == history_output_dir

#     if copy_history_seed and seed_overlaps_output:
#         if not history_file_head_path.exists():
#             print("HFILIN does not exist, not creating backup")
#             hfilin = history_file_head_path
#         else:
#             if seed_dir is None:
#                 seed_dir = base_dir / "restart_seeds" / history_ramsin_name
#             seed_dir = Path(seed_dir)
#             seed_dir.mkdir(parents=True, exist_ok=True)

#             basename = history_file_head_path.name[: -len("-head.txt")]
#             copied_head = seed_dir / history_file_head_path.name
#             shutil.copy2(history_file_head_path, copied_head)
#             for h5 in history_file_head_path.parent.glob(f"{basename}-g*.h5"):
#                 shutil.copy2(h5, seed_dir / h5.name)

#             hfilin = copied_head
#     else:
#         hfilin = history_file_head_path

#     parameters.setdefault("RUNTYPE", ramsin_str("HISTORY"))
#     parameters.setdefault("HFILIN", ramsin_str(str(hfilin)))
#     parameters.setdefault("TIMMAX", f"{new_timmax}")
#     for name in ("IAEROHIST", "ITRACHIST"):
#         if name not in parameters:
#             print(
#                 f"Setting {name}=0 by default for the history restart;"
#                 f" pass parameters={{'{name}': ...}} to override."
#             )
#             parameters[name] = "0"

#     return generate_ramsin(
#         ramsin_name=history_ramsin_name,
#         rams_run_dir=base_dir,
#         ramsin_template_path=ramsin_template_path,
#         parameters=parameters,
#     )


def render_rams_submit(
    *,
    run_name: str,
    n_nodes: int,
    n_cores: int,
    wall_time: str,
    rams_executable_path: PathLike,
    ramsin_path: PathLike,
    stdout_dir: PathLike,
    queue: str,
    account: str,
    modules: Sequence[str],
    mpi_launcher: str,
    prologue: Optional[str] = None,
    template_path: PathLike = _RAMS_SUBMIT_TEMPLATE_PATH,
) -> str:
    """Render a SLURM submission script for a RAMS run.

    Pure rendering: all machine-specific values must be supplied by the
    caller. Wrappers (e.g. ``ps.templates.render_rams_submit``) can
    resolve them from a local machine config and forward here.

    Returns the rendered script text; the caller is responsible for
    writing it (or use :func:`write_rams_submit_script`, which handles
    that plus the standard directory conventions).
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(Path(template_path).parent),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    tmpl = env.get_template(Path(template_path).name)
    return tmpl.render(
        account=account,
        modules=list(modules),
        mpi_launcher=mpi_launcher,
        prologue=prologue,
        queue=queue,
        run_name=run_name,
        n_nodes=n_nodes,
        n_cores=n_cores,
        wall_time=wall_time,
        rams_executable_path=str(rams_executable_path),
        ramsin_path=str(ramsin_path),
        stdout_dir=str(stdout_dir),
    )


def write_rams_submit_script(
    run_dir: PathLike,
    ramsin_name: str,
    *,
    render_fn: Callable[..., str] = render_rams_submit,
    run_name: Optional[str] = None,
    **render_kwargs,
) -> Path:
    """Render and write a SLURM submit script under the standard layout.

    Derives all path-related arguments from *run_dir* + *ramsin_name*:

    - ``ramsin_path`` = ``{run_dir}/RAMSIN.{ramsin_name}``
    - ``stdout_dir`` = ``{run_dir}/stdout/{ramsin_name}/`` (auto-created)
    - submit script written to ``{run_dir}/submit_slurm.{ramsin_name}.sh``

    *render_fn* defaults to :func:`render_rams_submit`. Pass a wrapper
    (e.g. ``ps.templates.render_rams_submit``) to inject machine config.
    *run_name* defaults to ``{run_dir.stem}.{ramsin_name}``.
    Any remaining kwargs are forwarded to *render_fn*.

    Returns the path of the written submit script.
    """
    run_dir = Path(run_dir)
    ramsin_path = run_dir / f"RAMSIN.{ramsin_name}"
    stdout_dir = stdout_dir_for(run_dir, ramsin_name)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    submit_path = run_dir / f"submit_slurm.{ramsin_name}.sh"

    rendered = render_fn(
        run_name=run_name if run_name is not None else f"{run_dir.stem}.{ramsin_name}",
        ramsin_path=ramsin_path,
        stdout_dir=stdout_dir,
        **render_kwargs,
    )
    submit_path.write_text(rendered)
    return submit_path


def setup_rams_run(
    run_dir,
    ramsin_parameters,
    ramsin_template_path,
    run_name: str | None = None,
    render_fn: Callable[..., str] = render_rams_submit,
    render_kwargs: dict = {},
    new_return_signature=True,
):
    # Predetermine the paths we'll use
    run_dir = Path(run_dir)
    run_dir.mkdir(exist_ok=True, parents=True)
    name_suffix = "." + run_name if run_name else ""
    ramsin_path = run_dir / f"RAMSIN{name_suffix}"
    stdout_dir = run_dir / "stdout"
    if run_name:
        stdout_dir = stdout_dir / run_name
    submit_script_path = run_dir / f"submit_slurm{name_suffix}.sh"

    # Make the directories
    build_rams_directory_structure(run_dir)
    stdout_dir.mkdir(exist_ok=True, parents=True)

    # Generate the ramsin
    ramsin = _generate_ramsin_text(
        run_dir=run_dir,
        ramsin_template_path=ramsin_template_path,
        parameters=ramsin_parameters,
    )
    ramsin_path.write_text(ramsin)

    # Generate the submit script
    submit_script_path.write_text(
        render_fn(
            run_name=run_name if run_name is not None else run_dir.stem,
            ramsin_path=ramsin_path,
            stdout_dir=stdout_dir,
            **render_kwargs,
        )
    )
    if new_return_signature:
        return ramsin_path, submit_script_path
    else:
        return submit_script_path


def setup_history_restart(
    run_dir,
    extension: _dt.timedelta,
    hr_directory_name="history_restarts",
    hr_name="hr1",
    parent_ramsin_path=None,
    hfilin=None,
    ramsin_parameters: Optional[dict[str, str]] = {},
    grid=1,
    copy_history_seed: bool = True,
    render_fn: Callable[..., str] = render_rams_submit,
    render_kwargs: dict = {},
):

    run_dir = Path(run_dir)

    # Make a history_restarts directory
    hrs_dir = run_dir / hr_directory_name
    hr_dir = hrs_dir / hr_name
    hr_dir.mkdir(exist_ok=True, parents=True)

    # Assume template ramsin is the parent ramsin if not specified
    if not parent_ramsin_path:
        parent_ramsin_path = run_dir / "RAMSIN"
    parent_ramsin_text = parent_ramsin_path.read_text()
    initial_timmax = float(_require_ramsin_field(parent_ramsin_text, "TIMMAX"))
    initial_start_time = Timestamp(
        f"{_require_ramsin_field(parent_ramsin_text, 'IYEAR1')}-{_require_ramsin_field(parent_ramsin_text, 'IMONTH1')}-{_require_ramsin_field(parent_ramsin_text, 'IDATE1')}"
        f" {_require_ramsin_field(parent_ramsin_text, 'ITIME1')}"
    )
    timeunit = (_parse_ramsin_field(parent_ramsin_text, "TIMEUNIT") or "h").lower()
    if timeunit not in _TIMEUNIT_SECONDS:
        raise ValueError(f"Unrecognized TIMEUNIT {timeunit!r}")
    initial_end_time = initial_start_time + Timedelta(initial_timmax, timeunit)

    new_timmax = (
        initial_timmax + extension.total_seconds() / _TIMEUNIT_SECONDS[timeunit]
    )

    # Get the right history restart time by default if not specified
    if hfilin:
        # Get the corresponding data file
        hfilin_data_fname = head_to_data_filename(hfilin)
    else:
        hfilin_data_fname, hfilin_head_fname = dt_to_rams_output_filenames(
            initial_end_time, grid=grid
        )
        hfilin = run_dir / "output" / hfilin_head_fname

    # Copy the starting data so it doesn't get overwritten
    print(hfilin)
    if copy_history_seed:
        if not hfilin.exists():
            print("HFILIN does not exist, not creating backup")
        else:
            # Only do this if the target files don't exist
            seed_dir = hr_dir / "restart_seeds"
            seed_dir.mkdir(parents=False, exist_ok=True)
            hfilin_data_path = run_dir / "output" / hfilin_data_fname
            target_data_path = seed_dir / hfilin_data_path.name
            target_head_path = seed_dir / hfilin.name
            if target_data_path.exists() or target_head_path.exists():
                print("Seed backup already exists, not copying")
            else:
                shutil.copy2(str(hfilin_data_path.resolve()), str(seed_dir.resolve()))
                shutil.copy2(str(hfilin.resolve()), str(seed_dir.resolve()))

    # Generate the file structure and ramsin
    # Don't create input or output directories to avoid confusion, since we're using
    # the parent's
    build_rams_directory_structure(hr_dir, input=False, output=False, derived=False)

    ramsin_parameters = ramsin_parameters | {
        "EXPNME": ramsin_str(
            _require_ramsin_field(parent_ramsin_text, "EXPNME") + "_" + hr_name
        ),
        "TIMMAX": str(new_timmax),
        "RUNTYPE": ramsin_str("HISTORY"),
        "HFILIN": ramsin_str(hfilin.resolve()),
    }
    for name in ("IAEROHIST", "ITRACHIST"):
        if name not in ramsin_parameters:
            print(
                f"Setting {name}=0 by default for the history restart;"
                f" pass ramsin_parameters={{'{name}': ...}} to override."
            )
            ramsin_parameters[name] = "0"
    ramsin_path = generate_ramsin(
        ramsin_name=hr_name,
        rams_run_dir=hr_dir,
        ramsin_template_path=parent_ramsin_path,
        parameters=ramsin_parameters,
        # Override the input and output directories
        rams_input_dir=str((run_dir / "input").resolve()),
        rams_output_dir=str((run_dir / "output").resolve()),
    )

    # Generate the submission script
    submit_script_path = hr_dir / f"submit_slurm.{hr_name}.sh"
    submit_script_path.write_text(
        render_fn(
            run_name=hr_name,
            ramsin_path=ramsin_path,
            stdout_dir=hr_dir / "stdout",
            **render_kwargs,
        )
    )
    return submit_script_path
