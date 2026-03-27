"""SLURM submission script generation for RAMS simulations.

Renders a Jinja2 template to produce a ready-to-submit SLURM batch script.
"""

from __future__ import annotations

from pathlib import Path

import jinja2

from carlee_tools.types_carlee_tools import PathLike

_DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "cumulus_rams_slurm_submission_template.sh"


def generate_slurm_submission_script(
    run_dir: PathLike,
    rams_executable_path: PathLike,
    project_code: str,
    user_email: str,
    memory: str = "0",
    template_path: PathLike = _DEFAULT_TEMPLATE,
    output_filename: str = "submit_slurm.sh",
    walltime_str: str = "06:00:00",
    n_nodes: int = 1,
    queue: str = "batch_short",
) -> None:
    """Generate a SLURM submission script for a RAMS simulation.

    Args:
        run_dir: Directory for this simulation run. The script and a
            ``stdout/`` subdirectory are created here.
        rams_executable_path: Path to the RAMS executable.
        project_code: HPC project/allocation code.
        user_email: Email address for job notifications.
        memory: Memory request string (``"0"`` for default).
        template_path: Path to the Jinja2 SLURM template.
        output_filename: Name of the generated script file.
        walltime_str: Walltime in ``HH:MM:SS`` format.
        n_nodes: Number of compute nodes.
        queue: SLURM partition/queue name.
    """
    run_dir = Path(run_dir)
    run_name = run_dir.stem

    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath="/"),
        undefined=jinja2.StrictUndefined,
    )
    template = template_env.get_template(str(template_path))

    n_cores = 128 * n_nodes

    stdout_dir = run_dir / "stdout"
    stdout_dir.mkdir(exist_ok=True)

    ramsin_path = run_dir / f"RAMSIN.{run_name}"

    rendered = template.render({
        "queue": queue,
        "user_email": user_email,
        "project_code": project_code,
        "memory": memory,
        "n_nodes": n_nodes,
        "n_cores": n_cores,
        "run_name": run_name,
        "wall_time": walltime_str,
        "stdout_dir": str(stdout_dir),
        "rams_executable_path": rams_executable_path,
        "ramsin_path": ramsin_path,
    })

    (run_dir / output_filename).write_text(rendered)
