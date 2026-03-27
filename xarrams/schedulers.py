from pathlib import Path

import jinja2

from carlee_tools import PathLike


def generate_slurm_submission_script(
    run_dir: PathLike,
    rams_executable_path: PathLike,
    project_code: str,
    user_email: str,
    memory: str = "0",
    template_path=Path(__file__).parent
    / "templates"
    / "cumulus_rams_slurm_submission_template.sh",
    output_filename="submit_slurm.sh",
    walltime_str: str = "06:00:00",
    n_nodes: int = 1,
    queue: str = "batch_short",
):
    # Convert run_dir to path
    run_dir = Path(run_dir)
    run_name = run_dir.stem

    template_loader = jinja2.FileSystemLoader(searchpath="/")
    template_env = jinja2.Environment(
        loader=template_loader, undefined=jinja2.StrictUndefined
    )

    pbs_submission_template = template_env.get_template(template_path)

    # Number of cores is 128*number of nodes
    n_cores = 128 * n_nodes

    # Make a stdout dir if it doesn't exist
    stdout_dir = run_dir.joinpath("stdout")
    stdout_dir.mkdir(exist_ok=True)

    ramsin_path = Path(run_dir).joinpath(f"RAMSIN.{run_name}")

    rendered_submission_script = pbs_submission_template.render({
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
    with (run_dir / output_filename).open("w") as f:
        f.write(rendered_submission_script)
