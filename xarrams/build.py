"""Build RAMS experiments from templated source code.

The RAMS source files may contain Jinja2 template variables (e.g. {{ my_param }}).
This module copies the source tree, renders all templates with user-supplied values,
compiles the code, and returns the path to the resulting executable.

Rendering uses Jinja2 with StrictUndefined, so any unspecified template variable
will raise an error — the source cannot be built without explicitly providing all
parameter values.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import meta
import re

from carlee_tools import PathLike


def find_template_variables(rams_source: PathLike) -> set[str]:
    """Scan all Fortran source files in a RAMS tree and return the set of Jinja2 template variable names.

    This is useful for discovering what variables need to be specified before building.

    Args:
        rams_source: Path to the RAMS source directory (the one containing src/, bin.rams/, etc.)

    Returns:
        Set of undeclared variable names found across all Fortran source files.
    """
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    variables: set[str] = set()

    for pattern in ("**/*.f90", "**/*.F90"):
        for f90_file in rams_source.glob(pattern):
            source = f90_file.read_text()
            try:
                ast = env.parse(source)
                variables |= meta.find_undeclared_variables(ast)
            except jinja2.TemplateSyntaxError:
                # If a file can't be parsed as a Jinja2 template, skip it —
                # it likely contains syntax that conflicts with Jinja2 (e.g. {{ in
                # a comment) but has no actual template variables.
                pass

    return variables


def build_rams_from_template(
    name: str,
    rams_source: PathLike,
    dest: PathLike,
    template_vars: dict[str, Any],
    make: bool = True,
    make_clean: bool = True,
) -> PathLike | None:
    """Copy RAMS source, render Jinja2 templates, compile, and return the executable path.

    Args:
        name: Experiment name. Appended to the RAMS version number to produce a unique
            executable name (e.g. "6.3.04_my_experiment").
        rams_source: Path to the RAMS source directory (the one containing src/, bin.rams/,
            include.mk, etc.). The Fortran files in this tree may contain Jinja2 template
            variables which will be rendered using ``template_vars``.
        dest: Destination directory for the experiment's copy of the source. Will be created
            if it doesn't exist. If it already exists, it will be overwritten.
        template_vars: Dictionary of template variable names to values. Every Jinja2 variable
            present in any Fortran source file must be provided here; missing variables will
            raise ``jinja2.UndefinedError``.
        make: Whether to compile the code. Defaults to True.
        make_clean: Whether to run ``make clean`` before building. Defaults to True.

    Returns:
        Path to the compiled RAMS executable.

    Raises:
        jinja2.UndefinedError: If any template variable in the source is not specified
            in ``template_vars``.
        subprocess.CalledProcessError: If compilation fails.
        FileNotFoundError: If ``rams_source`` does not exist or is missing expected structure.
    """
    rams_source = Path(rams_source).resolve()
    dest = Path(dest).resolve()

    # Before anything else, throw an error if we don't have all the variables we'll
    # need; jinja will catch this also, but we can give a more informative
    # error message this way
    source_template_variables = find_template_variables(rams_source)
    if set(template_vars.keys()) != source_template_variables:
        raise ValueError(
            "Passed template variable values did not match template variables in"
            f" source code;\nPassed variables: {list(template_vars.keys())}\nVariables"
            f" in source: {source_template_variables}"
        )

    if not rams_source.is_dir():
        raise FileNotFoundError(f"RAMS source directory not found: {rams_source}")

    # --- 1. Copy entire source tree to destination ---
    # Use rsync to only copy changed files so the build runs faster
    # Need to have it do individual files if the destination directory doesn't
    # already exist, in order to get the right structure, but don't want this
    # otherwise because it forces a rebuild of everything

    rsync_cmd_args = [
        "rsync",
        "-aP",
        "--delete",
        str(rams_source) + ("/" if not dest.exists() else ""),
        str(dest),
    ]
    print(" ".join(rsync_cmd_args))
    subprocess.run(rsync_cmd_args)

    # --- 2. Update RAMS_ROOT and RAMS_VERSION in include.mk ---
    include_mk = dest / "include.mk"
    if not include_mk.exists():
        raise FileNotFoundError(f"include.mk not found in {dest}")

    include_mk_text = include_mk.read_text()

    # Parse the existing version number (e.g. "6.3.04" from "RAMS_VERSION=6.3.04_thermals")
    version_line_match = re.search(
        r"^RAMS_VERSION\s*=\s*(.*)$",
        include_mk_text,
        flags=re.MULTILINE,
    )
    if version_line_match is None:
        raise ValueError("Could not find RAMS_VERSION definition in include.mk")

    version_number_match = re.search(
        r"(\d+(?:\.\d+)+)",
        version_line_match.group(1),
    )
    if version_number_match is None:
        raise ValueError(
            "Could not extract numeric version from RAMS_VERSION line: "
            f"{version_line_match.group(0)!r}"
        )

    rams_version = f"{version_number_match.group(1)}_{name}"

    # Apply both substitutions in memory, then write once
    include_mk_text = re.sub(
        r"^RAMS_ROOT\s*=.*$",
        f"RAMS_ROOT={dest}",
        include_mk_text,
        count=1,
        flags=re.MULTILINE,
    )
    include_mk_text = re.sub(
        r"^RAMS_VERSION\s*=.*$",
        f"RAMS_VERSION={rams_version}",
        include_mk_text,
        count=1,
        flags=re.MULTILINE,
    )
    include_mk.write_text(include_mk_text)

    # --- 3. Render Jinja2 templates in all Fortran source files ---
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)

    for pattern in ("**/*.f90", "**/*.F90"):
        for f90_file in dest.glob(pattern):
            source = f90_file.read_text()
            # Only process files that actually contain Jinja2 syntax
            if "{{" not in source and "{%" not in source:
                continue
            template = env.from_string(source)
            rendered = template.render(**template_vars)
            f90_file.write_text(rendered)

    # --- 4. Compile ---
    if make:
        build_dir = dest / "bin.rams"
        if not build_dir.is_dir():
            raise FileNotFoundError(f"bin.rams directory not found in {dest}")

        if make_clean:
            subprocess.run(
                ["make", "clean"],
                cwd=build_dir,
                check=True,
            )

        result = subprocess.run(
            ["make"],
            cwd=build_dir,
            check=True,
            text=True,
        )

        # --- 5. Find and return the executable path ---
        # The Makefile produces an executable named rams-{RAMS_VERSION} in bin.rams/
        # Find it by looking for the symlink named "rams" which points to the real executable
        rams_symlink = build_dir / "rams"
        if rams_symlink.exists():
            return (build_dir / rams_symlink.resolve().name).resolve()

        # Fallback: find any executable matching the rams-* pattern
        executables = list(build_dir.glob("rams-*"))
        executables = [e for e in executables if e.is_file() and not e.suffix]
        if executables:
            return executables[0].resolve()

        raise FileNotFoundError(
            f"Could not find RAMS executable in {build_dir}. "
            f"Build output:\n{result.stdout}\n{result.stderr}"
        )
