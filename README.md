# xarrams

Python utilities for working with [RAMS](https://vandenheever.atmos.colostate.edu/vdhpage/rams.php) (Regional Atmospheric Modeling System) output using xarray.

## Features

- **File I/O** — Read RAMS HDF5 output files into xarray Datasets with automatic dimension naming, coordinate assignment from header files, and optional pint unit attachment.
- **Thermodynamic calculations** — Derive temperature, pressure, humidity, buoyancy, and hydrometeor fields from raw RAMS output using MetPy.
- **Idealized soundings** — Generate Weisman & Klemp (1984) soundings with veering or unidirectional shear, and plot SkewT diagrams.
- **Simulation execution** — Generate RAMSIN configuration files from templates and launch serial or MPI-parallel RAMS runs.
- **Source building** — Build RAMS executables from Jinja2-templated Fortran source trees.
- **Dask integration** — Parallel reading and intermediate result caching for large datasets.

## Installation

```bash
pip install xarrams
```

With optional extras:

```bash
pip install xarrams[dask]    # dask + h5netcdf for parallel reading
pip install xarrams[build]   # jinja2 for templated source building
pip install xarrams[dev]     # development tools (pytest, ruff, mypy, etc.)
```

## Quick start

### Reading RAMS output

```python
import xarrams

# Read one or more RAMS output files
ds = xarrams.read_rams_output(
    ["a-A-2020-01-01-120000-g1.h5", "a-A-2020-01-01-121500-g1.h5"],
    parallel=True,
)
```

### Computing derived variables

```python
# Add thermodynamic variables (T, P, RH, buoyancy, etc.)
ds = xarrams.calculate_thermodynamic_variables(ds)
```

### Generating an idealized sounding

```python
import numpy as np

z_levels = xarrams.get_z_levels(deltaz=100, dzrat=1.1, dzmax=1000, nnzp=50)

sounding = xarrams.wk84_sounding(
    U_s=30,
    q_v0=14,
    shear_layer_depth=6000,
    veering=True,
    z_levels=z_levels,
)
```

### Running RAMS simulations

```python
xarrams.run_rams(
    parameter_sets_dict={"control": {"TIMMAX": "3600."}},
    run_dir="./runs",
    rams_executable_path="/path/to/rams",
    ramsin_template_path="./RAMSIN.template",
)
```

## Package structure

| Module | Description |
|---|---|
| `xarrams.constants` | RAMS constants, variable metadata, unit registry |
| `xarrams.io` | HDF5 file reading, filename parsing, dimension handling |
| `xarrams.calculations` | Thermodynamic and derived variable computations |
| `xarrams.soundings` | Sounding generation, SkewT plotting, RAMSIN formatting |
| `xarrams.execution` | RAMSIN generation and simulation launching |
| `xarrams.utils` | Grid generation, time utilities, log parsing |
| `xarrams.build` | Jinja2-templated RAMS source building |
| `xarrams.dask` | Dask diagnostics and intermediate caching |
| `xarrams.schedulers` | SLURM submission script generation |

## Development

```bash
# Install with pixi (recommended)
pixi run -e dev install
pixi run -e dev test

# Or with pip
pip install -e ".[dev]"
pytest
```

## License

MIT
