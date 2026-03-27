"""Thermodynamic and derived variable calculations for RAMS output.

Provides functions to compute derived meteorological quantities
(temperature, pressure, humidity, buoyancy, etc.) from raw RAMS
model output fields using MetPy.
"""

from __future__ import annotations

import xarray as xr
import metpy.calc as mpc
import metpy.constants as mpconstants
from metpy.units import units

from .constants import DEFAULT_BSR_VARIABLES


def calculate_thermodynamic_variables(
    ds: xr.Dataset,
    fail_if_missing_vars: bool = False,
) -> xr.Dataset:
    """Compute derived thermodynamic variables from basic RAMS fields.

    Calculates as many derived variables as possible from the fields
    present in *ds*.  Each derivation is skipped silently when its
    required inputs are absent (unless *fail_if_missing_vars* is set).

    Args:
        ds: Dataset containing raw RAMS output variables.
        fail_if_missing_vars: Raise instead of skipping when required
            variables are missing.

    Returns:
        The input dataset with additional derived variables:

        * **T** — temperature (K)
        * **R_condensate** — condensate mixing ratio (kg/kg)
        * **P** — pressure (hPa)
        * **dewpoint** — dewpoint temperature (K)
        * **vapor_pressure** — water vapor pressure (hPa)
        * **theta_e** — equivalent potential temperature (K)
        * **saturation_vapor_pressure** — (hPa)
        * **air_mass** — per-grid-cell air mass (kg)
        * **RH** — relative humidity (fraction, 0–1)
        * **supersaturated** — boolean flag
        * **theta_v** — virtual potential temperature (K)
        * **theta_rho** — density potential temperature (K)
        * **buoyancy** — buoyancy acceleration (m/s²)
        * **R_liquid** — total liquid water mixing ratio (kg/kg)
        * **R_ice** — total ice mixing ratio (kg/kg)
        * **PCPRR_mm_hr** — precipitation rate (mm/hr)
        * **mixing_ratio** — water vapor mixing ratio (kg/kg)
        * **air_density** — moist air density (kg/m³)

    Raises:
        ValueError: If *fail_if_missing_vars* is ``True`` and core
            input variables are absent.
    """
    needed_vars = [
        "PI", "THETA", "RTP", "RV", "DN0",
        "RCP", "RRP", "RPP", "RSP", "RAP", "RGP", "RHP",
    ]

    def vars_are_present(names: list[str]) -> bool:
        return all(x in ds.data_vars for x in names)

    if not vars_are_present(needed_vars) and fail_if_missing_vars:
        raise ValueError(
            "Not all variables needed for thermodynamic calculations are present "
            f"in dataset and fail_if_missing_vars was True; required: {needed_vars}"
        )

    if vars_are_present(["PI", "THETA"]):
        ds["T"] = ds["PI"] * ds["THETA"] / 1004.0

    if vars_are_present(["RTP", "RV"]):
        ds["R_condensate"] = ds["RTP"] - ds["RV"]

    if vars_are_present(["PI"]):
        ds["P"] = 1000.0 * ((ds["PI"] / 1004.0) ** (1004.0 / 287.0))

    if vars_are_present(["P", "RV"]):
        vp = mpc.vapor_pressure(ds["P"] * units("hPa"), ds["RV"] * units("kg/kg"))
        ds["dewpoint"] = mpc.dewpoint(vp).pint.to("K").pint.dequantify()
        ds["vapor_pressure"] = vp.pint.to("hPa").pint.dequantify()

    if vars_are_present(["P", "RV"]):
        ds["theta_e"] = (
            mpc.equivalent_potential_temperature(
                pressure=ds["P"] * units("hPa"),
                temperature=ds["T"] * units("K"),
                dewpoint=ds["dewpoint"] * units("K"),
            )
            .pint.to("K")
            .pint.dequantify()
        )

    if vars_are_present(["T"]):
        ds["saturation_vapor_pressure"] = (
            mpc.saturation_vapor_pressure(ds["T"] * units("K"))
            .pint.to("hPa")
            .pint.dequantify()
        )

    if vars_are_present(["DN0"]):
        ds["air_mass"] = ds["DN0"] * 500**2 * ds["z"].diff(dim="z")

    if vars_are_present(["P", "T", "RV"]):
        ds["RH"] = mpc.relative_humidity_from_mixing_ratio(
            ds["P"] * units("hPa"), ds["T"] * units("K"), ds["RV"]
        ).pint.dequantify()
        ds["supersaturated"] = ds["RH"] >= 1
        ds["theta_v"] = mpc.virtual_potential_temperature(
            pressure=ds["P"] * units("hPa"),
            temperature=ds["T"] * units("K"),
            mixing_ratio=ds["RV"],
        )

    if vars_are_present(["THETA", "RV", "R_condensate"]):
        ds["theta_rho"] = ds["THETA"] * (1 + 0.608 * ds["RV"] - ds["R_condensate"])

    if vars_are_present(["theta_rho"]) and "x" in ds.dims and "y" in ds.dims:
        tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
        ds["buoyancy"] = (
            mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
        ).pint.dequantify()

    if vars_are_present(["RCP", "RRP"]):
        ds["R_liquid"] = ds["RCP"] + ds["RRP"]

    if vars_are_present(["RPP", "RSP", "RAP", "RGP", "RHP"]):
        ds["R_ice"] = ds["RPP"] + ds["RSP"] + ds["RAP"] + ds["RGP"] + ds["RHP"]

    if vars_are_present(["PCPRR"]):
        ds["PCPRR_mm_hr"] = (
            (ds["PCPRR"] * units("kg/m^2/s") / mpconstants.density_water)
            .pint.to("mm/hr")
            .pint.dequantify()
        )

    if vars_are_present(["vapor_pressure", "P"]):
        ds["mixing_ratio"] = mpc.mixing_ratio(
            partial_press=ds["vapor_pressure"] * units("hPa"),
            total_press=ds["P"] * units("hPa"),
        ).pint.dequantify()

    if vars_are_present(["P", "T", "mixing_ratio"]):
        ds["air_density"] = (
            mpc.density(
                pressure=ds["P"] * units("hPa"),
                temperature=ds["T"] * units("K"),
                mixing_ratio=ds["mixing_ratio"],
            )
            .pint.to("kg/m^3")
            .pint.dequantify()
        )

    return ds


def calculate_derived_variables(storm_ds: xr.Dataset) -> xr.Dataset:
    """Compute derived variables and apply standard preprocessing to RAMS output.

    Calls :func:`calculate_thermodynamic_variables`, shifts x/y coordinates
    to start from zero, adds a ``t_minutes`` coordinate, and computes
    horizontal vorticity and divergence.

    Args:
        storm_ds: Dataset containing RAMS storm simulation output.

    Returns:
        Dataset with derived variables and preprocessing applied.
    """
    print("Calculating derived variables...")
    storm_ds = calculate_thermodynamic_variables(storm_ds)

    storm_ds["x"] = storm_ds["x"] - min(storm_ds["x"])
    storm_ds["y"] = storm_ds["y"] - min(storm_ds["y"])

    storm_ds = storm_ds.assign_coords(
        t_minutes=(storm_ds["time"] - storm_ds["time"].values[0]).dt.total_seconds() // 60
    )

    storm_ds["vertical_vorticity"] = (
        storm_ds["VC"].differentiate("x") - storm_ds["UC"].differentiate("y")
    )
    storm_ds["divergence"] = (
        storm_ds["UC"].differentiate("x") + storm_ds["VC"].differentiate("y")
    )

    for var in ["x", "y"]:
        storm_ds[f"{var}_middle"] = storm_ds[var].max().values / 2
        storm_ds[f"{var}_middle_ix"] = len(storm_ds[var]) // 2

    return storm_ds


def calculate_bsr_variables(
    ds: xr.Dataset,
    base_state: xr.Dataset,
    bsr_variables: list[str] | None = None,
) -> xr.Dataset:
    """Compute base-state-relative perturbation variables.

    The base state is horizontally averaged to produce a vertical profile,
    which is subtracted from *ds* to yield perturbation fields named
    ``{var}_bsr``.

    Args:
        ds: Input dataset.
        base_state: Reference dataset (must **not** have a time dimension).
        bsr_variables: Variables to process.  Defaults to
            :data:`~xarrams.constants.DEFAULT_BSR_VARIABLES`.

    Returns:
        Copy of *ds* with ``{var}_bsr`` variables added.

    Raises:
        ValueError: If *base_state* contains a time dimension.
    """
    if "time" in base_state.dims:
        raise ValueError("base_state dataset must not have a time dimension, to avoid confusion")

    ds = ds.copy()
    base_state = base_state.mean(["x", "y"])
    for var in bsr_variables or DEFAULT_BSR_VARIABLES:
        if var in ds.data_vars:
            ds[f"{var}_bsr"] = ds[var] - base_state[var]
    return ds
