"""Thermodynamic and derived variable calculations for RAMS output.

Provides functions to compute derived meteorological quantities
(temperature, pressure, humidity, buoyancy, etc.) from raw RAMS
model output fields using MetPy.
"""

from __future__ import annotations

import xarray as xr
import pandas as pd
import metpy.calc as mpc
import metpy.constants as mpconstants
from metpy.units import units
import matplotlib.pyplot as plt

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
        "PI",
        "THETA",
        "RTP",
        "RV",
        "DN0",
        "RCP",
        "RRP",
        "RPP",
        "RSP",
        "RAP",
        "RGP",
        "RHP",
    ]

    def vars_are_present(names: list[str]) -> bool:
        if not passed_dataframe:
            return all(x in ds.data_vars for x in names)
        else:
            # This will work for a dict, which is what we coerce pandas to
            return all(x in ds for x in names)

    # If this is pandas, convert it to a dict of arrays so we don't have to
    # add .values to every index into a variable
    passed_dataframe = isinstance(ds, pd.DataFrame)

    if passed_dataframe:
        ds = ds.to_dict(orient="list")

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

    if vars_are_present(["PCPRR"]):
        ds["PCPRR_mm_hr"] = (
            ((ds["PCPRR"] * units("kg m^-2 s^-1")) / mpconstants.density_water)
            .pint.to("mm/hr")
            .pint.dequantify()
        )
    if vars_are_present(["ACCPR"]):
        ds["ACCPR_mm"] = (
            ((ds["ACCPR"] * units("kg m^-2")) / mpconstants.density_water)
            .pint.to("mm")
            .pint.dequantify()
        )

    # Variables that are calculated using xarray functionality
    if not passed_dataframe:
        if vars_are_present(["theta_rho"]) and "x" in ds.dims and "y" in ds.dims:
            tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
            ds["buoyancy"] = (
                mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
            ).pint.dequantify()

        if vars_are_present(["DN0"]):
            ds["air_mass"] = ds["DN0"] * 500**2 * ds["z"].diff(dim="z")

    if passed_dataframe:
        return pd.DataFrame(ds)
    else:
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

    ds = ds.copy()
    if "time" in base_state.coords:
        if "time" in ds.coords:
            ds = ds.assign_coords(
                t_minutes=(ds["time"] - base_state["time"]).dt.total_seconds() / 60
            )
        # Then drop it from the base state to avoid confusion
        base_state = base_state.squeeze()

    base_state = base_state.mean(["x", "y"])
    for var in bsr_variables or DEFAULT_BSR_VARIABLES:
        if var in ds.data_vars:
            ds[f"{var}_bsr"] = ds[var] - base_state[var]
    return ds


import numpy as np
import pint
from scipy.integrate import cumulative_trapezoid

ureg = pint.UnitRegistry()
Q = ureg.Quantity


def flux_to_tendency(
    grid_spacing,
    atten_length,
    flux_amp,
    cp=1004 * units("J/(kg*K)"),
    surface_pressure=1000 * units("hPa"),
    surface_temp=298 * units("K"),
    R_d=287 * units("J/(kg*K)"),
):
    """
    Approximate K/hr potential-temperature tendency produced in the lowest
    model layer by a Klaasen-and-Clark surface flux forcing.

    Assumes zmn(2) ≈ 0 (lowest layer starts at the surface) and the layer has
    thickness `grid_spacing`. The vertical-decay factor for the lowest layer
    is then (1 − exp(−dz/L)), and the heating rate is

        dθ/dt = flux_amp · (1 − exp(−dz/L)) / (ρ · dz · cp).

    For dz ≪ L this asymptotes to flux_amp / (ρ · L · cp); grid spacing drops
    out. Density is computed from the ideal-gas law at the given surface
    pressure and temperature.

    Parameters
    ----------
    grid_spacing : pint Quantity (length)
        Vertical thickness of the lowest model layer.
    atten_length : pint Quantity (length)
        K&C vertical e-folding depth (= zt at iflux_k_atten in RAMS).
    flux_amp : pint Quantity (power / area)
        Peak column-integrated surface flux (= flux_amp_wm2 in RAMSIN).
    """
    rho = (surface_pressure / (R_d * surface_temp)).to("kg/m**3")
    ratio = (grid_spacing / atten_length).to("dimensionless").magnitude
    decay = 1.0 - np.exp(-ratio)
    layer_flux = flux_amp * decay  # W/m² deposited in the lowest layer
    return (layer_flux / (rho * grid_spacing * cp)).to("K/hour")


def tendency_to_flux(
    grid_spacing,
    atten_length,
    tendency,
    cp=1004 * units("J/(kg*K)"),
    surface_pressure=1000 * units("hPa"),
    surface_temp=298 * units("K"),
    R_d=287 * units("J/(kg*K)"),
):
    """Inverse of `flux_to_tendency`: required flux_amp for a desired tendency."""
    rho = (surface_pressure / (R_d * surface_temp)).to("kg/m**3")
    ratio = (grid_spacing / atten_length).to("dimensionless").magnitude
    decay = 1.0 - np.exp(-ratio)
    return (tendency * rho * grid_spacing * cp / decay).to("W/m**2")


def integrated_dtheta(
    grid_spacing,
    z_atten_length,
    x_atten_length,
    flux_amp,
    rampup_duration,
    peak_duration,
    rampdown_duration,
    cp=1004 * units("J/(kg*K)"),
    surface_pressure=1000 * units("hPa"),
    surface_temp=298 * units("K"),
    R_d=287 * units("J/(kg*K)"),
    timestep=1 * units("s"),
):
    """
    Total potential-temperature change imparted to the lowest model layer
    over a full ramp-up / peak / ramp-down K&C surface flux pulse, plus
    diagnostic plots including a 2D x-z cross section of max heating rate
    and total ΔΘ.

    The temporal factor matches the RAMS branching in ruser.f90: linear
    ramp 0→1 over [tstart, tmax), constant 1 over [tmax, tdecay), linear
    ramp 1→0 over [tdecay, tend), 0 otherwise.

    Vertical and horizontal decay match the K&C distribution used in RAMS:
        vert_decay(k)  = exp(-(z_k - z_0)/L_z) − exp(-(z_{k+1} - z_0)/L_z)
        horiz_gauss(x) = exp(−(x / L_x)²)

    Returns
    -------
    pint Quantity (K)
        Cumulative ΔΘ at the end of the pulse in the lowest layer at the
        forcing center.
    """
    # Build time grid + temporal factor, matching ruser.f90 branching
    tstart = 0 * units("s")
    tmax = tstart + rampup_duration
    tdecay = tmax + peak_duration
    tend = tdecay + rampdown_duration
    times = np.arange(
        0, tend.m_as("s") + timestep.m_as("s"), timestep.m_as("s")
    ) * units("s")
    t = times.m_as("s")
    ts, tm, td, te = (
        tstart.m_as("s"),
        tmax.m_as("s"),
        tdecay.m_as("s"),
        tend.m_as("s"),
    )
    temporal_factor = np.where(
        t < ts,
        0.0,
        np.where(
            t < tm,
            (t - ts) / (tm - ts),
            np.where(
                t < td,
                1.0,
                np.where(t < te, 1.0 - (t - td) / (te - td), 0.0),
            ),
        ),
    )
    column_fluxes = temporal_factor * flux_amp

    # K&C lowest-layer decay factor — matches flux_to_tendency in xrr.calculations
    decay = 1.0 - np.exp(-(grid_spacing / z_atten_length).m_as("dimensionless"))
    layer_fluxes = column_fluxes * decay

    # scipy strips units, so hand it plain magnitudes in known units and
    # re-attach the result unit (W/m^2 * s = J/m^2)
    def integrate_flux(fluxes):
        return cumulative_trapezoid(
            y=fluxes.m_as("W/m^2"),
            x=times.m_as("s"),
            initial=0,
        ) * units("J/m^2")

    integrated_column_flux = integrate_flux(column_fluxes)
    integrated_layer_flux = integrate_flux(layer_fluxes)

    rho = (surface_pressure / (R_d * surface_temp)).to("kg/m^3")
    dTheta_dt = (layer_fluxes / (rho * grid_spacing * cp)).to("K/hr")
    integrated_dTheta = (integrated_layer_flux / (rho * grid_spacing * cp)).to("K")

    # 2D x-z grids for cross sections — extend several atten lengths in each direction
    n_z = int(np.ceil((6 * z_atten_length / grid_spacing).m_as("dimensionless")))
    z_edges = np.arange(n_z + 1) * grid_spacing
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    ze = z_edges.m_as("m")
    Lz = z_atten_length.m_as("m")
    vert_decay_per_layer = np.exp(-ze[:-1] / Lz) - np.exp(-ze[1:] / Lz)

    n_x_half = int(np.ceil((4 * x_atten_length / grid_spacing).m_as("dimensionless")))
    x_centers = np.arange(-n_x_half, n_x_half + 1) * grid_spacing
    xc = x_centers.m_as("m")
    Lx = x_atten_length.m_as("m")
    horiz_gauss = np.exp(-((xc / Lx) ** 2))

    # 2D fields: outer product of vert_decay (n_z,) and horiz_gauss (n_x,)
    decay_2d = np.outer(vert_decay_per_layer, horiz_gauss)  # (n_z, n_x)
    heating_rate_max_2d = (flux_amp * decay_2d / (rho * grid_spacing * cp)).to("K/hr")
    total_dtheta_2d = (
        integrated_column_flux[-1] * decay_2d / (rho * grid_spacing * cp)
    ).to("K")

    fig, axs = plt.subplots(
        ncols=2,
        nrows=4,
        figsize=(8, 12),
        layout="constrained",
        squeeze=False,
    )

    axs[0, 0].plot(times, column_fluxes)
    axs[0, 1].plot(times, integrated_column_flux)
    axs[1, 0].plot(times, layer_fluxes)
    axs[1, 1].plot(times, integrated_layer_flux)
    axs[2, 0].plot(times, dTheta_dt)
    axs[2, 1].plot(times, integrated_dTheta)

    axs[0, 0].set_title(r"$F$ (column)")
    axs[0, 1].set_title(r"Total energy imparted (column)")
    axs[1, 0].set_title(r"$F$ (lowest model layer)")
    axs[1, 1].set_title(r"Total energy imparted (lowest model layer)")
    axs[2, 0].set_title(r"$d\theta/dt$")
    axs[2, 1].set_title(r"$\Delta \theta$")

    im0 = axs[3, 0].pcolormesh(
        x_centers.m_as("km"),
        z_centers.m_as("m"),
        heating_rate_max_2d.m_as("K/hr"),
        shading="auto",
    )
    fig.colorbar(im0, ax=axs[3, 0], label="K/hr")
    axs[3, 0].set_title(r"Max instantaneous $d\theta/dt$")
    axs[3, 0].set_xlabel("x (km)")
    axs[3, 0].set_ylabel("z (m)")

    im1 = axs[3, 1].pcolormesh(
        x_centers.m_as("km"),
        z_centers.m_as("m"),
        total_dtheta_2d.m_as("K"),
        shading="auto",
    )
    fig.colorbar(im1, ax=axs[3, 1], label="K")
    axs[3, 1].set_title(r"Total $\Delta \theta$")
    axs[3, 1].set_xlabel("x (km)")
    axs[3, 1].set_ylabel("z (m)")

    return integrated_dTheta[-1]


def bubble_perturbation_field(
    theta_amp,
    moisture_amp,
    horizontal_radius,
    vertical_radius,
    vertical_center,
    grid_spacing=50 * units("m"),
    base_theta=300 * units("K"),
    base_rv=0.01,
):
    """
    Plot 2D x-z cross sections of θ and θ_v from a RAMSIN cosine-squared
    bubble (ibubble=2/4 in ruser.f90), allowing `vertical_center` near or
    below the surface for bubbles that intersect the ground.

    Shape factor at each (x, z):
        r_norm = sqrt((x / horizontal_radius)² + ((z - vertical_center) / vertical_radius)²)
        factor = cos²(π · r_norm / 2)   for r_norm < 1, else 0
    θ(x,z)   = base_theta + theta_amp · factor
    r_v(x,z) = base_rv · (1 + moisture_amp · factor)
    θ_v(x,z) ≈ θ · (1 + 0.608 · r_v)

    Parameters
    ----------
    theta_amp : pint Quantity (K)
        BTHP — peak θ perturbation at bubble center.
    moisture_amp : float
        BRTP — fractional peak r_v perturbation (e.g. 0.2 = +20%).
    horizontal_radius, vertical_radius : pint Quantity (length)
        Bubble half-widths (bubradx, bubradz).
    vertical_center : pint Quantity (length)
        Height of bubble center above ground; can be small or negative
        so that the bubble is partially below z=0.
    base_theta : pint Quantity (K)
    base_rv : float (kg/kg)
        Background state used so θ_v has a meaningful absolute value.
    """
    # x-z grid: extend 1.5x bubble radius horizontally; vertically from
    # z=0 (ground) up through the top of the bubble
    x_extent = 1.5 * horizontal_radius
    n_x_half = int(np.ceil((x_extent / grid_spacing).m_as("dimensionless")))
    x_centers = np.arange(-n_x_half, n_x_half + 1) * grid_spacing

    z_top = vertical_center + 1.5 * vertical_radius
    n_z = int(np.ceil((z_top / grid_spacing).m_as("dimensionless")))
    z_centers = (np.arange(n_z) + 0.5) * grid_spacing

    X, Z = np.meshgrid(x_centers.m_as("m"), z_centers.m_as("m"), indexing="xy")
    rx = X / horizontal_radius.m_as("m")
    rz = (Z - vertical_center.m_as("m")) / vertical_radius.m_as("m")
    r_norm = np.sqrt(rx**2 + rz**2)
    factor = np.where(r_norm < 1, np.cos(np.pi * r_norm / 2) ** 2, 0.0)

    theta = base_theta.m_as("K") + theta_amp.m_as("K") * factor  # K
    rv = base_rv * (1 + moisture_amp * factor)  # kg/kg
    theta_v = theta * (1 + 0.608 * rv)  # K
    theta_v_base = base_theta.m_as("K") * (1 + 0.608 * base_rv)

    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(10, 4), layout="constrained", squeeze=False
    )

    im0 = axs[0, 0].pcolormesh(
        x_centers.m_as("km"),
        z_centers.m_as("m"),
        theta - base_theta.m_as("K"),
        shading="auto",
    )
    fig.colorbar(im0, ax=axs[0, 0], label="K")
    axs[0, 0].set_title(r"$\Delta \theta$")
    axs[0, 0].set_xlabel("x (km)")
    axs[0, 0].set_ylabel("z (m)")

    im1 = axs[0, 1].pcolormesh(
        x_centers.m_as("km"),
        z_centers.m_as("m"),
        theta_v - theta_v_base,
        shading="auto",
    )
    fig.colorbar(im1, ax=axs[0, 1], label="K")
    axs[0, 1].set_title(r"$\Delta \theta_v$")
    axs[0, 1].set_xlabel("x (km)")
    axs[0, 1].set_ylabel("z (m)")

    return theta, theta_v
