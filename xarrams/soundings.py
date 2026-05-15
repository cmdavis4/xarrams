"""Atmospheric sounding generation, plotting, and I/O for RAMS.

Includes the Weisman & Klemp (1984) idealized sounding, SkewT plotting,
and utilities for writing soundings in RAMS-compatible format.
"""

from __future__ import annotations

from typing import Optional, Union, List

import matplotlib.pyplot as plt
import metpy.calc as mpc
import metpy.constants as mpconstants
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr

from carlee_tools.types_carlee_tools import PathLike

from .constants import SOUNDING_NAMELIST_VARIABLES


def format_sounding_field_ramsin_str(
    values: Union[List[float], np.ndarray], decimal_places: int = 4
) -> str:
    """Format sounding field values as RAMSIN-compatible string.

    Args:
        values: Array of values for a sounding field
        decimal_places: Number of decimal places to use in formatting (default: 4)

    Returns:
        Formatted string suitable for RAMSIN files
    """
    values = np.array(values)
    format_str = f"%.{decimal_places}f"

    return ",\n          ".join([
        np.array2string(
            values[ix : ix + 5],
            formatter={"float_kind": lambda x: format_str % x},
            separator=", ",
        )[1:-1]
        for ix in range(0, len(values), 5)
    ])


def with_updated_sounding_fields(
    this_param_set: dict[str, str],
    sounding: pd.DataFrame,
    update_sounding_field_flags: bool = True,
) -> dict[str, str]:
    """Return a copy of *this_param_set* with sounding data injected.

    Args:
        this_param_set: RAMSIN parameter dictionary to update.
        sounding: DataFrame with columns ``PS``, ``TS``, ``RTS``, ``US``, ``VS``.
        update_sounding_field_flags: Also set ``IPSFLG``, ``ITSFLG``,
            ``IRTSFLG``, and ``IUSFLG`` to standard values.

    Returns:
        New parameter dictionary with sounding fields replaced.
    """
    this_param_set = dict(this_param_set)
    this_param_set.update({
        "PS": format_sounding_field_ramsin_str(sounding["PS"].values),
        "TS": format_sounding_field_ramsin_str(sounding["TS"].values),
        "RTS": format_sounding_field_ramsin_str(sounding["RTS"].values),
        "US": format_sounding_field_ramsin_str(sounding["US"].values),
        "VS": format_sounding_field_ramsin_str(sounding["VS"].values),
    })
    if update_sounding_field_flags:
        print(
            "Setting pressures to mb, temps to °C, RHs to percent, wind to U and V"
            " components"
        )
        this_param_set.update({
            "IPSFLG": "0",
            "ITSFLG": "0",
            "IRTSFLG": "3",
            "IUSFLG": "0",
        })
    return this_param_set


def write_rams_formatted_sounding(
    df: pd.DataFrame,
    output_path: PathLike,
    second_copy: Optional[PathLike] = None,
) -> None:
    """Write sounding data to a RAMS-formatted CSV file.

    Args:
        df: DataFrame with at least the columns ``PS``, ``TS``, ``RTS``, ``US``, ``VS``.
        output_path: Primary output file path.
        second_copy: Optional secondary path for a backup copy.

    Raises:
        ValueError: If required columns are missing or ``PS`` is not
            monotonically decreasing with unique values.
    """
    if not all(x in df.columns for x in SOUNDING_NAMELIST_VARIABLES):
        raise ValueError(
            f"Sounding dataframe must contain columns {SOUNDING_NAMELIST_VARIABLES}"
        )
    if not (df["PS"].is_monotonic_decreasing and df["PS"].nunique() == len(df)):
        raise ValueError(
            "'PS' field must be monotonically decreasing with no duplicate values"
        )

    output_paths = [output_path]
    if second_copy:
        output_paths.append(second_copy)
    for path in output_paths:
        df[SOUNDING_NAMELIST_VARIABLES].to_csv(
            str(path),
            sep=",",
            header=False,
            index=False,
            float_format="%.4f",
            lineterminator=",\n",
        )


def plot_sounding(
    column_df: pd.DataFrame,
    barbs: bool = True,
) -> plt.Figure:
    """Plot a SkewT diagram (with optional hodograph) from a vertical column.

    Args:
        column_ds: 1-D dataset indexed by ``z`` containing at least ``P``,
            ``T``, ``dewpoint``, ``UC``, and ``VC``.
        barbs: If ``True``, draw wind barbs; otherwise draw an inset hodograph.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If *column_ds* does not have exactly one dimension named ``z``.
    """
    from metpy.plots import Hodograph, SkewT
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    column_df = column_df[column_df["z"] >= 0].sort_values("z")

    fig = plt.figure()
    skewt = SkewT(fig, rotation=30)
    skewt.plot(column_df["PS"], column_df["TS"], "r")
    # Calculate the dewpoint
    if "dewpoint" not in column_df.columns:
        column_df["dewpoint"] = (
            mpc.dewpoint_from_relative_humidity(
                temperature=column_df["TS"].values * units("degC"),
                relative_humidity=column_df["RTS"].to_numpy(dtype=float) / 100,
            )
            .to("degC")
            .magnitude
        )
    skewt.plot(column_df["PS"].values, column_df["dewpoint"].values, "blue")

    wk_parcel_path = mpc.parcel_profile(
        column_df["PS"].values * units("hPa"),
        column_df["TS"].iloc[0].item() * units("K"),
        column_df["dewpoint"].iloc[0].item() * units("degC"),
    )
    skewt.plot(
        column_df["PS"],
        wk_parcel_path,
        color="grey",
        linestyle="dashed",
        linewidth=2,
    )
    skewt.ax.set_xlabel("Temperature (°C)")

    if barbs:
        coarsened = column_df.iloc[::4]
        skewt.plot_barbs(
            pressure=coarsened["PS"] * units("hPa"),
            u=coarsened["US"] * units("m/s"),
            v=coarsened["VS"] * units("m/s"),
        )
    else:
        ax_hod = inset_axes(skewt.ax, "40%", "40%", loc=1)
        component_range = max(column_df["US"].max(), column_df["VS"].max()) + 1
        h = Hodograph(ax_hod, component_range=component_range)
        h.add_grid(increment=10)
        h.plot_colormapped(column_df["US"], column_df["VS"], column_df["z"])

    # Dual-label y-axis with pressure and height
    p_ticks = skewt.ax.get_yticks()
    z_at_ticks = np.interp(p_ticks, column_df["PS"][::-1], column_df["z"][::-1])
    new_labels = []
    for p, z in zip(p_ticks, z_at_ticks):
        height_str = f"{z / 1000:.1f} km" if z >= 1000 else f"{int(z)} m"
        new_labels.append(f"{height_str}, {int(p)} hPa")
    skewt.ax.set_yticklabels(new_labels)

    return fig


def calculate_sounding_derived_vars(df):
    df["dewpoint"] = mpc.dewpoint_from_relative_humidity(
        temperature=df["TS"].values * units("degC"),
        relative_humidity=df["RTS"].values * units("percent"),
    ).to("degC")
    df["q_v"] = mpc.mixing_ratio_from_relative_humidity(
        pressure=df["PS"].values * units("hPa"),
        temperature=df["TS"].values * units("degC"),
        relative_humidity=df["RTS"].values * units("percent"),
    ).to("g/kg")
    df["theta"] = mpc.potential_temperature(
        pressure=df["PS"].values * units("hPa"),
        temperature=df["TS"].values * units("degC"),
    ).to("K")
    df["theta_v"] = mpc.virtual_potential_temperature(
        pressure=df["PS"].values * units("hPa"),
        temperature=df["TS"].values * units("degC"),
        mixing_ratio=df["q_v"].values * units("g/kg"),
    ).to("K")
    df["theta_e"] = mpc.equivalent_potential_temperature(
        pressure=df["PS"].values * units("hPa"),
        temperature=df["TS"].values * units("degC"),
        dewpoint=df["dewpoint"].values * units("degC"),
    ).to("K")
    return df


def wk84_sounding(
    U_s: float,
    q_v0: float,
    veering: bool,
    z_levels: Union[np.ndarray, list[float]],
    shear_layer_depth: float = 4000,
    z_tropopause: float = 12_000,
    theta_tropopause: float = 343,
    T_tropopause: float = 213,
    theta_0: float = 300,
    p_sfc: float = 100_000,
    max_height: float = 23_000,
    z_increment: float = 10,
) -> pd.DataFrame:
    """Generate an idealized Weisman & Klemp (1984) atmospheric sounding.

    Implements the analytic θ and RH profiles of WK84 eq (1)–(2), recovers
    pressure via a single-pass hydrostatic integration with θ_v ≈ θ, and
    applies the q_v0 cap of WK84 throughout the column. The "well-mixed"
    boundary layer in WK84 arises *only* from this cap clipping q_v wherever
    RH·q_sat would exceed q_v0 — θ itself is stably stratified at roughly
    2 K/km in the lower troposphere, not flat.

    Args:
        U_s: Maximum surface U-wind speed (m/s).
        q_v0: Cap on water-vapor mixing ratio (g/kg). Applied wherever
            RH·q_sat would otherwise exceed it; this is what produces the
            quasi-mixed BL in WK84.
        shear_layer_depth: Depth of the wind-shear layer (m).
        veering: ``True`` for a semicircular (veering) hodograph;
            ``False`` for unidirectional linear shear.
        z_levels: Heights (m) at which to interpolate the final sounding.
        z_tropopause: Tropopause height (m).
        theta_tropopause: Potential temperature at the tropopause (K).
        T_tropopause: Temperature at the tropopause (K).
        theta_0: Surface potential temperature (K).
        p_sfc: Surface pressure (Pa). WK84 use 100 000 Pa.
        max_height: Top of the high-resolution construction grid (m).
        z_increment: Spacing of the construction grid (m).

    Returns:
        DataFrame with columns ``z`` (m), ``PS`` (hPa), ``TS`` (°C),
        ``RTS`` (percent RH), ``US`` (m/s), ``VS`` (m/s).

    References:
        Weisman, M. L. & Klemp, J. B. (1984). *Mon. Wea. Rev.*, 112, 2479–2498.

        Seigel, R. B. & van den Heever, S. C. (2014). *Mon. Wea. Rev.*, 142, 1087–1104.
    """
    # Thermodynamic constants — matched to RAMS rconstants for consistency.
    C_p = 1004.0  # J/(kg·K)
    R_d = 287.0  # J/(kg·K)
    p_00 = 100_000.0  # Pa
    g = mpconstants.earth_gravity.to("m/s^2").magnitude

    z_levels = np.asarray(z_levels)
    wk_zs = np.arange(0, max_height, z_increment)

    # Analytic θ (WK84 eq 1) — no flattening; lower troposphere is ~2 K/km stable.
    wk_theta = np.where(
        wk_zs <= z_tropopause,
        theta_0 + (theta_tropopause - theta_0) * (wk_zs / z_tropopause) ** (5.0 / 4),
        theta_tropopause * np.exp(g * (wk_zs - z_tropopause) / (C_p * T_tropopause)),
    )

    # Analytic RH (WK84 eq 2).
    wk_rh = np.where(
        wk_zs <= z_tropopause,
        1.0 - 0.75 * (wk_zs / z_tropopause) ** (5.0 / 4),
        0.25,
    )

    # Single-pass hydrostatic integration of dπ/dz = -g / (c_p θ_v) with θ_v ≈ θ.
    # Moisture changes θ_v by at most ~0.85% (since q_v ≤ ~0.014), so the
    # q_v feedback into pressure is well below WK84's precision. No closed form
    # for the integral of 1/θ exists with this θ profile, so we trapezoidally
    # integrate on the uniform construction grid.
    dpi_dz = -g / (C_p * wk_theta)
    pi_sfc = (p_sfc / p_00) ** (R_d / C_p)
    wk_pi = pi_sfc + np.concatenate((
        [0.0],
        np.cumsum(0.5 * (dpi_dz[:-1] + dpi_dz[1:]) * np.diff(wk_zs)),
    ))
    wk_p = p_00 * wk_pi ** (C_p / R_d)  # Pa
    wk_T = wk_theta * wk_pi  # K

    # q_v0 cap (WK84): clip q_v at q_v0 wherever RH·q_sat would exceed it.
    # This is what produces the apparent well-mixed BL.
    q_sat = (
        mpc.saturation_mixing_ratio(wk_p * units("Pa"), wk_T * units("K"))
        .to("kg/kg")
        .magnitude
    )
    q_v = np.minimum(wk_rh * q_sat, q_v0 * 1e-3)  # q_v0 g/kg → kg/kg
    wk_rh = q_v / q_sat

    # Wind shear (Seigel & van den Heever 2014 pressure-based formulation).
    shear_layer_top_z_idx = np.argmin(np.abs(wk_zs - shear_layer_depth))
    if veering:
        V_s = U_s / 2
        pressure_norm = (wk_p - wk_p[0]) / (wk_p[shear_layer_top_z_idx] - wk_p[0])
        wk_U = (-U_s / 2) * (np.cos(np.pi * pressure_norm) - 1)
        wk_U[shear_layer_top_z_idx + 1 :] = wk_U[shear_layer_top_z_idx]
        wk_V = V_s * np.sin(np.pi * pressure_norm)
        wk_V[shear_layer_top_z_idx + 1 :] = wk_V[shear_layer_top_z_idx]
    else:
        linear_winds = np.linspace(0, U_s, shear_layer_top_z_idx)
        wk_U = np.zeros(len(wk_zs))
        wk_U[:shear_layer_top_z_idx] = linear_winds
        wk_U[shear_layer_top_z_idx:] = linear_winds[-1]
        wk_V = np.zeros(len(wk_U))

    df = pd.DataFrame({
        "z": z_levels,
        "PS": np.interp(z_levels, wk_zs, wk_p / 100.0),  # Pa → hPa
        "TS": np.interp(z_levels, wk_zs, wk_T - 273.15),  # K → °C
        "RTS": np.interp(z_levels, wk_zs, wk_rh * 100.0),  # frac → percent
        "US": np.interp(z_levels, wk_zs, wk_U),
        "VS": np.interp(z_levels, wk_zs, wk_V),
    })

    return df
