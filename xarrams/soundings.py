"""Atmospheric sounding generation, plotting, and I/O for RAMS.

Includes the Weisman & Klemp (1984) idealized sounding, SkewT plotting,
and utilities for writing soundings in RAMS-compatible format.
"""

from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
import metpy.calc as mpc
import metpy.constants as mpconstants
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr

from carlee_tools.types_carlee_tools import PathLike

from .constants import SOUNDING_NAMELIST_VARIABLES


def format_sounding_field_ramsin_str(values: Union[list[float], np.ndarray]) -> str:
    """Format sounding field values as a RAMSIN-compatible string.

    Produces a comma-separated multi-line string with 5 values per line,
    suitable for direct inclusion in a RAMSIN namelist.

    Args:
        values: Numeric values to format.

    Returns:
        Formatted string.
    """
    values = np.asarray(values, dtype=float)
    return ",\n          ".join(
        np.array2string(
            values[ix : ix + 5],
            formatter={"float_kind": lambda x: "%.4f" % x},
            separator=",    ",
        )[1:-1]
        for ix in range(0, len(values), 5)
    )


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
        print("Setting pressures to mb, temps to °C, RHs to percent, wind to U and V components")
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
        raise ValueError(f"Sounding dataframe must contain columns {SOUNDING_NAMELIST_VARIABLES}")
    if not (df["PS"].is_monotonic_decreasing and df["PS"].nunique() == len(df)):
        raise ValueError("'PS' field must be monotonically decreasing with no duplicate values")

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
    column_ds: xr.Dataset,
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

    if "z" not in column_ds.dims or len(column_ds.dims) != 1:
        raise ValueError(
            f'column_ds must have exactly one dimension named "z", but has: {column_ds.dims}'
        )

    column_df = column_ds.to_dataframe().reset_index()
    column_df = column_df[column_df["z"] >= 0].sort_values("z")

    fig = plt.figure()
    skewt = SkewT(fig, rotation=30)
    skewt.plot(column_df["P"], column_df["T"] - 273.15, "r")
    skewt.plot(column_df["P"], column_df["dewpoint"] - 273.15, "blue")

    wk_parcel_path = mpc.parcel_profile(
        column_df["P"].values * units("hPa"),
        column_df["T"].iloc[0] * units("K"),
        column_df["dewpoint"].iloc[0] * units("K"),
    )
    skewt.plot(
        column_df["P"],
        wk_parcel_path,
        color="grey",
        linestyle="dashed",
        linewidth=2,
    )
    skewt.ax.set_xlabel("Temperature (°C)")

    if barbs:
        coarsened = column_df.iloc[::4]
        skewt.plot_barbs(
            pressure=coarsened["P"] * units("hPa"),
            u=coarsened["UC"] * units("m/s"),
            v=coarsened["VC"] * units("m/s"),
        )
    else:
        ax_hod = inset_axes(skewt.ax, "40%", "40%", loc=1)
        component_range = max(column_df["UC"].max(), column_df["VC"].max()) + 1
        h = Hodograph(ax_hod, component_range=component_range)
        h.add_grid(increment=10)
        h.plot_colormapped(column_df["UC"], column_df["VC"], column_df["z"])

    # Dual-label y-axis with pressure and height
    p_ticks = skewt.ax.get_yticks()
    z_at_ticks = np.interp(p_ticks, column_df["P"][::-1], column_df["z"][::-1])
    new_labels = []
    for p, z in zip(p_ticks, z_at_ticks):
        height_str = f"{z / 1000:.1f} km" if z >= 1000 else f"{int(z)} m"
        new_labels.append(f"{height_str}, {int(p)} hPa")
    skewt.ax.set_yticklabels(new_labels)

    return fig


def wk84_sounding(
    U_s: float,
    q_v0: float,
    shear_layer_depth: float,
    veering: bool,
    z_levels: Union[np.ndarray, list[float]],
    z_tropopause: float = 12_000,
    theta_tropopause: float = 343,
    T_tropopause: float = 213,
    theta_0: float = 300,
    max_height: float = 23_000,
    z_increment: float = 10,
) -> pd.DataFrame:
    """Generate an idealized Weisman & Klemp (1984) atmospheric sounding.

    Creates a thermodynamic sounding for numerical weather simulations
    (supercell studies, MCS research, etc.) with configurable wind shear.

    Args:
        U_s: Maximum surface U-wind speed (m/s).
        q_v0: Surface water-vapor mixing ratio (g/kg).
        shear_layer_depth: Depth of the wind-shear layer (m).
        veering: ``True`` for a semicircular (veering) hodograph;
            ``False`` for unidirectional linear shear.
        z_levels: Heights (m) at which to interpolate the final sounding.
        z_tropopause: Tropopause height (m).
        theta_tropopause: Potential temperature at the tropopause (K).
        T_tropopause: Temperature at the tropopause (K).
        theta_0: Surface potential temperature (K).
        max_height: Top of the high-resolution construction grid (m).
        z_increment: Spacing of the construction grid (m).

    Returns:
        DataFrame with columns ``z``, ``P``, ``T``, ``RH``, ``U``, ``V``.

    References:
        Weisman, M. L. & Klemp, J. B. (1984). *Mon. Wea. Rev.*, 112, 2479–2498.

        Seigel, R. B. & van den Heever, S. C. (2014). *Mon. Wea. Rev.*, 142, 1087–1104.
    """
    C_p = 1004  # J/(kg·K), hardcoded for RAMS consistency
    z_levels = np.asarray(z_levels)

    wk_zs = np.arange(0, max_height, z_increment)
    wk_ps = mpc.height_to_pressure_std(wk_zs * units("m"))

    shear_layer_top_z_idx = np.argmin(np.abs(wk_zs - shear_layer_depth))

    if veering:
        V_s = U_s / 2
        pressure_norm = (wk_ps - wk_ps[0]) / (wk_ps[shear_layer_top_z_idx] - wk_ps[0])
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

    wk_theta = np.where(
        wk_zs <= z_tropopause,
        theta_0 + (theta_tropopause - theta_0) * (wk_zs / z_tropopause) ** (5.0 / 4),
        theta_tropopause
        * np.exp(mpconstants.earth_gravity * (wk_zs - z_tropopause) / (C_p * T_tropopause)),
    )

    p_idx_900hpa = np.argmin(np.abs(wk_ps - 900 * units("hPa")))
    wk_theta[:p_idx_900hpa] = wk_theta[p_idx_900hpa]

    wk_rhs = np.where(
        wk_zs <= z_tropopause,
        1.0 - 0.75 * (wk_zs / z_tropopause) ** (5.0 / 4),
        0.25,
    )

    wk_Ts = mpc.temperature_from_potential_temperature(wk_ps, wk_theta * units("K")).to("degC")

    q_v0_rhs = mpc.relative_humidity_from_mixing_ratio(wk_ps, wk_Ts, q_v0 * units("g/kg")).to("")
    wk_rhs = np.where(q_v0_rhs < 1, q_v0_rhs, wk_rhs)

    wk_df = pd.DataFrame({
        "PS": wk_ps,
        "TS": wk_Ts,
        "RTS": wk_rhs * 100.0,
        "US": wk_U,
        "VS": wk_V,
    })

    return pd.DataFrame({
        "z": z_levels,
        "P": np.interp(z_levels, wk_zs, wk_df["PS"]),
        "T": np.interp(z_levels, wk_zs, wk_df["TS"]),
        "RH": np.interp(z_levels, wk_zs, wk_df["RTS"]),
        "U": np.interp(z_levels, wk_zs, wk_df["US"]),
        "V": np.interp(z_levels, wk_zs, wk_df["VS"]),
    })
