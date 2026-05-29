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
from pint import Quantity
import xarray as xr

from carlee_tools.types_carlee_tools import PathLike
from carlee_tools.plotting import clean_legend

from .constants import SOUNDING_NAMELIST_VARIABLES, C_p, R_d, p0, reps


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
            "Setting pressures to hPa, temps to K, RHs to percent, wind to U and V"
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
    skew=None,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    subplot_spec=None,
) -> plt.Figure:
    """Plot a SkewT diagram (with optional hodograph) from a vertical column.

    Args:
        column_df: 1-D dataframe indexed by ``z`` containing at least ``PS``,
            ``TS``, ``RTS`` (or ``dewpoint``), ``US``, and ``VS``.
        barbs: If ``True``, draw wind barbs; otherwise draw an inset hodograph.
        skew: Optional pre-built ``metpy.plots.SkewT`` to draw into. When
            provided, the other placement arguments are ignored.
        ax: Optional existing Axes whose grid slot should host the SkewT.
            The axes is removed and a SkewT is created in its place — handy
            for dropping a sounding into one panel of a ``plt.subplots`` grid.
        fig: Optional matplotlib Figure to attach a new SkewT to. If none of
            ``skew``, ``ax``, or ``fig`` is given, a new figure is created.
        subplot_spec: Optional ``SubplotSpec`` (or 3-int tuple) specifying
            where in ``fig`` to place the SkewT — use this to embed the
            sounding in one panel of a multi-panel figure.

    Returns:
        The matplotlib Figure containing the SkewT.
    """
    from metpy.plots import Hodograph, SkewT
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    column_df = column_df[column_df["z"] >= 0].sort_values("z")

    if skew is not None:
        skewt = skew
        fig = skewt.ax.figure
    else:
        if ax is not None:
            fig = ax.figure
            subplot_spec = ax.get_subplotspec()
            ax.remove()
        elif fig is None:
            fig = plt.figure()
        skewt = SkewT(fig, rotation=30, subplot=subplot_spec)
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

    parcel_Ts = mpc.parcel_profile(
        column_df["PS"].values * units("hPa"),
        column_df["TS"].iloc[0].item() * units("degC"),
        column_df["dewpoint"].iloc[0].item() * units("degC"),
    )
    skewt.plot(
        column_df["PS"],
        parcel_Ts.to("degC").magnitude,
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


_WK_DEFAULT_THETA_0 = 300 * units("K")
_WK_DEFAULT_Z_TROPOPAUSE = 12_000 * units("m")
_WK_DEFAULT_THETA_TROPOPAUSE = 343 * units("K")
_WK_DEFAULT_T_TROPOPAUSE = 213 * units("K")


def _WK_DEFAULT_THETA_CALCULATION(
    z,
    theta_0=_WK_DEFAULT_THETA_0,
    z_tropopause=_WK_DEFAULT_Z_TROPOPAUSE,
    theta_tropopause=_WK_DEFAULT_THETA_TROPOPAUSE,
    T_tropopause=_WK_DEFAULT_T_TROPOPAUSE,
):
    return np.where(
        z < z_tropopause,
        theta_0 + (theta_tropopause - theta_0) * (z / z_tropopause) ** 1.25,
        theta_tropopause
        * np.exp((mpconstants.g / (T_tropopause * C_p)) * (z - z_tropopause)),
    )


def _WK_DEFAULT_RH_CALCULATION(
    z,
    z_tropopause=_WK_DEFAULT_Z_TROPOPAUSE,
):
    return np.where(z < z_tropopause, 1.0 - 0.75 * (z / z_tropopause) ** 1.25, 0.25)


def wk84_sounding(
    U_s: Quantity,
    q_v0: Quantity,
    veering: bool = False,
    z_levels: Union[np.ndarray, list[float], None] = None,
    shear_layer_depth: Quantity = 4000 * units("m"),
    z_tropopause: Quantity = _WK_DEFAULT_Z_TROPOPAUSE,
    theta_tropopause: Quantity = _WK_DEFAULT_THETA_TROPOPAUSE,
    T_tropopause: Quantity = _WK_DEFAULT_T_TROPOPAUSE,
    theta_0: Quantity = _WK_DEFAULT_THETA_0,
    p_sfc: Quantity = p0,
    max_height: Quantity = 23_000 * units("m"),
    z_increment: Quantity = 10 * units("m"),
    theta_fn: callable = _WK_DEFAULT_THETA_CALCULATION,
    rh_fn: callable = _WK_DEFAULT_RH_CALCULATION,
    diagnostics: bool = True,
    plot: bool = False,
) -> pd.DataFrame:
    """Generate an idealized Weisman & Klemp (1984) atmospheric sounding.

    Implements the analytic θ and RH profiles of WK84 eq (1)–(2), recovers
    pressure via a single-pass hydrostatic integration with θ_v ≈ θ, and
    applies the q_v0 cap of WK84 throughout the column. The "well-mixed"
    boundary layer in WK84 arises *only* from this cap clipping q_v wherever
    RH·q_sat would exceed q_v0 — θ itself is stably stratified at roughly
    2 K/km in the lower troposphere, not flat.

    Args:
        U_s: Maximum surface U-wind speed (unitful, e.g. ``m/s``).
        q_v0: Cap on water-vapor mixing ratio (unitful, e.g. ``g/kg``).
            Applied wherever RH·q_sat would otherwise exceed it; this is
            what produces the quasi-mixed BL in WK84.
        veering: ``True`` for a semicircular (veering) hodograph;
            ``False`` for unidirectional linear shear.
        z_levels: Heights (m) at which to interpolate the final sounding.
            If ``None``, the internal construction grid is returned.
        shear_layer_depth: Depth of the wind-shear layer (unitful length).
        z_tropopause: Tropopause height (unitful length).
        theta_tropopause: Potential temperature at the tropopause
            (unitful temperature).
        T_tropopause: Temperature at the tropopause (unitful temperature).
        theta_0: Surface potential temperature (unitful temperature).
        p_sfc: Surface pressure (unitful pressure). WK84 use 1000 hPa.
        max_height: Top of the high-resolution construction grid
            (unitful length).
        z_increment: Spacing of the construction grid (unitful length).
        theta_fn: Callable returning the θ profile given ``z`` and the
            tropopause parameters; defaults to the WK84 analytic form.
        rh_fn: Callable returning the RH profile given ``z`` and
            ``z_tropopause``; defaults to the WK84 analytic form.

    Returns:
        DataFrame with columns ``z`` (m), ``PS`` (hPa), ``TS`` (°C),
        ``RTS`` (percent RH), ``US`` (m/s), ``VS`` (m/s).

    References:
        Weisman, M. L. & Klemp, J. B. (1984). *Mon. Wea. Rev.*, 112, 2479–2498.

        Seigel, R. B. & van den Heever, S. C. (2014). *Mon. Wea. Rev.*, 142, 1087–1104.
    """

    # This is intended to be as exact a translation of CM1's implementation of this
    # sounding as possible
    # Thermodynamic constants — matched to RAMS rconstants for consistency.

    # Calculate derived quantities for the surface
    pi_sfc = (p_sfc / p0) ** (R_d / C_p)
    T_sfc = mpc.temperature_from_potential_temperature(
        pressure=p_sfc, potential_temperature=theta_0
    )
    qv_sfc = mpc.saturation_mixing_ratio(
        total_press=p_sfc,
        temperature=T_sfc,
    )
    theta_v_sfc = theta_0 * (1.0 + qv_sfc * reps) / (1.0 + qv_sfc)

    # Initialize z values we'll use
    internal_zs = np.arange(
        0,
        (max_height + 1 * units("m")).to("m").magnitude,
        step=z_increment.to("m").magnitude,
    ) * units("m")

    # The potential temperature and RH are analytically prescribed, from which we'll then diagnose q_v and pressure
    thetas = theta_fn(
        z=internal_zs,
        theta_0=theta_0,
        z_tropopause=z_tropopause,
        theta_tropopause=theta_tropopause,
        T_tropopause=T_tropopause,
    )
    rhs = rh_fn(z=internal_zs, z_tropopause=z_tropopause)

    # Strip units once for the inner numerical work; reattach at boundaries.
    # The pint Quantity arithmetic in the inner z-loop dominates runtime
    # otherwise.
    p0_Pa = p0.to("Pa").magnitude
    C_p_val = C_p.to("J/kg/K").magnitude
    R_d_val = R_d.to("J/kg/K").magnitude
    g_val = mpconstants.g.to("m/s^2").magnitude
    q_v0_val = q_v0.to("kg/kg").magnitude
    pi_sfc_val = float(pi_sfc.magnitude if hasattr(pi_sfc, "magnitude") else pi_sfc)
    theta_v_sfc_val = theta_v_sfc.to("K").magnitude
    zs_val = internal_zs.to("m").magnitude
    thetas_val = np.asarray(
        thetas.to("K").magnitude if hasattr(thetas, "to") else thetas
    )
    rhs_val = np.asarray(rhs.magnitude if hasattr(rhs, "magnitude") else rhs)

    # Initialize mixing ratios and pressures
    qvs = np.zeros(len(zs_val))
    pis = np.zeros(len(zs_val))

    # CM1 does a hardcoded 20 iterations
    for _ in range(20):
        theta_vs = thetas_val * (1 + reps * qvs) / (1 + qvs)
        # For pressure, integrate over the vertical, starting from this
        pis[0] = pi_sfc_val - g_val * zs_val[0] / (
            C_p_val * 0.5 * (theta_v_sfc_val + theta_vs[0])
        )
        for z_ix in range(1, len(zs_val)):
            pis[z_ix] = pis[z_ix - 1] - g_val * (zs_val[z_ix] - zs_val[z_ix - 1]) / (
                C_p_val * 0.5 * (theta_vs[z_ix] + theta_vs[z_ix - 1])
            )
        Ps_val = p0_Pa * (pis ** (C_p_val / R_d_val))
        # Re-attach units at the metpy call boundary; metpy uses its own
        # internal R_d / C_p / p0 which differ slightly from constants.py,
        # so an inline T = theta * (P/p0)^(R_d/C_p) substitution shifts
        # the dewpoint closure vs CM1.
        Ps = Ps_val * units("Pa")
        Ts = mpc.temperature_from_potential_temperature(
            pressure=Ps, potential_temperature=thetas_val * units("K")
        )
        q_vsat = (
            mpc.saturation_mixing_ratio(total_press=Ps, temperature=Ts)
            .to("kg/kg")
            .magnitude
        )
        Ts_val = Ts.to("K").magnitude
        qvs = rhs_val * q_vsat
        # Cap it at q_v0
        qvs = np.minimum(qvs, q_v0_val)

    # Reattach units for downstream code that expects Quantities
    Ps = Ps_val * units("Pa")
    Ts = Ts_val * units("K")
    # Convert this back to an RH; I assume this is just for full consistency?
    rhs = (
        mpc.relative_humidity_from_mixing_ratio(
            pressure=Ps,
            temperature=Ts,
            mixing_ratio=qvs * units("dimensionless"),
        )
        .to("dimensionless")
        .magnitude
    )

    # Wind shear (Seigel & van den Heever 2014 pressure-based formulation).
    shear_layer_top_z_idx = np.argmin(np.abs(internal_zs - shear_layer_depth))
    if veering:
        V_s = U_s / 2
        pressure_norm = (Ps - Ps[0]) / (Ps[shear_layer_top_z_idx] - Ps[0])
        wk_U = (-U_s / 2) * (np.cos(np.pi * pressure_norm) - 1)
        wk_U[shear_layer_top_z_idx + 1 :] = wk_U[shear_layer_top_z_idx]
        wk_V = V_s * np.sin(np.pi * pressure_norm)
        wk_V[shear_layer_top_z_idx + 1 :] = wk_V[shear_layer_top_z_idx]
    else:
        linear_winds = np.linspace(0, U_s, shear_layer_top_z_idx)
        wk_U = np.zeros(len(internal_zs))
        wk_U[:shear_layer_top_z_idx] = linear_winds
        wk_U[shear_layer_top_z_idx:] = linear_winds[-1]
        wk_V = np.zeros(len(wk_U))

    if z_levels is not None:
        internal_zs = internal_zs.to("m").magnitude
        df = pd.DataFrame({
            "z": z_levels,
            "PS": np.interp(z_levels, internal_zs, Ps.to("hPa").magnitude),
            "TS": np.interp(z_levels, internal_zs, Ts.to("degC").magnitude),
            "RTS": np.interp(z_levels, internal_zs, rhs * 100.0),  # frac → percent
            "US": np.interp(z_levels, internal_zs, wk_U),
            "VS": np.interp(z_levels, internal_zs, wk_V),
        })
    else:
        df = pd.DataFrame({
            "z": internal_zs,
            "PS": Ps.to("hPa").magnitude,
            "TS": Ts.to("degC").magnitude,
            "RTS": rhs * 100.0,  # frac → percent
            "US": wk_U,
            "VS": wk_V,
        })

    return df
