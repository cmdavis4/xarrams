"""Atmospheric sounding generation, plotting, and I/O for RAMS.

Includes the Weisman & Klemp (1984) idealized sounding, SkewT plotting,
and utilities for writing soundings in RAMS-compatible format.
"""

from __future__ import annotations
import warnings

import os
import re
from typing import Optional, Union, List

import matplotlib.pyplot as plt
from matplotlib import font_manager
import metpy.calc as mpc
import metpy.constants as mpconstants
from metpy.units import units
import numpy as np
import pandas as pd
from pint import Quantity
import xarray as xr
from tqdm.auto import tqdm

from carlee_tools.types_carlee_tools import PathLike
from carlee_tools.plotting import clean_legend, get_nth_color

from .constants import SOUNDING_NAMELIST_VARIABLES, C_p, R_d, p0, reps
from .calculations import calculate_thermodynamic_variables
from .execution import _parse_ramsin_field


def to_ramsin_values_str(
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


def format_sounding_field_ramsin_str(*args, **kwargs):
    warnings.warn(
        "format_sounding_field_ramsin_str has been renamed to to_ramsin_values_str. "
        "format_sounding_field_ramsin_str is deprecated and will be removed eventually;"
        " use the new name instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return to_ramsin_values_str(*args, **kwargs)


def sounding_df_to_ramsin_dict(df):
    return {k: to_ramsin_values_str(df[k].values) for k in SOUNDING_NAMELIST_VARIABLES}


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
    warnings.warn(
        "with_updated_sounding_fields is deprecated and will be removed eventually. Use"
        " sounding_df_to_ramsin_dict instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    this_param_set = dict(this_param_set)
    this_param_set.update(sounding_df_to_ramsin_dict(sounding))
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


def plot_sounding_skewt(
    column_df: pd.DataFrame,
    barbs: bool = True,
    mixing_ratios=[],
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
        ax_hod = inset_axes(skewt.ax, "25%", "25%", loc=1)
        component_range = max(column_df["US"].max(), column_df["VS"].max()) + 1
        h = Hodograph(ax_hod, component_range=component_range)
        h.add_grid(increment=10)
        h.plot_colormapped(column_df["US"], column_df["VS"], column_df["z"])
        # The inset is a fixed fraction of the skew-T panel's size regardless of
        # how large that panel is, so its tick labels need their own small,
        # absolute size rather than inheriting the ambient (possibly
        # slide-scaled) rcParams tick size, which overlaps at this footprint.
        ax_hod.tick_params(axis="both", labelsize=8)

    # Add lines of constant mixing ratio
    if mixing_ratios:
        skewt.plot_mixing_lines(
            mixing_ratio=np.array(mixing_ratios)
        )

    # Dual-label y-axis with pressure and height. Deliberately relabels the
    # existing tick positions (via set_yticklabels alone, not set_yticks first)
    # rather than pinning them to a FixedLocator — the pressure axis is
    # log-scaled, and constrained_layout can still revise its view limits after
    # this point; pinning ticks here left minor-tick autoscaling solving against
    # a stale range and raising a log-domain math error at draw time.
    pressure_tick_locations = skewt.ax.get_yticks()
    height_at_pressure_ticks_m = np.interp(
        pressure_tick_locations, column_df["PS"][::-1], column_df["z"][::-1]
    )
    dual_axis_tick_labels = []
    for pressure_hpa, height_m in zip(pressure_tick_locations, height_at_pressure_ticks_m):
        height_str = f"{height_m / 1000:.1f}km" if height_m >= 1000 else f"{int(height_m)}m"
        dual_axis_tick_labels.append(f"{int(pressure_hpa)} hPa · {height_str}")
    # These combined "pressure · height" labels are long and, on the log-scaled
    # pressure axis, pack tightly together toward the surface (equal hPa steps
    # span shrinking log-distance at higher pressure) — so they need a smaller
    # font than the ambient tick size to stay legible, scaled off that ambient
    # size (which may be a named size like "medium", hence resolving it through
    # FontProperties) rather than a fixed absolute value, so this still tracks
    # whatever matplotlib style is active.
    ambient_ytick_fontsize = font_manager.FontProperties(
        size=plt.rcParams["ytick.labelsize"]
    ).get_size_in_points()
    dual_axis_tick_fontsize = max(9, 0.6 * ambient_ytick_fontsize)
    skewt.ax.set_yticklabels(dual_axis_tick_labels, fontsize=dual_axis_tick_fontsize)
    skewt.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    skewt.ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
    # 10 degC-spaced major ticks across a -40..50 range are the standard skew-T
    # convention and pack tightly at the ambient tick size regardless of panel
    # width; shrink to match the y-axis dual-label size above so the two axes
    # read as one consistent (smaller-than-ambient) scale for this panel.
    skewt.ax.tick_params(axis="x", labelsize=dual_axis_tick_fontsize)
    skewt.ax.grid(which="minor", axis="x", alpha=0.3)

    return fig


# Temporary alias for backwards compatibility
plot_sounding = plot_sounding_skewt


def to_sounding_df(ds):
    # Reject datasets with multiple times
    ds = ds.copy().squeeze()
    if ("time" in ds.dims) or ("time" in ds.coords):
        if ds["time"].size > 1:
            raise ValueError(
                "ds cannot have a time dimension with more than one value, to avoid"
                " confusion"
            )
    # If it has x and y, take the mean over them
    mean_vars = [x for x in ds.dims if x in ["x", "y"]]
    if mean_vars:
        ds = ds.mean(mean_vars)
    # Create a clean copy, calculate state variables we'll need, drop fictitious z
    # levels
    ds = calculate_thermodynamic_variables(ds)
    ds = ds.sel(z=ds["z"].values[ds["z"].values > 0])
    return pd.DataFrame({
        "z": ds["z"],
        "PS": ds["P"],
        "TS": (ds["T"].values * units("K")).to("degC").magnitude,
        "RTS": ds["RH"] * 100.0,  # frac → percent
        "US": ds["UC"],
        "VS": ds["VC"],
    })


def _parse_ramsin_sounding_array(
    ramsin_text: str, field_name: str
) -> Optional[np.ndarray]:
    """Parse a multi-line sounding array (``PS``, ``TS``, ...) from a RAMSIN.

    The single-field parser in :mod:`execution` stops at the first comma,
    which truncates the comma-separated sounding profiles. This grabs every
    value of *field_name* from its ``=`` up to the next namelist variable
    assignment (a line beginning ``NAME =``) or the closing ``$END``.

    Args:
        ramsin_text: Full text of the RAMSIN namelist.
        field_name: Sounding field to extract (e.g. ``"PS"``).

    Returns:
        Array of the field's values, or ``None`` if the field is absent.
    """
    # Capture everything after "FIELD =" lazily until we hit either the next
    # namelist assignment (a line that starts with an identifier followed by
    # "="), the end of the namelist block ($END), or the end of the text.
    field_value_match = re.search(
        rf"(?ms)^\s*{re.escape(field_name)}\s*=\s*(.*?)"
        rf"(?=^\s*[A-Za-z_]\w*\s*=|^\s*\$END\b|\Z)",
        ramsin_text,
    )
    if field_value_match is None:
        return None

    # Drop any Fortran "!" trailing comments before tokenizing the numbers.
    raw_values_text = re.sub(r"!.*", "", field_value_match.group(1))

    # Split on commas/whitespace and parse each token as a float, translating
    # Fortran double-precision exponent markers ("1.0D3") into Python form.
    sounding_values = []
    for token in re.split(r"[,\s]+", raw_values_text.strip()):
        if not token:
            continue
        sounding_values.append(float(token.replace("D", "e").replace("d", "e")))
    return np.array(sounding_values)


def sounding_df_from_ramsin(ramsin: Union[PathLike, str]) -> pd.DataFrame:
    """Parse a RAMSIN's initial sounding into a sounding DataFrame.

    Reads the ``$MODEL_SOUND`` block of a RAMSIN — the ``IPSFLG``, ``ITSFLG``,
    ``IRTSFLG``, and ``IUSFLG`` unit flags plus the ``PS``, ``TS``, ``RTS``,
    ``US``, and ``VS`` profile arrays — and converts it to the same column
    layout produced by :func:`to_sounding_df`: ``z`` (m), ``PS`` (hPa),
    ``TS`` (degC), ``RTS`` (percent RH), ``US`` (m/s), ``VS`` (m/s).

    The unit conversions mirror RAMS' own ``arrsnd`` routine
    (``src/init/rhhi.f90``) so the returned profile matches what RAMS actually
    ingests — including the hydrostatic integration RAMS uses to assign height
    levels (``z``) to the pressure-specified input sounding.

    Args:
        ramsin: Path to a RAMSIN file, or the raw RAMSIN text itself.

    Returns:
        DataFrame with columns ``z``, ``PS``, ``TS``, ``RTS``, ``US``, ``VS``.

    Raises:
        NotImplementedError: If ``IPSFLG`` is not 0 (millibar levels). The
            ``IPSFLG=1`` height-level form needs RAMS' interleaved hydrostatic
            pressure integration, which is not reproduced here.
        ValueError: If a required sounding flag or array is missing.
    """
    # Accept either a path to a RAMSIN file or the namelist text directly.
    try:
        ramsin_is_a_file = os.path.isfile(ramsin)
    except (TypeError, ValueError):
        # Very long strings / embedded nulls aren't valid path arguments.
        ramsin_is_a_file = False
    if ramsin_is_a_file:
        with open(ramsin) as ramsin_file:
            ramsin_text = ramsin_file.read()
    else:
        ramsin_text = str(ramsin)

    # --- Unit flags ---------------------------------------------------------
    # Read a scalar integer flag, reusing the single-field RAMSIN parser.
    def read_flag(flag_name: str, default: Optional[int] = None) -> int:
        raw_flag = _parse_ramsin_field(ramsin_text, flag_name)
        if raw_flag is None:
            if default is None:
                raise ValueError(f"RAMSIN sounding flag {flag_name} not found")
            return default
        return int(raw_flag)

    pressure_flag = read_flag("IPSFLG")  # 0=mb, 1=height(m) w/ PS(1)=sfc pressure
    temperature_flag = read_flag("ITSFLG")  # 0=degC, 1=K, 2=theta(K)
    moisture_flag = read_flag("IRTSFLG")  # 0/1=dewpt, 2=mixrat, 3=RH%, 4=dewpt depr.
    wind_flag = read_flag("IUSFLG", default=0)  # 0=U/V components, else dir/speed

    if pressure_flag not in (0, 1):
        raise NotImplementedError(
            "Only IPSFLG=0 (millibar levels) and IPSFLG=1 (height levels) are"
            f" supported; got IPSFLG={pressure_flag}."
        )

    # --- Sounding arrays ----------------------------------------------------
    raw_sounding_fields = {}
    for field_name in SOUNDING_NAMELIST_VARIABLES:
        field_values = _parse_ramsin_sounding_array(ramsin_text, field_name)
        if field_values is None:
            raise ValueError(f"RAMSIN sounding field {field_name} not found")
        raw_sounding_fields[field_name] = field_values

    pressure_mb = raw_sounding_fields["PS"]
    temperature_in = raw_sounding_fields["TS"]
    moisture_in = raw_sounding_fields["RTS"]
    us = raw_sounding_fields["US"].astype(float)
    vs = raw_sounding_fields["VS"].astype(float)

    # RAMS treats the first PS==0 entry as the end of the sounding; truncate
    # all fields to the number of real levels before that sentinel.
    end_of_sounding = np.where(pressure_mb == 0.0)[0]
    n_levels = int(end_of_sounding[0]) if end_of_sounding.size else len(pressure_mb)
    pressure_mb = pressure_mb[:n_levels]
    temperature_in = temperature_in[:n_levels]
    moisture_in = moisture_in[:n_levels]
    us = us[:n_levels]
    vs = vs[:n_levels]

    # --- Thermodynamic constants (matched to RAMS rconstants) ---------------
    R_d_val = R_d.to("J/kg/K").magnitude
    C_p_val = C_p.to("J/kg/K").magnitude
    p00_Pa = p0.to("Pa").magnitude
    # RAMS hardcodes g = 9.80 (rconstants.f90), not the standard 9.80665; use
    # its value so the hydrostatic integration matches RAMS bit-for-bit.
    g_val = 9.80
    R_over_cp = R_d_val / C_p_val  # rocp
    cp_over_R = C_p_val / R_d_val  # cpor
    p00_to_rocp = p00_Pa**R_over_cp  # p00k

    # --- Pressure: -> Pa ----------------------------------------------------
    # IPSFLG=0 gives pressure directly in millibars. IPSFLG=1 instead gives
    # heights (m) for levels 2..N, with PS(1) the surface pressure in mb, and
    # RAMS recovers pressure by integrating the hydrostatic equation upward.
    if pressure_flag == 0:
        # Pressure given in millibars.
        pressure_pa = pressure_mb * 100.0
    else:
        # PS holds the surface pressure (mb) then heights (m); the input array
        # is named *pressure_mb* but here its tail entries are heights.
        height_levels_m = pressure_mb
        # Virtual-temperature correction factor for the integration. RAMS only
        # applies it when moisture is supplied as a mixing ratio (IRTSFLG=2);
        # *moisture_in* is then in g/kg, hence the 1e-3.
        if moisture_flag == 2:
            virtual_correction = 1.0 + 0.61 * moisture_in * 1.0e-3
        else:
            virtual_correction = np.ones(n_levels)
        # Offset that converts the raw TS values to Kelvin for the integration
        # (only used for the temperature/Kelvin forms, not theta).
        temperature_offset = 273.15 if temperature_flag == 0 else 0.0

        pressure_pa = np.zeros(n_levels)
        pressure_pa[0] = height_levels_m[0] * 100.0  # surface pressure mb -> Pa
        for k in range(1, n_levels):
            # Layer thickness between successive input height levels.
            layer_thickness_m = height_levels_m[k] - height_levels_m[k - 1]
            if temperature_flag in (0, 1):
                # Layer-mean (virtual) temperature in Kelvin from the TS inputs.
                layer_mean_temperature = 0.5 * (
                    (temperature_in[k] + temperature_offset) * virtual_correction[k]
                    + (temperature_in[k - 1] + temperature_offset)
                    * virtual_correction[k - 1]
                )
                pressure_pa[k] = pressure_pa[k - 1] * np.exp(
                    -g_val * layer_thickness_m / (R_d_val * layer_mean_temperature)
                )
            else:  # temperature_flag == 2: TS holds potential temperature
                # Layer-mean (virtual) potential temperature.
                layer_mean_theta = 0.5 * (
                    temperature_in[k] * virtual_correction[k]
                    + temperature_in[k - 1] * virtual_correction[k - 1]
                )
                # Hydrostatic integration in Exner/theta form.
                pressure_pa[k] = (
                    pressure_pa[k - 1] ** R_over_cp
                    - g_val
                    * layer_thickness_m
                    * p00_to_rocp
                    / (C_p_val * layer_mean_theta)
                ) ** cp_over_R

    # --- Winds: direction/speed -> U,V components if needed -----------------
    # RAMS leaves 9999. entries untouched here and interpolates them below.
    if wind_flag != 0:
        wind_is_given = us != 9999.0
        wind_direction_deg = us[wind_is_given]
        wind_speed = vs[wind_is_given]
        us[wind_is_given] = -wind_speed * np.sin(np.deg2rad(wind_direction_deg))
        vs[wind_is_given] = -wind_speed * np.cos(np.deg2rad(wind_direction_deg))
    # Interpolate any 9999. (missing) wind levels in pressure, as RAMS does.
    wind_is_missing = us == 9999.0
    if wind_is_missing.any():
        wind_is_given = ~wind_is_missing
        # np.interp needs increasing sample points, but pressure decreases with
        # height, so reverse both the samples and their coordinates.
        us[wind_is_missing] = np.interp(
            pressure_pa[wind_is_missing],
            pressure_pa[wind_is_given][::-1],
            us[wind_is_given][::-1],
        )
        vs[wind_is_missing] = np.interp(
            pressure_pa[wind_is_missing],
            pressure_pa[wind_is_given][::-1],
            vs[wind_is_given][::-1],
        )

    # --- Temperature: -> Kelvin ---------------------------------------------
    if temperature_flag == 0:
        # Temperature given in degrees Celsius.
        temperature_k = temperature_in + 273.15
    elif temperature_flag == 1:
        # Temperature already in Kelvin.
        temperature_k = temperature_in.astype(float)
    elif temperature_flag == 2:
        # Potential temperature in Kelvin -> actual temperature.
        temperature_k = (pressure_pa / p00_Pa) ** R_over_cp * temperature_in
    else:
        raise ValueError(f"Unknown temperature flag ITSFLG={temperature_flag}")

    # --- Moisture: -> mixing ratio (kg/kg) ----------------------------------
    # Saturation mixing ratio at a given pressure/temperature, matching RAMS'
    # internal `mrsl` call (metpy's formulation differs trivially).
    def saturation_mixing_ratio(pressure_pa_arr, temperature_k_arr):
        return (
            mpc.saturation_mixing_ratio(
                total_press=pressure_pa_arr * units("Pa"),
                temperature=temperature_k_arr * units("K"),
            )
            .to("kg/kg")
            .magnitude
        )

    if moisture_flag == 0:
        # Dew point in degrees Celsius.
        mixing_ratio_kgkg = saturation_mixing_ratio(pressure_pa, moisture_in + 273.15)
    elif moisture_flag == 1:
        # Dew point in Kelvin.
        mixing_ratio_kgkg = saturation_mixing_ratio(pressure_pa, moisture_in)
    elif moisture_flag == 2:
        # Mixing ratio in g/kg.
        mixing_ratio_kgkg = moisture_in * 1.0e-3
    elif moisture_flag == 3:
        # Relative humidity in percent: scale the saturation mixing ratio.
        mixing_ratio_kgkg = (
            saturation_mixing_ratio(pressure_pa, temperature_k) * moisture_in * 0.01
        )
    elif moisture_flag == 4:
        # Dew point depression in Kelvin (subtracted from temperature).
        mixing_ratio_kgkg = saturation_mixing_ratio(
            pressure_pa, temperature_k - moisture_in
        )
    else:
        raise ValueError(f"Unknown humidity flag IRTSFLG={moisture_flag}")

    # --- Heights: hydrostatic integration from the surface (z=0) ------------
    # Mirrors the `hs` loop in arrsnd: integrate the hypsometric equation
    # downward in pressure using layer-mean virtual temperature.
    virtual_temperature_k = temperature_k * (1.0 + 0.61 * mixing_ratio_kgkg)
    heights_m = np.zeros(n_levels)
    for k in range(1, n_levels):
        # Layer-mean virtual temperature across the (k-1, k) interval.
        layer_mean_tv = 0.5 * (virtual_temperature_k[k] + virtual_temperature_k[k - 1])
        # Hypsometric thickness; pressure decreases upward so the log term is
        # negative, and the leading minus sign makes the height increase.
        heights_m[k] = (
            heights_m[k - 1]
            - R_d_val
            * layer_mean_tv
            * (np.log(pressure_pa[k]) - np.log(pressure_pa[k - 1]))
            / g_val
        )

    # --- Assemble the output DataFrame in to_sounding_df's column layout ----
    # Report RH in the same convention the rest of this module stores it in --
    # the plain ratio w / w_sat -- which is exactly what the IRTSFLG=3 branch
    # above inverts, and what calculate_sounding_derived_vars reads back.
    # metpy's relative_humidity_from_mixing_ratio would return the
    # thermodynamic e / e_sat instead, which round-trips against neither.
    relative_humidity_percent = (
        mixing_ratio_kgkg
        / saturation_mixing_ratio(pressure_pa, temperature_k)
        * 100.0  # frac -> percent
    )
    return pd.DataFrame({
        "z": heights_m,
        "PS": pressure_pa / 100.0,  # Pa -> hPa
        "TS": temperature_k - 273.15,  # K -> degC
        "RTS": relative_humidity_percent,
        "US": us,
        "VS": vs,
    })


def calculate_sounding_derived_vars(df, cape=True, cape_z_max=4000, cape_stride=1):
    df = df.copy()
    # `RTS` is stored in the CM1/RAMS sense -- the plain mixing-ratio ratio
    # w / w_sat -- because that is how both models recover qv from a sounding
    # (CM1 base.F: rh0 = qv0 / rslf), and that is what wk84_sounding writes.
    # metpy's mixing_ratio_from_relative_humidity instead treats its RH argument
    # as the thermodynamic e / e_sat, so feeding RTS to it does *not* invert what
    # was written: it returns a qv ~1.8% low at the surface and tilts the
    # well-mixed layer that the q_v0 cap exists to make flat. Recover qv with the
    # same convention it was stored in.
    Ps_for_qv = df["PS"].values * units("hPa")
    Ts_for_qv = df["TS"].values * units("degC")
    rh_ratio = (df["RTS"].values * units("percent")).to("dimensionless")
    df["q_v"] = (
        rh_ratio
        * mpc.saturation_mixing_ratio(
            total_press=Ps_for_qv, temperature=Ts_for_qv
        )
    ).to("g/kg")
    # Take the dewpoint from that same qv (through its vapor pressure) rather
    # than from RTS directly, so the dewpoint-derived diagnostics below (LCL,
    # theta_e, CAPE/CIN) sit on the same moisture profile that is written to the
    # model, instead of on a second, slightly different one.
    df["dewpoint"] = mpc.dewpoint(
        mpc.vapor_pressure(Ps_for_qv, df["q_v"].values * units("g/kg"))
    ).to("degC")
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
    df["lapse_rate"] = (df["TS"].diff() / df["z"].diff()) * 1000
    df["theta_lapse_rate"] = (df["theta"].diff() / df["z"].diff()) * 1000

    # Precalculate the relevant variables as unitful arrays
    Ps = df["PS"].values * units("hPa")
    Ts = df["TS"].values * units("degC")
    DPs = df["dewpoint"].values * units("degC")

    # LCL can accept arrays of values, so we calculate the LCL at every starting
    # height with a single (cheap) vectorized call.
    lcl_ps, _ = mpc.lcl(pressure=Ps, temperature=Ts, dewpoint=DPs)
    # Interpolate the LCL pressure back onto the column's pressure–height
    # relationship to get the LCL height. np.interp needs the sample points
    # (pressures) increasing, but PS decreases monotonically with height, so
    # reverse both arrays.
    df["lcl"] = np.interp(
        lcl_ps.to("hPa").magnitude,
        df["PS"].values[::-1],
        df["z"].values[::-1],
    )

    # CAPE/CIN profile. This is the expensive part: metpy's parcel_profile is a
    # per-level moist-adiabat integration, and we run one per *starting* level.
    #
    # We only vary the starting level of the parcel over the low levels (the
    # CAPE/CIN integration itself always extends over the full column above the
    # parcel — truncating the column truncates the positive area and undercounts
    # CAPE). `cape_stride` further subsamples the starting levels and linearly
    # interpolates between them, which is plenty for diagnostic plots when the
    # sounding is at fine (e.g. 10 m) vertical resolution.
    #   cape=True  -> compute over starting levels with z <= cape_z_max
    #   cape=False -> compute over the full column
    if cape:
        n_levels = int((df["z"] <= cape_z_max).sum())
    else:
        n_levels = len(df)
    start_ixs = list(range(0, n_levels, cape_stride))

    capes = np.full(len(df), np.nan)
    cins = np.full(len(df), np.nan)
    for ix in tqdm(start_ixs):
        profile = mpc.parcel_profile(
            pressure=Ps[ix:],
            temperature=Ts[ix],
            dewpoint=DPs[ix],
        )
        cape_ix, cin_ix = mpc.cape_cin(
            pressure=Ps[ix:],
            temperature=Ts[ix:],
            dewpoint=DPs[ix:],
            parcel_profile=profile,
        )
        capes[ix] = cape_ix.to("J/kg").magnitude
        cins[ix] = cin_ix.to("J/kg").magnitude

    # Fill in the skipped (strided) starting levels by interpolation.
    if cape_stride > 1 and len(start_ixs) > 1:
        filled = np.arange(start_ixs[-1] + 1)
        capes[: start_ixs[-1] + 1] = np.interp(filled, start_ixs, capes[start_ixs])
        cins[: start_ixs[-1] + 1] = np.interp(filled, start_ixs, cins[start_ixs])

    df["cape"] = capes * units("J/kg")
    df["cin"] = cins * units("J/kg")

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
    # Convert the capped mixing ratio back to the RH that gets written to the
    # sounding. RAMS (and CM1) recover qv from a sounding RH as
    #     qv = RH * q_sat(P, T),
    # i.e. they treat the stored RH as the simple ratio w / w_sat, so we must
    # build RH the same way here. metpy.relative_humidity_from_mixing_ratio
    # would instead return the thermodynamic humidity e / e_sat, which differs
    # by a factor (eps + w_sat) / (eps + w). That factor inflates the recovered
    # qv by ~1.5% above the q_v0 cap and, because w_sat falls with height, tilts
    # what should be a constant well-mixed-layer qv. Forming RH as w / w_sat
    # makes the recovered qv land flat on q_v0, matching CM1 base.F
    # (rh0 = qv0 / rslf). q_vsat is the saturation mixing ratio from the final
    # iteration above, evaluated at this same Ps / Ts.
    rhs = qvs / q_vsat

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


def plot_sounding_diagnostics(sounding_df, ll_z_cutoff=4000, axs=None):
    """Plot a 2x2 sounding summary: skew-T, theta/lapse rate, CAPE/CIN, moisture.

    The three non-skew-T panels share a height (z) vertical axis and are clipped
    to `ll_z_cutoff` — the low levels are what these particular diagnostics (BL
    theta structure, low-level CAPE/CIN, low-level moisture) are for, and a full
    tropospheric depth would squeeze that detail into a thin sliver. The skew-T
    keeps its native pressure/temperature axes and always shows the full column;
    it gets a secondary height label on its y-ticks from `plot_sounding_skewt`.

    A horizontal dotted line marks the surface parcel's LCL on every panel —
    pressure-valued on the skew-T since that is its native coordinate,
    height-valued on the other three — so the same physical level lines up
    across panels despite the differing y-axis quantities.

    Args:
        sounding_df: DataFrame from `to_sounding_df` and
            `calculate_sounding_derived_vars`, with at least `z`, `PS`, `theta`,
            `theta_v`, `theta_lapse_rate`, `cape`, `cin`, `q_v`, `RTS`, and `lcl`.
        ll_z_cutoff: Height (m) above which the three height-axis panels are
            clipped. Pass `None` to show the full column on every panel.
        axs: Optional pre-built 2x2 array of Axes (e.g. a slice of a larger
            `plt.subplots` grid) to draw into, for composing this as one block
            of a bigger figure. If omitted, a new figure is created.

    Returns:
        The matplotlib Figure containing the four panels.
    """
    if axs is None:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(9, 9), layout="constrained")
    else:
        axs = np.asarray(axs).reshape(2, 2)
        fig = axs[0, 0].figure

    # Skew-T (unaffected by ll_z_cutoff; always the full column). Despite the
    # name, plot_sounding_skewt returns the parent Figure, not a SkewT object —
    # its component axes (the SkewX plot itself, plus the hodograph/wind-barb
    # inset) are recovered below via is_skewt() filtering skewt_fig.axes.
    skewt_fig = plot_sounding_skewt(sounding_df, ax=axs[0, 0], barbs=False)
    skewt_ax = next(a for a in skewt_fig.axes if is_skewt(a))
    skewt_ax.set_title("Skew-T")

    if ll_z_cutoff:
        low_level_df = sounding_df.loc[sounding_df["z"] <= ll_z_cutoff]
    else:
        low_level_df = sounding_df

    # Potential temperature profiles, with lapse rate on a shared-y twin axis
    theta_ax = axs[0, 1]
    theta_ax.set_title("Potential temperature")
    theta_var_labels = {"theta": r"$\theta$", "theta_v": r"$\theta_v$", "theta_e": r"$\theta_e$"}
    for column_name, legend_label in theta_var_labels.items():
        theta_ax.plot(
            low_level_df[column_name].values, low_level_df["z"], label=legend_label
        )
    theta_ax.set_ylabel("z (m)")
    theta_ax.set_xlabel("K")
    # The raw finite-difference lapse rate is noisy at native (near-1s) sonde
    # resolution, so it's drawn thin and faint — a background-texture reference
    # rather than a headline series that would otherwise dominate the panel.
    lapse_rate_ax = theta_ax.twiny()
    (lapse_rate_line,) = lapse_rate_ax.plot(
        low_level_df["theta_lapse_rate"].values,
        low_level_df["z"],
        linestyle="dashed",
        linewidth=1.0,
        alpha=0.5,
        color="grey",
        label="lapse rate",
    )
    lapse_rate_ax.set_xlabel(r"$\Delta$K/km", color="grey")
    lapse_rate_ax.tick_params(axis="x", colors="grey")

    # CAPE/CIN
    cape_ax = axs[1, 0]
    for column_name in ["cape", "cin"]:
        cape_ax.plot(low_level_df[column_name].values, low_level_df["z"], label=column_name)
    cape_ax.set_ylabel("z (m)")
    cape_ax.set_xlabel("J/kg")
    cape_ax.set_title("CAPE/CIN")
    clean_legend(cape_ax, frameon=False)

    # Moisture: q_v and RH on a shared-y twin axis, color-coded onto the axis
    # labels/ticks instead of a legend box, matching the theta_v/theta contrast.
    moisture_ax = axs[1, 1]
    moisture_ax.set_title("Moisture")
    qv_color, rh_color = get_nth_color(0), get_nth_color(1)
    moisture_ax.plot(low_level_df["q_v"], low_level_df["z"], color=qv_color, label=r"$q_v$")
    moisture_ax.set_ylabel("z (m)")
    moisture_ax.set_xlabel(r"$q_v$ (g/kg)", color=qv_color)
    moisture_ax.tick_params(axis="x", colors=qv_color)
    rh_ax = moisture_ax.twiny()
    rh_ax.plot(low_level_df["RTS"], low_level_df["z"], color=rh_color, label="RH")
    rh_ax.set_xlabel("RH (%)", color=rh_color)
    rh_ax.tick_params(axis="x", colors=rh_color)

    for non_skewt_ax in [theta_ax, lapse_rate_ax, cape_ax, moisture_ax, rh_ax]:
        non_skewt_ax.minorticks_on()
        non_skewt_ax.grid(which="major", alpha=0.4)
        non_skewt_ax.grid(which="minor", alpha=0.15)

    # Surface parcel's LCL, marked on every panel: pressure-valued on the skew-T
    # (its native vertical coordinate), height-valued on the other three.
    lcl_height_m = sounding_df.iloc[0]["lcl"]
    lcl_pressure_hpa = np.interp(
        lcl_height_m, sounding_df["z"].values, sounding_df["PS"].values
    )
    skewt_ax.axhline(lcl_pressure_hpa, linestyle="dotted", color="skyblue", alpha=0.6)
    for height_axis_ax in [theta_ax, cape_ax, moisture_ax]:
        height_axis_ax.axhline(lcl_height_m, linestyle="dotted", color="skyblue", alpha=0.6)
    # Zero-length proxy artist so "LCL" appears in the theta panel's legend below
    # (the only panel with a full legend box, since moisture uses colored labels
    # instead and CAPE/CIN's legend is built separately via clean_legend).
    theta_ax.plot([], [], linestyle="dotted", color="skyblue", alpha=0.6, label="LCL")

    # The lapse-rate scatter on the twinned axis covers nearly the full height
    # and width of this panel, so "best" placement (which only weighs theta_ax's
    # own artists, not that scatter) can land the legend on top of data. Low-z is
    # always low-theta (theta increases with height), so the lower-right corner
    # is reliably data-free — but only if the legend box is kept short, hence
    # the reduced font and tightened spacing below (an ambient slide-scaled
    # legend.fontsize would make a 4-row box tall enough to reach back up into
    # the theta/theta_v curves).
    ambient_legend_fontsize = font_manager.FontProperties(
        size=plt.rcParams["legend.fontsize"]
    ).get_size_in_points()
    theta_legend_handles, theta_legend_labels = theta_ax.get_legend_handles_labels()
    theta_ax.legend(
        handles=theta_legend_handles + [lapse_rate_line],
        labels=theta_legend_labels + ["lapse rate"],
        loc="lower right",
        fontsize=max(9, 0.65 * ambient_legend_fontsize),
        handlelength=1.5,
        labelspacing=0.3,
        frameon=False,
    )

    return fig


def is_skewt(ax):
    from metpy.plots import skewt

    return isinstance(ax, skewt.SkewXAxes)


def plot_base_state_diagnostics(bs_ds):
    if "time" in bs_ds.dims:
        bs_ds = bs_ds.isel(time=0)
    bs_df = to_sounding_df(bs_ds)
    bs_df = calculate_sounding_derived_vars(bs_df)
    lcl = bs_df.iloc[0]["lcl"]

    # Always do sounding diagnostics
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(9, 13), layout="constrained")
    plot_sounding_diagnostics(bs_df, axs=axs[:2, :])

    # Do CCN if present
    if ("CN1NP" in bs_ds.data_vars) or ("CN2NP" in bs_ds.data_vars):
        ccn_ax = axs[2, 0]
        if "CN1NP" in bs_ds.data_vars:
            ccn_ax.plot(
                bs_ds["CN1NP"].mean(["x", "y"]).values / 1e6,
                bs_ds["z"].values,
                label="CCN1",
            )
        if "CN2NP" in bs_ds.data_vars:
            ccn_ax.plot(
                bs_ds["CN2NP"].mean(["x", "y"]).values / 1e6,
                bs_ds["z"].values,
                label="CCN2",
            )
        clean_legend(ccn_ax, frameon=False)
        ccn_ax.set_title("CCN number concentration")
        ccn_ax.set_xlabel("#/mg")
        ccn_ax.set_ylabel("z (m)")
    else:
        axs[2, 0].set_axis_off()

    axs[2, 1].set_axis_off()

    for ax in axs.flatten()[:-1]:
        ax.axhline(lcl, linestyle="dotted", color="skyblue", alpha=0.6)

    return fig
