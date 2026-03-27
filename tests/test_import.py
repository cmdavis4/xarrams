"""Test that all public symbols are importable."""

import xarrams


def test_version():
    """Version string is set."""
    assert isinstance(xarrams.__version__, str)
    assert xarrams.__version__  # non-empty


def test_function_imports():
    """Key functions are importable and callable."""
    from xarrams import (
        calculate_bsr_variables,
        calculate_derived_variables,
        calculate_thermodynamic_variables,
        fill_rams_output_dimensions,
        format_sounding_field_ramsin_str,
        generate_ramsin,
        get_datetime,
        get_grid_number,
        get_z_levels,
        infer_rams_dimensions,
        parse_rams_stdout_walltimes,
        plot_sounding,
        ramsin_str,
        read_rams_output,
        run_rams,
        to_header_filepath,
        to_rams_output_filename,
        to_t_minutes,
        with_t_minutes_coord,
        with_updated_sounding_fields,
        wk84_sounding,
        write_rams_formatted_sounding,
    )
    for fn in [
        calculate_bsr_variables,
        calculate_derived_variables,
        calculate_thermodynamic_variables,
        fill_rams_output_dimensions,
        format_sounding_field_ramsin_str,
        generate_ramsin,
        get_datetime,
        get_grid_number,
        get_z_levels,
        infer_rams_dimensions,
        parse_rams_stdout_walltimes,
        plot_sounding,
        ramsin_str,
        read_rams_output,
        run_rams,
        to_header_filepath,
        to_rams_output_filename,
        to_t_minutes,
        with_t_minutes_coord,
        with_updated_sounding_fields,
        wk84_sounding,
        write_rams_formatted_sounding,
    ]:
        assert callable(fn)


def test_constant_imports():
    """Constants are importable and have expected types."""
    import pandas as pd

    from xarrams import (
        DEFAULT_BSR_VARIABLES,
        HYDROMETEOR_SPECIES_FULL_NAMES,
        RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
        RAMS_DT_FORMAT,
        RAMS_DT_STRFTIME_STR,
        RAMS_FILENAME_DATETIME_REGEX,
        RAMS_VARIABLES_DF,
        SOUNDING_NAMELIST_VARIABLES,
        ureg,
    )

    assert isinstance(RAMS_DT_FORMAT, str)
    assert RAMS_DT_STRFTIME_STR == RAMS_DT_FORMAT
    assert isinstance(RAMS_FILENAME_DATETIME_REGEX, str)
    assert isinstance(RAMS_ANALYSIS_FILE_DIMENSIONS_DICT, dict)
    assert isinstance(DEFAULT_BSR_VARIABLES, list)
    assert isinstance(HYDROMETEOR_SPECIES_FULL_NAMES, dict)
    assert isinstance(SOUNDING_NAMELIST_VARIABLES, list)
    assert isinstance(RAMS_VARIABLES_DF, pd.DataFrame)
    assert len(RAMS_VARIABLES_DF) > 0
    assert "name" in RAMS_VARIABLES_DF.columns
    assert "units" in RAMS_VARIABLES_DF.columns
    assert ureg is not None


def test_dask_imports():
    """Dask utilities are importable."""
    from xarrams import dask_diagnostics, reload_intermediate

    assert callable(dask_diagnostics)
    assert callable(reload_intermediate)


def test_all_exports_match():
    """Every name in __all__ is actually importable."""
    for name in xarrams.__all__:
        assert hasattr(xarrams, name), f"{name} listed in __all__ but not importable"
