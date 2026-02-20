"""Test xarrams imports."""

def test_xarrams_imports():
    """Test basic package imports."""
    import xarrams
    assert xarrams.__version__ == "0.1.0"


def test_function_imports():
    """Test importing key functions."""
    from xarrams import (
        read_rams_output,
        calculate_thermodynamic_variables,
        calculate_derived_variables,
        plot_sounding,
        get_datetime,
        get_grid_number,
        to_rams_output_filename,
    )
    assert callable(read_rams_output)
    assert callable(calculate_thermodynamic_variables)
    assert callable(plot_sounding)
    assert callable(get_datetime)
    assert callable(get_grid_number)
    assert callable(to_rams_output_filename)


def test_constants_imports():
    """Test importing constants."""
    from xarrams import (
        RAMS_DT_FORMAT,
        RAMS_DT_STRFTIME_STR,
        RAMS_VARIABLES_DF,
    )
    import pandas as pd
    assert isinstance(RAMS_DT_FORMAT, str)
    assert isinstance(RAMS_DT_STRFTIME_STR, str)
    assert isinstance(RAMS_VARIABLES_DF, pd.DataFrame)
