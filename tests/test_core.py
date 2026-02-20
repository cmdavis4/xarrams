"""Test xarrams core functionality."""

import pytest
from datetime import datetime
from pathlib import Path


def test_get_datetime():
    """Test parsing datetime from RAMS filename."""
    from xarrams import get_datetime

    filename = "a-A-2020-01-01-120000-g1.h5"
    dt = get_datetime(filename)

    assert isinstance(dt, datetime)
    assert dt.year == 2020
    assert dt.month == 1
    assert dt.day == 1
    assert dt.hour == 12


def test_get_grid_number():
    """Test parsing grid number from RAMS filename."""
    from xarrams import get_grid_number

    assert get_grid_number("a-A-2020-01-01-120000-g1.h5") == 1
    assert get_grid_number("a-A-2020-01-01-120000-g3.h5") == 3


def test_to_rams_output_filename():
    """Test generating RAMS output filename."""
    from xarrams import to_rams_output_filename
    from datetime import datetime

    dt = datetime(2020, 1, 1, 12, 0, 0)
    filename = to_rams_output_filename(dt, lite=False, grid=1)

    assert filename == "a-A-2020-01-01-120000-g1.h5"


def test_rams_dt_format():
    """Test RAMS datetime format constant."""
    from xarrams import RAMS_DT_FORMAT, RAMS_DT_STRFTIME_STR

    assert RAMS_DT_FORMAT == "%Y-%m-%d-%H%M%S"
    assert RAMS_DT_STRFTIME_STR == RAMS_DT_FORMAT


def test_ramsin_variables_dataframe():
    """Test that RAMS variables dataframe loads correctly."""
    from xarrams import RAMS_VARIABLES_DF
    import pandas as pd

    assert isinstance(RAMS_VARIABLES_DF, pd.DataFrame)
    assert 'name' in RAMS_VARIABLES_DF.columns
    assert 'units' in RAMS_VARIABLES_DF.columns
    assert len(RAMS_VARIABLES_DF) > 0


def test_get_z_levels():
    """Test vertical level generation."""
    from xarrams import get_z_levels
    import numpy as np

    levels = get_z_levels(deltaz=100, dzrat=1.1, dzmax=1000, nnzp=50)

    assert isinstance(levels, np.ndarray)
    assert len(levels) == 50
    assert levels[0] < 0  # Sub-ground level
    assert levels[-1] > levels[0]  # Increasing
