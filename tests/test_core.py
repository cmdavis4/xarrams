"""Tests for core xarrams functionality (IO, utils, soundings, etc.)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# IO: filename parsing
# ---------------------------------------------------------------------------


class TestGetDatetime:
    def test_basic(self):
        from xarrams import get_datetime

        dt = get_datetime("a-A-2020-01-01-120000-g1.h5")
        assert isinstance(dt, datetime)
        assert dt == datetime(2020, 1, 1, 12, 0, 0)

    def test_lite_file(self):
        from xarrams import get_datetime

        dt = get_datetime("a-L-2023-06-15-183000-g2.h5")
        assert dt == datetime(2023, 6, 15, 18, 30, 0)

    def test_invalid_raises(self):
        from xarrams import get_datetime

        with pytest.raises(ValueError, match="Unable to parse datetime"):
            get_datetime("invalid_filename.h5")


class TestGetGridNumber:
    def test_grid_1(self):
        from xarrams import get_grid_number

        assert get_grid_number("a-A-2020-01-01-120000-g1.h5") == 1

    def test_grid_3(self):
        from xarrams import get_grid_number

        assert get_grid_number("a-A-2020-01-01-120000-g3.h5") == 3

    def test_invalid_raises(self):
        from xarrams import get_grid_number

        with pytest.raises(ValueError, match="Unable to parse grid number"):
            get_grid_number("no-grid-here.txt")


class TestToRamsOutputFilename:
    def test_analysis(self):
        from xarrams import to_rams_output_filename

        dt = datetime(2020, 1, 1, 12, 0, 0)
        assert to_rams_output_filename(dt, lite=False, grid=1) == "a-A-2020-01-01-120000-g1.h5"

    def test_lite(self):
        from xarrams import to_rams_output_filename

        dt = datetime(2020, 1, 1, 12, 0, 0)
        assert to_rams_output_filename(dt, lite=True, grid=2) == "a-L-2020-01-01-120000-g2.h5"


class TestToHeaderFilepath:
    def test_basic(self):
        from xarrams import to_header_filepath

        result = to_header_filepath("/some/path/a-A-2020-01-01-120000-g1.h5")
        assert result == Path("/some/path/a-A-2020-01-01-120000-head.txt")

    def test_grid_2(self):
        from xarrams import to_header_filepath

        result = to_header_filepath("a-A-2020-01-01-120000-g2.h5")
        assert result.name == "a-A-2020-01-01-120000-head.txt"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_rams_dt_format(self):
        from xarrams import RAMS_DT_FORMAT

        assert RAMS_DT_FORMAT == "%Y-%m-%d-%H%M%S"
        # Verify it works as a format string
        dt = datetime(2020, 1, 1, 12, 0, 0)
        assert dt.strftime(RAMS_DT_FORMAT) == "2020-01-01-120000"

    def test_hydrometeor_species(self):
        from xarrams import HYDROMETEOR_SPECIES_FULL_NAMES

        assert "CP" in HYDROMETEOR_SPECIES_FULL_NAMES
        assert HYDROMETEOR_SPECIES_FULL_NAMES["CP"] == "cloud"
        assert len(HYDROMETEOR_SPECIES_FULL_NAMES) == 8

    def test_sounding_variables(self):
        from xarrams import SOUNDING_NAMELIST_VARIABLES

        assert SOUNDING_NAMELIST_VARIABLES == ["PS", "TS", "RTS", "US", "VS"]

    def test_variables_df_no_hash_units(self):
        from xarrams import RAMS_VARIABLES_DF

        assert "#" not in RAMS_VARIABLES_DF["units"].values


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


class TestGetZLevels:
    def test_with_nnzp(self):
        from xarrams import get_z_levels

        levels = get_z_levels(deltaz=100, dzrat=1.1, dzmax=1000, nnzp=50)
        assert isinstance(levels, np.ndarray)
        assert len(levels) == 51  # nnzp + 1 (includes sub-ground level)
        assert levels[0] < 0  # sub-ground level
        assert all(levels[i] < levels[i + 1] for i in range(len(levels) - 1))  # monotonically increasing

    def test_with_max_height(self):
        from xarrams import get_z_levels

        levels = get_z_levels(deltaz=100, dzrat=1.0, dzmax=100, max_height=5000)
        assert levels[-1] >= 5000

    def test_requires_nnzp_or_max_height(self):
        from xarrams import get_z_levels

        with pytest.raises(ValueError, match="Must pass one of"):
            get_z_levels(deltaz=100, dzrat=1.1, dzmax=1000)

    def test_first_level_negative(self):
        from xarrams import get_z_levels

        levels = get_z_levels(deltaz=200, dzrat=1.0, dzmax=200, nnzp=10)
        assert levels[0] == -100.0
        assert levels[1] == 100.0


class TestRamsinStr:
    def test_basic(self):
        from xarrams import ramsin_str

        assert ramsin_str("hello") == "'hello'"
        assert ramsin_str(42) == "'42'"


class TestTimeFunctions:
    def test_to_t_minutes_numpy(self):
        from xarrams import to_t_minutes

        start = np.datetime64("2020-01-01T00:00:00")
        times = np.array([
            np.datetime64("2020-01-01T00:30:00"),
            np.datetime64("2020-01-01T01:00:00"),
        ])
        result = to_t_minutes(times, start)
        np.testing.assert_array_equal(result, [30, 60])

    def test_to_t_minutes_list(self):
        from xarrams import to_t_minutes

        start = np.datetime64("2020-01-01T00:00:00")
        times = [
            np.datetime64("2020-01-01T00:15:00"),
            np.datetime64("2020-01-01T00:45:00"),
        ]
        result = to_t_minutes(times, start)
        assert result == [15, 45]


# ---------------------------------------------------------------------------
# Soundings
# ---------------------------------------------------------------------------


class TestFormatSoundingField:
    def test_basic(self):
        from xarrams import format_sounding_field_ramsin_str

        result = format_sounding_field_ramsin_str([1.0, 2.0, 3.0])
        assert "1.0000" in result
        assert "2.0000" in result
        assert "3.0000" in result


class TestWriteRamsFormattedSounding:
    def test_valid_sounding(self, tmp_path):
        from xarrams import write_rams_formatted_sounding

        df = pd.DataFrame({
            "PS": [1000.0, 900.0, 800.0, 700.0],
            "TS": [25.0, 20.0, 15.0, 10.0],
            "RTS": [80.0, 70.0, 60.0, 50.0],
            "US": [5.0, 10.0, 15.0, 20.0],
            "VS": [0.0, 2.0, 4.0, 6.0],
        })
        out = tmp_path / "sounding.csv"
        write_rams_formatted_sounding(df, out)
        assert out.exists()
        content = out.read_text()
        assert "1000.0000" in content

    def test_second_copy(self, tmp_path):
        from xarrams import write_rams_formatted_sounding

        df = pd.DataFrame({
            "PS": [1000.0, 900.0],
            "TS": [25.0, 20.0],
            "RTS": [80.0, 70.0],
            "US": [5.0, 10.0],
            "VS": [0.0, 2.0],
        })
        out1 = tmp_path / "sounding1.csv"
        out2 = tmp_path / "sounding2.csv"
        write_rams_formatted_sounding(df, out1, second_copy=out2)
        assert out1.exists()
        assert out2.exists()

    def test_missing_columns_raises(self, tmp_path):
        from xarrams import write_rams_formatted_sounding

        df = pd.DataFrame({"PS": [1000.0], "TS": [25.0]})
        with pytest.raises(ValueError, match="must contain columns"):
            write_rams_formatted_sounding(df, tmp_path / "out.csv")

    def test_non_monotonic_pressure_raises(self, tmp_path):
        from xarrams import write_rams_formatted_sounding

        df = pd.DataFrame({
            "PS": [800.0, 900.0],  # increasing, not decreasing
            "TS": [25.0, 20.0],
            "RTS": [80.0, 70.0],
            "US": [5.0, 10.0],
            "VS": [0.0, 2.0],
        })
        with pytest.raises(ValueError, match="monotonically decreasing"):
            write_rams_formatted_sounding(df, tmp_path / "out.csv")


class TestWithUpdatedSoundingFields:
    def test_updates_params(self):
        from xarrams import with_updated_sounding_fields

        sounding = pd.DataFrame({
            "PS": [1000.0, 900.0],
            "TS": [25.0, 20.0],
            "RTS": [80.0, 70.0],
            "US": [5.0, 10.0],
            "VS": [0.0, 2.0],
        })
        params = {"TIMMAX": "3600."}
        result = with_updated_sounding_fields(params, sounding)

        # Original should not be mutated
        assert "PS" not in params
        # Result should have sounding fields
        assert "PS" in result
        assert "TIMMAX" in result
        # Flags should be set
        assert result["IPSFLG"] == "0"

    def test_no_flag_update(self):
        from xarrams import with_updated_sounding_fields

        sounding = pd.DataFrame({
            "PS": [1000.0, 900.0],
            "TS": [25.0, 20.0],
            "RTS": [80.0, 70.0],
            "US": [5.0, 10.0],
            "VS": [0.0, 2.0],
        })
        result = with_updated_sounding_fields({}, sounding, update_sounding_field_flags=False)
        assert "IPSFLG" not in result


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestGenerateRamsin:
    def test_basic_generation(self, tmp_path):
        from xarrams import generate_ramsin

        template = tmp_path / "RAMSIN.template"
        template.write_text(
            "$MODEL_GRIDS\n"
            " TIMMAX = 3600.,\n"
            " NNXP = 100,\n"
            "$END\n"
        )
        result = generate_ramsin(
            ramsin_name="test",
            parameters={"TIMMAX": "7200."},
            rams_input_dir=None,
            rams_output_dir=None,
            ramsin_dir=tmp_path,
            ramsin_template_path=template,
        )
        assert "7200." in result
        assert (tmp_path / "RAMSIN.test").exists()

    def test_missing_field_raises(self, tmp_path):
        from xarrams import generate_ramsin

        template = tmp_path / "RAMSIN.template"
        template.write_text("$MODEL_GRIDS\n TIMMAX = 3600.,\n$END\n")
        with pytest.raises(ValueError, match="not found in template"):
            generate_ramsin(
                ramsin_name="test",
                parameters={"NONEXISTENT_FIELD": "42"},
                rams_input_dir=None,
                rams_output_dir=None,
                ramsin_dir=tmp_path,
                ramsin_template_path=template,
            )


class TestRunRamsForRamsin:
    def test_dry_run(self, tmp_path):
        from xarrams import run_rams_for_ramsin

        ramsin = tmp_path / "RAMSIN.test"
        ramsin.write_text("test")
        stdout = tmp_path / "test.stdout"
        exe = tmp_path / "rams"
        exe.write_text("#!/bin/bash\necho hello")

        result = run_rams_for_ramsin(
            ramsin_path=ramsin,
            stdout_path=stdout,
            rams_executable_path=exe,
            dry_run=True,
            log_command=False,
            log_ramsin=False,
        )
        assert result is True

    def test_long_path_raises(self, tmp_path):
        from xarrams import run_rams_for_ramsin

        long_path = tmp_path / ("x" * 300)
        with pytest.raises(ValueError, match="256 characters"):
            run_rams_for_ramsin(
                ramsin_path=long_path,
                stdout_path=tmp_path / "out",
                rams_executable_path=tmp_path / "exe",
            )


# ---------------------------------------------------------------------------
# Unit registry
# ---------------------------------------------------------------------------


class TestUnitRegistry:
    def test_custom_units_loaded(self):
        from xarrams import ureg

        # These custom units should be defined in rams_pint_units.txt
        q = ureg.Quantity(1, "fraction")
        assert q.magnitude == 1
