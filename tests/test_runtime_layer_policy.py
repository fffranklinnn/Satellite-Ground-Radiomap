from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scripts.generate_multisat_timeseries_radiomap as multisat_module

import main as main_module
from src.context.frame_context import FrameContext
from src.context.grid_spec import GridSpec
from src.pipeline.benchmark_runner import BenchmarkRunner
from src.planning import MissingRequiredInputError, resolve_layer_policy
from src.planning import enabled_layer_config
from src.products.manifest import collect_input_file_paths


ORIGIN_LAT = 34.0
ORIGIN_LON = 108.0
GRID = GridSpec.from_legacy_args(ORIGIN_LAT, ORIGIN_LON, 256.0, 8, 8)
TS = datetime(2025, 1, 3, 0, 0, 0, tzinfo=timezone.utc)
FRAME = FrameContext(
    frame_id="frame_000",
    timestamp=TS,
    grid=GRID,
    norad_id="25544",
    sat_elevation_deg=45.0,
    sat_azimuth_deg=180.0,
    sat_slant_range_m=600_000.0,
)


def _mock_layers():
    l1 = MagicMock()
    l1.propagate_entry.return_value = MagicMock(frame_id=FRAME.frame_id)
    l1.fallbacks_used = []
    l1.clear_fallbacks = MagicMock()

    l2 = MagicMock()
    l2.propagate_terrain.return_value = MagicMock(frame_id=FRAME.frame_id)

    l3 = MagicMock()
    l3.refine_urban.return_value = MagicMock(frame_id=FRAME.frame_id)

    return l1, l2, l3


def test_resolver_decorates_manifest_metadata():
    config = {"scene": {"profile": "urban_flat"}}
    policy = resolve_layer_policy(config)
    assert policy.scene_profile == "urban_flat"
    assert policy.enabled_layers == ("l1_macro", "l3_urban")


def test_benchmark_runner_uses_policy_and_records_metadata(tmp_path):
    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": True},
        },
        "data_validation": {"strict": False},
    }
    l1, l2, l3 = _mock_layers()
    runner = BenchmarkRunner(
        frame_builder=MagicMock(build=MagicMock(return_value=FRAME)),
        l1_layer=l1,
        l2_layer=l2,
        l3_layer=l3,
        config=config,
        data_snapshot_id="snap_001",
    )

    with patch("src.pipeline.benchmark_runner.export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch("src.pipeline.benchmark_runner.collect_input_file_paths", return_value={}), \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        manifest = runner.run_frame(TS, tmp_path, ["path_loss_map"])

    assert l2.propagate_terrain.call_count == 0
    assert l3.refine_urban.call_count == 1
    assert manifest.metadata["scene_profile"] == "urban_flat"
    assert manifest.metadata["enabled_layers"] == ("l1_macro", "l3_urban")
    assert manifest.metadata["disabled_layers"]["l2_topo"]["reason_type"] == "user_override"


def test_main_run_simulation_uses_policy_and_records_metadata(tmp_path):
    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": True},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "logging": {"level": "INFO"},
        "output": {"save_individual_layers": False, "save_composite": True, "dpi": 72},
        "performance": {"enable_profiling": False},
        "data_validation": {"strict": False, "snapshot_id": "snap_001"},
    }
    l1, l2, l3 = _mock_layers()

    fake_logger = MagicMock()
    fake_sim_logger = MagicMock()
    fake_fb = MagicMock(build=MagicMock(return_value=FRAME))
    fake_manifest = MagicMock(metadata={})

    with patch.object(main_module, "setup_logger", return_value=fake_logger), \
         patch.object(main_module, "SimulationLogger", return_value=fake_sim_logger), \
         patch.object(main_module, "initialize_layers", return_value=(l1, l2, l3)), \
         patch.object(main_module, "build_frame_builder", return_value=fake_fb), \
         patch.object(main_module, "collect_input_file_paths", return_value={}), \
         patch.object(main_module, "export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch.object(main_module, "plot_radio_map"), \
         patch.object(main_module, "plot_layer_comparison"), \
         patch.object(main_module, "ProductManifest") as manifest_cls, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        manifest_cls.build.return_value = fake_manifest
        main_module.run_simulation(config, tmp_path)

    assert l2.propagate_terrain.call_count == 0
    assert l3.refine_urban.call_count == 1
    manifest_cls.build.assert_called()
    _, kwargs = manifest_cls.build.call_args
    assert kwargs["metadata"]["scene_profile"] == "urban_flat"
    assert kwargs["metadata"]["enabled_layers"] == ["l1_macro", "l3_urban"]


def test_main_run_simulation_marks_missing_input_without_failing(tmp_path):
    missing_tiles = tmp_path / "missing-tiles"
    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": True, "tile_cache_root": str(missing_tiles)},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "logging": {"level": "INFO"},
        "output": {"save_individual_layers": False, "save_composite": True, "dpi": 72},
        "performance": {"enable_profiling": False},
        "data_validation": {"strict": False, "snapshot_id": "snap_001"},
    }
    l1, l2, l3 = _mock_layers()

    fake_logger = MagicMock()
    fake_sim_logger = MagicMock()
    fake_fb = MagicMock(build=MagicMock(return_value=FRAME))
    fake_manifest = MagicMock(metadata={})

    with patch.object(main_module, "setup_logger", return_value=fake_logger), \
         patch.object(main_module, "SimulationLogger", return_value=fake_sim_logger), \
         patch.object(main_module, "initialize_layers", return_value=(l1, l2, l3)), \
         patch.object(main_module, "build_frame_builder", return_value=fake_fb), \
         patch.object(main_module, "collect_input_file_paths", return_value={}), \
         patch.object(main_module, "export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch.object(main_module, "plot_radio_map"), \
         patch.object(main_module, "plot_layer_comparison"), \
         patch.object(main_module, "ProductManifest") as manifest_cls, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        manifest_cls.build.return_value = fake_manifest
        main_module.run_simulation(config, tmp_path)

    assert l2.propagate_terrain.call_count == 0
    assert l3.refine_urban.call_count == 0
    manifest_cls.build.assert_called()
    _, kwargs = manifest_cls.build.call_args
    assert kwargs["metadata"]["scene_profile"] == "urban_flat"
    assert kwargs["metadata"]["disabled_layers"]["l3_urban"]["reason_type"] == "missing_input"


def test_main_run_simulation_skips_disabled_constructor(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")
    dem_path = project_root / "data" / "dem.tif"
    dem_path.write_text("dem", encoding="utf-8")

    config = {
        "scene": {"profile": "mountain_rural"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": True, "dem_file": "data/dem.tif"},
            "l3_urban": {"enabled": True, "tile_cache_root": "data/tiles"},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "logging": {"level": "INFO"},
        "output": {"save_individual_layers": False, "save_composite": True, "dpi": 72},
        "performance": {"enable_profiling": False},
        "data_validation": {"strict": True, "snapshot_id": "snap_001"},
    }

    l1_layer = MagicMock()
    l1_layer.propagate_entry.return_value = MagicMock(frame_id=FRAME.frame_id)
    l1_layer.fallbacks_used = []
    l1_layer.clear_fallbacks = MagicMock()
    l2_layer = MagicMock()
    l2_layer.propagate_terrain.return_value = MagicMock(frame_id=FRAME.frame_id)

    fake_logger = MagicMock()
    fake_sim_logger = MagicMock()
    fake_fb = MagicMock(build=MagicMock(return_value=FRAME))
    fake_selector = MagicMock()
    fake_selector.select.return_value = {"norad_id": "25544"}

    monkeypatch.chdir(tmp_path)
    with patch.object(main_module, "setup_logger", return_value=fake_logger), \
         patch.object(main_module, "SimulationLogger", return_value=fake_sim_logger), \
         patch.object(main_module, "L1MacroLayer", return_value=l1_layer) as l1_ctor, \
         patch.object(main_module, "L2TopoLayer", return_value=l2_layer) as l2_ctor, \
         patch.object(main_module, "L3UrbanLayer") as l3_ctor, \
         patch.object(main_module, "build_frame_builder", return_value=fake_fb), \
         patch.object(main_module, "collect_input_file_paths", return_value={}), \
         patch.object(main_module, "export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch.object(main_module, "plot_radio_map"), \
         patch.object(main_module, "plot_layer_comparison"), \
         patch("src.planning.satellite_selector.SatelliteSelector", return_value=fake_selector), \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        main_module.run_simulation(config, tmp_path, project_root=project_root)

    assert l1_ctor.called
    assert l2_ctor.called
    assert l3_ctor.call_count == 0
    assert l2_layer.propagate_terrain.call_count == 1
    assert l1_layer.propagate_entry.call_count == 1


def test_main_run_simulation_hashes_original_config(tmp_path):
    project_root = tmp_path / "repo"
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")

    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "logging": {"level": "INFO"},
        "output": {"save_individual_layers": False, "save_composite": True, "dpi": 72},
        "performance": {"enable_profiling": False},
        "data_validation": {"strict": False, "snapshot_id": "snap_001"},
    }

    l1, l2, l3 = _mock_layers()
    fake_logger = MagicMock()
    fake_sim_logger = MagicMock()
    fake_fb = MagicMock(build=MagicMock(return_value=FRAME))
    fake_manifest = MagicMock(metadata={})

    with patch.object(main_module, "setup_logger", return_value=fake_logger), \
         patch.object(main_module, "SimulationLogger", return_value=fake_sim_logger), \
         patch.object(main_module, "initialize_layers", return_value=(l1, l2, l3)), \
         patch.object(main_module, "build_frame_builder", return_value=fake_fb), \
         patch.object(main_module, "collect_input_file_paths", return_value={}), \
         patch.object(main_module, "export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch.object(main_module, "plot_radio_map"), \
         patch.object(main_module, "plot_layer_comparison"), \
         patch.object(main_module, "ProductManifest") as manifest_cls, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        manifest_cls.build.return_value = fake_manifest
        main_module.run_simulation(config, tmp_path, project_root=project_root)

    _, kwargs = manifest_cls.build.call_args
    assert kwargs["config"] == config
    assert kwargs["config"]["layers"]["l1_macro"]["tle_file"] == "data/starlink-2025-tle/2025-01-01.tle"


def test_main_run_simulation_preserves_l1_geometry_when_l1_disabled(tmp_path):
    config = {
        "scene": {"profile": "plain_sparse"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {
                "enabled": True,
                "coverage_km": 512.0,
                "grid_size": 128,
            },
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "logging": {"level": "INFO"},
        "output": {"save_individual_layers": False, "save_composite": True, "dpi": 72},
        "performance": {"enable_profiling": False},
        "data_validation": {"strict": False, "snapshot_id": "snap_001"},
    }
    l1, l2, l3 = _mock_layers()
    fake_logger = MagicMock()
    fake_sim_logger = MagicMock()
    fake_fb = MagicMock(build=MagicMock(return_value=FRAME))
    fake_manifest = MagicMock(metadata={})

    with patch.object(main_module, "setup_logger", return_value=fake_logger), \
         patch.object(main_module, "SimulationLogger", return_value=fake_sim_logger), \
         patch.object(main_module, "initialize_layers", return_value=(l1, l2, l3)), \
         patch.object(main_module, "build_frame_builder", return_value=fake_fb) as build_fb, \
         patch.object(main_module, "collect_input_file_paths", return_value={}), \
         patch.object(main_module, "export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch.object(main_module, "plot_radio_map"), \
         patch.object(main_module, "plot_layer_comparison"), \
         patch.object(main_module, "ProductManifest") as manifest_cls, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)):
        manifest_cls.build.return_value = fake_manifest
        main_module.run_simulation(config, tmp_path)

    build_config = build_fb.call_args.args[0]
    assert build_config["layers"]["l1_macro"]["coverage_km"] == 512.0
    assert build_config["layers"]["l1_macro"]["grid_size"] == 128
    assert build_config["layers"]["l2_topo"]["enabled"] is False
    assert build_config["layers"]["l3_urban"]["enabled"] is False


def test_main_cli_preflight_uses_policy_filtered_config(tmp_path):
    config = {
        "scene": {"profile": "mountain_rural"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": str(tmp_path / "tle.tle")},
            "l2_topo": {"enabled": True, "dem_file": str(tmp_path / "dem.tif")},
            "l3_urban": {"enabled": True, "tile_cache_root": str(tmp_path / "missing-tiles")},
        },
        "output": {"directory": str(tmp_path / "out")},
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "data_validation": {"strict": False},
    }
    fake_args = MagicMock(
        config="unused.yaml",
        output=None,
        strict_data=False,
        check_data_only=True,
    )
    captured = {}

    with patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(main_module, "load_config", return_value=config), \
         patch.object(main_module, "validate_data_integrity", side_effect=lambda **kwargs: captured.update(kwargs) or {"errors": [], "warnings": [], "checks": [], "strict": kwargs["strict"]}), \
         patch.object(main_module, "format_data_validation_report", return_value="ok"), \
         patch.object(main_module, "run_simulation") as run_sim:
        with pytest.raises(SystemExit) as exc:
            main_module.main()

    assert exc.value.code == 0
    assert run_sim.call_count == 0
    assert captured["config"]["layers"]["l3_urban"]["enabled"] is False


def test_main_cli_preflight_keeps_missing_input_enabled_under_strict_data(tmp_path):
    tle_path = tmp_path / "tle.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")
    config = {
        "scene": {"profile": "mountain_rural"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": str(tle_path)},
            "l2_topo": {"enabled": True, "dem_file": str(tmp_path / "missing-dem.tif")},
            "l3_urban": {"enabled": True, "tile_cache_root": str(tmp_path / "missing-tiles")},
        },
        "output": {"directory": str(tmp_path / "out")},
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "data_validation": {"strict": False},
    }
    fake_args = MagicMock(
        config="unused.yaml",
        output=None,
        strict_data=True,
        check_data_only=True,
    )

    with patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(main_module, "load_config", return_value=config), \
         patch.object(main_module, "validate_data_integrity") as validate_data, \
         patch.object(main_module, "run_simulation") as run_sim:
        with pytest.raises(MissingRequiredInputError):
            main_module.main()

    assert run_sim.call_count == 0
    assert validate_data.call_count == 0


def test_main_cli_preflight_mirrors_missing_input_policy_disable(tmp_path):
    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": str(tmp_path / "tle.tle")},
            "l2_topo": {"enabled": True, "dem_file": str(tmp_path / "dem.tif")},
            "l3_urban": {"enabled": True, "tile_cache_root": str(tmp_path / "missing-tiles")},
        },
        "output": {"directory": str(tmp_path / "out")},
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
        "data_validation": {"strict": False},
    }
    fake_args = MagicMock(
        config="unused.yaml",
        output=None,
        strict_data=False,
        check_data_only=True,
    )
    captured = {}

    def _validate(**kwargs):
        captured.update(kwargs)
        return {"errors": [], "warnings": [], "checks": [], "strict": kwargs["strict"]}

    with patch.object(main_module.argparse.ArgumentParser, "parse_args", return_value=fake_args), \
         patch.object(main_module, "load_config", return_value=config), \
         patch.object(main_module, "validate_data_integrity", side_effect=_validate), \
         patch.object(main_module, "format_data_validation_report", return_value="ok"), \
         patch.object(main_module, "run_simulation") as run_sim:
        with pytest.raises(SystemExit) as exc:
            main_module.main()

    assert exc.value.code == 0
    assert run_sim.call_count == 0
    assert captured["strict"] is False
    assert captured["config"]["layers"]["l3_urban"]["enabled"] is False


def test_benchmark_runner_strict_paths_use_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")
    dem_path = project_root / "data" / "dem.tif"
    dem_path.write_text("dem", encoding="utf-8")

    config = {
        "scene": {"profile": "mountain_rural"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": True, "dem_file": "data/dem.tif"},
            "l3_urban": {"enabled": True, "tile_cache_root": "data/tiles"},
        },
        "data_validation": {"strict": True},
    }

    l1, l2, l3 = _mock_layers()
    fake_selector = MagicMock()
    fake_selector.select.return_value = {"norad_id": "25544"}
    runner = BenchmarkRunner(
        frame_builder=MagicMock(build=MagicMock(return_value=FRAME)),
        l1_layer=l1,
        l2_layer=l2,
        l3_layer=l3,
        config=config,
        data_snapshot_id="snap_001",
        project_root=project_root,
    )

    monkeypatch.chdir(tmp_path)
    with patch("src.pipeline.benchmark_runner.export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch("src.pipeline.benchmark_runner.collect_input_file_paths", wraps=collect_input_file_paths) as input_paths, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)), \
         patch("src.planning.satellite_selector.SatelliteSelector", return_value=fake_selector):
        manifest = runner.run_frame(TS, tmp_path, ["path_loss_map"])

    assert input_paths.called
    assert manifest.input_file_hashes["tle_file"]
    assert manifest.input_file_hashes["dem_file"]


def test_benchmark_runner_hashes_original_config(tmp_path):
    project_root = tmp_path / "repo"
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")

    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        },
        "data_validation": {"strict": False},
    }

    l1, l2, l3 = _mock_layers()
    runner = BenchmarkRunner(
        frame_builder=MagicMock(build=MagicMock(return_value=FRAME)),
        l1_layer=l1,
        l2_layer=l2,
        l3_layer=l3,
        config=config,
        data_snapshot_id="snap_001",
        project_root=project_root,
    )

    fake_manifest = MagicMock(metadata={})
    with patch("src.pipeline.benchmark_runner.export_dataset", return_value=({"path_loss_map": str(tmp_path / "out.npy")}, None)), \
         patch("src.pipeline.benchmark_runner.collect_input_file_paths", return_value={}), \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.zeros((8, 8), dtype=np.float32), l1_db=None, l2_db=None, l3_db=None)), \
         patch("src.pipeline.benchmark_runner.ProductManifest") as manifest_cls:
        manifest_cls.build.return_value = fake_manifest
        runner.run_frame(TS, tmp_path, ["path_loss_map"])

    _, kwargs = manifest_cls.build.call_args
    assert kwargs["config"] == config
    assert kwargs["config"]["layers"]["l1_macro"]["tle_file"] == "data/starlink-2025-tle/2025-01-01.tle"


def test_benchmark_runner_strict_missing_input_raises(tmp_path):
    missing_tiles = tmp_path / "missing-tiles"
    config = {
        "scene": {"profile": "urban_flat"},
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": True, "tile_cache_root": str(missing_tiles)},
        },
        "data_validation": {"strict": True},
    }
    l1, l2, l3 = _mock_layers()
    runner = BenchmarkRunner(
        frame_builder=MagicMock(build=MagicMock(return_value=FRAME)),
        l1_layer=l1,
        l2_layer=l2,
        l3_layer=l3,
        config=config,
        data_snapshot_id="snap_001",
    )

    with pytest.raises(MissingRequiredInputError):
        runner.run_frame(TS, tmp_path, ["path_loss_map"])


def test_multisat_compute_satellite_maps_respects_policy_disabled_layers():
    l1 = MagicMock()
    entry = MagicMock(
        norad_id="25544",
        sat_lat_deg=34.0,
        sat_lon_deg=108.0,
        sat_alt_m=550000.0,
        azimuth_deg=np.full((8, 8), 180.0, dtype=np.float32),
        elevation_deg=np.full((8, 8), 45.0, dtype=np.float32),
        slant_range_m=np.full((8, 8), 600_000.0, dtype=np.float32),
        total_loss_db=np.full((8, 8), 150.0, dtype=np.float32),
        native_grid=GRID,
    )
    l1.propagate_entry.return_value = entry
    l1.config = {"tle_file": "test.tle"}
    l2 = MagicMock()
    l3 = MagicMock()

    frame_builder = MagicMock(grid=GRID, build=MagicMock(return_value=FRAME))
    with patch("src.planning.satellite_selector.SatelliteSelector") as selector_cls, \
         patch("src.compose.project_to_product_grid", return_value={}), \
         patch("src.context.multiscale_map.MultiScaleMap.compose", return_value=MagicMock(composite_db=np.full((8, 8), 150.0, dtype=np.float32))):
        selector_cls.return_value.select.return_value = {"norad_id": "25544"}
        l1_map, l2_map, l3_map, total_map, sat_info, frame = multisat_module.compute_satellite_maps(
            l1_layer=l1,
            l2_layer=l2,
            l3_layer=l3,
            frame_builder=frame_builder,
            timestamp=TS,
            norad_id="25544",
            enable_l2=False,
            enable_l3=False,
        )

    assert l2.propagate_terrain.call_count == 0
    assert l3.refine_urban.call_count == 0
    np.testing.assert_array_equal(l1_map, entry.total_loss_db)
    np.testing.assert_array_equal(l2_map, np.zeros((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(l3_map, np.zeros((8, 8), dtype=np.float32))
    np.testing.assert_array_equal(total_map, np.full((8, 8), 150.0, dtype=np.float32))
    assert sat_info["norad_id"] == "25544"
    assert frame is FRAME


def test_multisat_check_required_data_ignores_scene_disabled_inputs(tmp_path):
    tle_path = tmp_path / "test.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")
    config = {
        "scene": {"profile": "urban_flat"},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": str(tle_path)},
            "l2_topo": {"enabled": True, "dem_file": str(tmp_path / "missing-dem.tif")},
            "l3_urban": {"enabled": True, "tile_cache_root": str(tmp_path)},
        },
    }

    multisat_module.check_required_data(
        tmp_path,
        config,
        allow_missing=False,
        strict=False,
        benchmark=False,
    )


def test_multisat_skips_l1_construction_when_policy_disables_it(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    config = {
        "scene": {"profile": "plain_sparse"},
        "origin": {"latitude": 34.0, "longitude": 108.0},
        "layers": {
            "l1_macro": {"enabled": False, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": False, "dem_file": "data/dem.tif"},
            "l3_urban": {"enabled": False},
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
    }

    fake_args = MagicMock(
        config="unused.yaml",
        start=None,
        end=None,
        step_minutes=None,
        max_frames=None,
        origin_lat=None,
        origin_lon=None,
        fusion_mode="best-server",
        max_satellites=8,
        min_elevation_deg=None,
        soft_min_power=1e-30,
        norad_id=None,
        output_dir=str(project_root / "output"),
        output_prefix="multisat_ts",
        dpi=180,
        cmap="viridis",
        allow_missing_data=True,
    )

    fake_fb = MagicMock(grid=GRID, build=MagicMock(return_value=FRAME))
    fake_writer = MagicMock()
    fake_writer.__enter__ = MagicMock(return_value=fake_writer)
    fake_writer.__exit__ = MagicMock(return_value=False)

    (project_root / "output" / "png").mkdir(parents=True)
    (project_root / "output" / "npy").mkdir(parents=True)
    (project_root / "output" / "frame_json").mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    with patch.object(multisat_module, "parse_args", return_value=fake_args), \
         patch.object(multisat_module, "load_config", return_value=config), \
         patch.object(multisat_module, "L1MacroLayer") as l1_ctor, \
         patch.object(multisat_module, "L2TopoLayer", return_value=MagicMock()) as l2_ctor, \
         patch.object(multisat_module, "L3UrbanLayer", return_value=MagicMock()) as l3_ctor, \
         patch.object(multisat_module, "build_frame_builder_for_script", return_value=fake_fb), \
         patch.object(multisat_module, "make_output_dirs", return_value={
             "root": project_root / "output",
             "png": project_root / "output" / "png",
             "npy": project_root / "output" / "npy",
             "json": project_root / "output" / "frame_json",
             "manifest": project_root / "output" / "manifest.jsonl",
         }), \
         patch.object(multisat_module, "ManifestWriter", return_value=fake_writer), \
         patch.object(multisat_module, "plot_radio_map"), \
         patch.object(multisat_module, "export_dataset", return_value=({"path_loss_map": str(project_root / "output" / "npy" / "frame.npy")}, None)):
        multisat_module.main()

    assert l1_ctor.call_count == 0
    assert l2_ctor.call_count == 0
    assert l3_ctor.call_count == 0


def test_multisat_allows_missing_scene_profile_in_normal_mode(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    tle_file = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    dem_file = project_root / "data" / "dem.tif"
    tiles_dir = project_root / "data" / "tiles"
    config = {
        "origin": {"latitude": 34.0, "longitude": 108.0},
        "layers": {
            "l1_macro": {
                "enabled": True,
                "tle_file": str(tle_file),
            },
            "l2_topo": {
                "enabled": True,
                "dem_file": str(dem_file),
            },
            "l3_urban": {
                "enabled": True,
                "tile_cache_root": str(tiles_dir),
            },
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
    }

    tle_file.parent.mkdir(parents=True)
    tiles_dir.mkdir(parents=True)
    tle_file.write_text("dummy tle", encoding="utf-8")
    dem_file.write_text("dem", encoding="utf-8")
    (project_root / "output" / "png").mkdir(parents=True)
    (project_root / "output" / "npy").mkdir(parents=True)
    (project_root / "output" / "frame_json").mkdir(parents=True)

    fake_args = MagicMock(
        config="unused.yaml",
        start=None,
        end=None,
        step_minutes=None,
        max_frames=None,
        origin_lat=None,
        origin_lon=None,
        fusion_mode="best-server",
        max_satellites=8,
        min_elevation_deg=None,
        soft_min_power=1e-30,
        norad_id=None,
        output_dir=str(project_root / "output"),
        output_prefix="multisat_ts",
        dpi=180,
        cmap="viridis",
        allow_missing_data=False,
    )

    fake_l1 = MagicMock()
    fake_l1.NO_COVERAGE_LOSS_DB = 300.0
    fake_l1.get_visible_satellites.return_value = []
    fake_l1.clear_fallbacks = MagicMock()
    fake_l1.fallbacks_used = []
    fake_l1.target_norad_ids = None
    fake_l2 = MagicMock()
    fake_l3 = MagicMock()
    fake_fb = MagicMock(grid=GRID, build=MagicMock(return_value=FRAME))
    fake_writer = MagicMock()
    fake_writer.__enter__ = MagicMock(return_value=fake_writer)
    fake_writer.__exit__ = MagicMock(return_value=False)

    monkeypatch.chdir(tmp_path)
    with patch.object(multisat_module, "parse_args", return_value=fake_args), \
         patch.object(multisat_module, "load_config", return_value=config), \
         patch.object(multisat_module, "L1MacroLayer", return_value=fake_l1) as l1_ctor, \
         patch.object(multisat_module, "L2TopoLayer", return_value=fake_l2) as l2_ctor, \
         patch.object(multisat_module, "L3UrbanLayer", return_value=fake_l3) as l3_ctor, \
         patch.object(multisat_module, "build_frame_builder_for_script", return_value=fake_fb), \
         patch.object(multisat_module, "make_output_dirs", return_value={
             "root": project_root / "output",
             "png": project_root / "output" / "png",
             "npy": project_root / "output" / "npy",
             "json": project_root / "output" / "frame_json",
             "manifest": project_root / "output" / "manifest.jsonl",
         }), \
         patch.object(multisat_module, "ManifestWriter", return_value=fake_writer), \
         patch.object(multisat_module, "plot_radio_map"), \
         patch.object(multisat_module, "export_dataset", return_value=({"path_loss_map": str(project_root / "output" / "npy" / "frame.npy")}, None)):
        multisat_module.main()

    assert l1_ctor.call_count == 1


def test_multisat_frame_builder_uses_policy_aligned_layer_config():
    config = {
        "origin": {"latitude": ORIGIN_LAT, "longitude": ORIGIN_LON},
        "layers": {
            "l1_macro": {"enabled": True, "coverage_km": 256.0, "grid_size": 256},
            "l2_topo": {"enabled": False, "coverage_km": 25.6, "grid_size": 256},
            "l3_urban": {"enabled": True, "coverage_km": 0.256, "grid_size": 256},
        },
    }

    frame_builder = multisat_module.build_frame_builder_for_script(config, ORIGIN_LAT, ORIGIN_LON)

    assert frame_builder.coverage.l2_grid.width_m == frame_builder.coverage.l1_grid.width_m


def test_multisat_strict_missing_input_without_scene_profile_raises(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    tle_file = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    dem_file = project_root / "data" / "dem.tif"
    tiles_dir = project_root / "data" / "tiles"
    config = {
        "origin": {"latitude": 34.0, "longitude": 108.0},
        "layers": {
            "l1_macro": {
                "enabled": True,
                "tle_file": str(tle_file),
            },
            "l2_topo": {
                "enabled": True,
                "dem_file": str(dem_file),
            },
            "l3_urban": {
                "enabled": True,
                "tile_cache_root": str(tiles_dir),
            },
        },
        "time": {
            "start": "2025-01-03T00:00:00+00:00",
            "end": "2025-01-03T00:00:00+00:00",
            "step_hours": 1,
        },
    }

    tle_file.parent.mkdir(parents=True)
    tiles_dir.mkdir(parents=True)
    tle_file.write_text("dummy tle", encoding="utf-8")
    (project_root / "output" / "png").mkdir(parents=True)
    (project_root / "output" / "npy").mkdir(parents=True)
    (project_root / "output" / "frame_json").mkdir(parents=True)

    fake_args = MagicMock(
        config="unused.yaml",
        start=None,
        end=None,
        step_minutes=None,
        max_frames=None,
        origin_lat=None,
        origin_lon=None,
        fusion_mode="best-server",
        max_satellites=8,
        min_elevation_deg=None,
        soft_min_power=1e-30,
        norad_id=None,
        output_dir=str(project_root / "output"),
        output_prefix="multisat_ts",
        dpi=180,
        cmap="viridis",
        allow_missing_data=False,
    )

    fake_fb = MagicMock(grid=GRID, build=MagicMock(return_value=FRAME))
    fake_writer = MagicMock()
    fake_writer.__enter__ = MagicMock(return_value=fake_writer)
    fake_writer.__exit__ = MagicMock(return_value=False)

    monkeypatch.chdir(tmp_path)
    with patch.object(multisat_module, "parse_args", return_value=fake_args), \
         patch.object(multisat_module, "load_config", return_value=config), \
         patch.object(multisat_module, "build_frame_builder_for_script", return_value=fake_fb), \
         patch.object(multisat_module, "make_output_dirs", return_value={
             "root": project_root / "output",
             "png": project_root / "output" / "png",
             "npy": project_root / "output" / "npy",
             "json": project_root / "output" / "frame_json",
             "manifest": project_root / "output" / "manifest.jsonl",
         }), \
         patch.object(multisat_module, "ManifestWriter", return_value=fake_writer), \
         patch.object(multisat_module, "plot_radio_map"), \
         patch.object(multisat_module, "export_dataset", return_value=({"path_loss_map": str(project_root / "output" / "npy" / "frame.npy")}, None)):
        with pytest.raises(MissingRequiredInputError):
            multisat_module.main()


def test_multisat_strict_paths_use_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")
    dem_path = project_root / "data" / "dem.tif"
    dem_path.write_text("dem", encoding="utf-8")
    tiles_dir = project_root / "data" / "tiles"
    tiles_dir.mkdir(parents=True)

    config = {
        "scene": {"profile": "urban_flat"},
        "layers": {
            "l1_macro": {"enabled": True, "tle_file": "data/starlink-2025-tle/2025-01-01.tle"},
            "l2_topo": {"enabled": True, "dem_file": "data/dem.tif"},
            "l3_urban": {"enabled": True, "tile_cache_root": "data/tiles"},
        },
    }

    monkeypatch.chdir(tmp_path)
    multisat_module.check_required_data(
        project_root,
        config,
        allow_missing=False,
        strict=True,
        benchmark=False,
    )
    normalized = multisat_module.normalize_layer_paths(project_root, config)
    paths = multisat_module.collect_input_file_paths(
        multisat_module.enabled_layer_config(normalized, ("l1_macro", "l2_topo", "l3_urban")),
        strict=True,
    )
    assert paths["tle_file"] == str(tle_path)
    assert paths["dem_file"] == str(dem_path)


def test_multisat_strict_paths_use_nested_tle_file(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "data" / "starlink-2025-tle").mkdir(parents=True)
    tle_path = project_root / "data" / "starlink-2025-tle" / "2025-01-01.tle"
    tle_path.write_text("dummy tle", encoding="utf-8")

    config = {
        "scene": {"profile": "urban_flat"},
        "layers": {
            "l1_macro": {"enabled": True, "tle": {"file": "data/starlink-2025-tle/2025-01-01.tle"}},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        },
    }

    monkeypatch.chdir(tmp_path)
    multisat_module.check_required_data(
        project_root,
        config,
        allow_missing=False,
        strict=True,
        benchmark=False,
    )
    normalized = multisat_module.normalize_layer_paths(project_root, config)
    paths = multisat_module.collect_input_file_paths(
        multisat_module.enabled_layer_config(normalized, ("l1_macro",)),
        strict=True,
    )
    assert paths["tle_file"] == str(tle_path)


def test_multisat_manifest_records_shared_policy_metadata(tmp_path):
    config = {
        "scene": {"profile": "mountain_rural"},
        "layers": {
            "l1_macro": {"enabled": True},
            "l2_topo": {"enabled": True},
            "l3_urban": {"enabled": True},
        },
    }

    policy = resolve_layer_policy(config)
    manifest = multisat_module.ProductManifest.build(
        frame_id="frame_x",
        timestamp_utc=TS.isoformat(),
        config=config,
        data_snapshot_id="snap_001",
        metadata=multisat_module.layer_policy_metadata(policy),
    )

    assert manifest.metadata["scene_profile"] == "mountain_rural"
    assert manifest.metadata["enabled_layers"] == ("l1_macro", "l2_topo")
    assert manifest.metadata["disabled_layers"]["l3_urban"]["reason_type"] == "scene_policy"
