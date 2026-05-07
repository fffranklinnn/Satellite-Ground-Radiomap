from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import main as main_module
from src.context.frame_context import FrameContext
from src.context.grid_spec import GridSpec
from src.pipeline.benchmark_runner import BenchmarkRunner
from src.planning import MissingRequiredInputError, resolve_layer_policy


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
