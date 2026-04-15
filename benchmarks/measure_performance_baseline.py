#!/usr/bin/env python3
"""
Measure performance baseline: wall time and peak memory for single-frame
and multi-frame pipeline runs.

Usage:
    python benchmarks/measure_performance_baseline.py [--config CONFIG] [--output OUTPUT]
"""

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.engine import RadioMapAggregator
from src.layers.base import LayerContext


def _measure_single_frame(config: dict) -> dict:
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]
    frame_dt = datetime.fromisoformat(config["time"]["start"]).replace(tzinfo=timezone.utc)

    l3_cfg = config["layers"].get("l3_urban", {})
    incident_dir_cfg = l3_cfg.get("incident_dir")
    context = LayerContext.from_any({"incident_dir": incident_dir_cfg}) if incident_dir_cfg else None

    l1 = L1MacroLayer(config["layers"]["l1_macro"], origin_lat, origin_lon)
    l2 = L2TopoLayer(config["layers"]["l2_topo"], origin_lat, origin_lon)
    l3 = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    agg = RadioMapAggregator(l1, l2, l3)

    tracemalloc.start()
    t0 = time.perf_counter()
    agg.get_layer_contributions(origin_lat, origin_lon, frame_dt, context)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "elapsed_s": round(elapsed, 3),
        "peak_memory_mb": round(peak_bytes / 1024 / 1024, 2),
        "frame_time_utc": frame_dt.isoformat(),
    }


def _measure_multi_frame(config: dict, n_frames: int = 4) -> dict:
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]
    start_dt = datetime.fromisoformat(config["time"]["start"]).replace(tzinfo=timezone.utc)
    step = timedelta(hours=config["time"].get("step_hours", 6))

    l3_cfg = config["layers"].get("l3_urban", {})
    incident_dir_cfg = l3_cfg.get("incident_dir")
    context = LayerContext.from_any({"incident_dir": incident_dir_cfg}) if incident_dir_cfg else None

    l1 = L1MacroLayer(config["layers"]["l1_macro"], origin_lat, origin_lon)
    l2 = L2TopoLayer(config["layers"]["l2_topo"], origin_lat, origin_lon)
    l3 = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
    agg = RadioMapAggregator(l1, l2, l3)

    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(n_frames):
        frame_dt = start_dt + i * step
        agg.get_layer_contributions(origin_lat, origin_lon, frame_dt, context)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n_frames": n_frames,
        "total_elapsed_s": round(elapsed, 3),
        "per_frame_elapsed_s": round(elapsed / n_frames, 3),
        "peak_memory_mb": round(peak_bytes / 1024 / 1024, 2),
    }


def measure(config_path: str, output_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Measuring single-frame performance...")
    single = _measure_single_frame(config)
    print(f"  elapsed={single['elapsed_s']}s  peak_mem={single['peak_memory_mb']}MB")

    print("Measuring multi-frame performance (4 frames)...")
    multi = _measure_multi_frame(config, n_frames=4)
    print(f"  total={multi['total_elapsed_s']}s  per_frame={multi['per_frame_elapsed_s']}s  peak_mem={multi['peak_memory_mb']}MB")

    result = {
        "schema_version": "1",
        "config_path": config_path,
        "single_frame": single,
        "multi_frame": multi,
        "notes": (
            "Peak memory measured with tracemalloc (Python-level allocations only). "
            "NumPy/C-level allocations may not be fully captured. "
            "These baselines are pre-refactor; use for comparison after FrameContext introduction."
        ),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBaseline written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/golden_scenes_config.yaml")
    parser.add_argument("--output", default="benchmarks/baselines/performance_baseline.json")
    args = parser.parse_args()
    measure(args.config, args.output)


if __name__ == "__main__":
    main()
