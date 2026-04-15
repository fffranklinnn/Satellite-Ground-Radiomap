#!/usr/bin/env python3
"""
Golden-scene regression analysis for BenchmarkRunner (task26 / AC-8).

Runs the fixed-frame pipeline via BenchmarkRunner against the golden frame list,
compares output arrays and manifests against frozen baselines, and writes a
machine-readable report to benchmarks/baselines/regression_report.json.

Usage:
    python benchmarks/run_regression.py [--config CONFIG] [--output-dir DIR]

Exit codes:
    0  All comparisons passed
    1  One or more comparisons failed
    2  Configuration or data error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context import GridSpec, CoverageSpec, FrameBuilder
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.pipeline.benchmark_runner import BenchmarkRunner


GOLDEN_DIR = ROOT / "benchmarks" / "golden_scenes"
BASELINES_DIR = ROOT / "benchmarks" / "baselines"
FRAME_LIST_PATH = GOLDEN_DIR / "frame_list.json"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
DEFAULT_CONFIG = ROOT / "benchmarks" / "golden_scenes_config.yaml"
REPORT_PATH = BASELINES_DIR / "regression_report.json"

# Tolerance for numeric comparison (relative error)
REL_TOL = 1e-4
ABS_TOL = 1e-2  # dB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_layers(config: dict, origin_lat: float, origin_lon: float):
    l1_layer = l2_layer = l3_layer = None
    layers_cfg = config.get("layers", {})

    if layers_cfg.get("l1_macro", {}).get("enabled", False):
        l1_layer = L1MacroLayer(layers_cfg["l1_macro"], origin_lat, origin_lon)

    if layers_cfg.get("l2_topo", {}).get("enabled", False):
        l2_layer = L2TopoLayer(layers_cfg["l2_topo"], origin_lat, origin_lon)

    if layers_cfg.get("l3_urban", {}).get("enabled", False):
        l3_layer = L3UrbanLayer(layers_cfg["l3_urban"], origin_lat, origin_lon)

    return l1_layer, l2_layer, l3_layer


def build_frame_builder(config: dict) -> FrameBuilder:
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]
    l1_cfg = config.get("layers", {}).get("l1_macro", {})
    coarse_km = float(l1_cfg.get("coverage_km", 256.0))
    grid_size = int(l1_cfg.get("grid_size", 256))
    grid = GridSpec.from_legacy_args(origin_lat, origin_lon, coarse_km, grid_size, grid_size)
    return FrameBuilder(grid=grid)


def compare_arrays(
    name: str,
    actual: np.ndarray,
    reference: np.ndarray,
    rel_tol: float = REL_TOL,
    abs_tol: float = ABS_TOL,
) -> Dict[str, Any]:
    """Compare two arrays and return a comparison record."""
    shape_match = actual.shape == reference.shape
    if not shape_match:
        return {
            "name": name,
            "passed": False,
            "reason": f"shape mismatch: actual={actual.shape} reference={reference.shape}",
        }

    diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-10)
    rel_err = float(np.max(diff / denom))
    abs_err = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    passed = rel_err < rel_tol or abs_err < abs_tol
    return {
        "name": name,
        "passed": passed,
        "max_relative_error": rel_err,
        "max_absolute_error_db": abs_err,
        "mean_absolute_error_db": mean_diff,
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "shape": list(actual.shape),
    }


def run_regression(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run the regression analysis and return the report dict."""
    config = load_config(config_path)
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]

    # Load golden manifest
    if not GOLDEN_MANIFEST_PATH.exists():
        return {"error": f"Golden manifest not found: {GOLDEN_MANIFEST_PATH}"}

    with open(GOLDEN_MANIFEST_PATH, "r") as f:
        golden_manifest = json.load(f)

    # Load frame list
    timestamps = BenchmarkRunner.load_frame_list(FRAME_LIST_PATH)

    # Build layers and runner
    l1_layer, l2_layer, l3_layer = build_layers(config, origin_lat, origin_lon)
    fb = build_frame_builder(config)
    data_snapshot_id = config.get("data_validation", {}).get("snapshot_id", "")

    runner = BenchmarkRunner(
        frame_builder=fb,
        l1_layer=l1_layer,
        l2_layer=l2_layer,
        l3_layer=l3_layer,
        config=config,
        data_snapshot_id=data_snapshot_id,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results = runner.run(
        timestamps=timestamps,
        output_dir=output_dir,
        product_types=["path_loss_map"],
    )

    comparisons: List[Dict[str, Any]] = []

    # Compare output arrays against golden scenes
    # The BenchmarkRunner produces path_loss_map.npy; compare against l1l2l3_composite.npy
    golden_composite_path = GOLDEN_DIR / "l1l2l3_composite.npy"
    if golden_composite_path.exists():
        golden_arr = np.load(golden_composite_path)
        # Find the produced npy file
        produced_npy = list(output_dir.glob("*path_loss_map.npy"))
        if produced_npy:
            actual_arr = np.load(produced_npy[0])
            comparisons.append(
                compare_arrays("path_loss_map_vs_l1l2l3_composite", actual_arr, golden_arr)
            )
        else:
            comparisons.append({
                "name": "path_loss_map_vs_l1l2l3_composite",
                "passed": False,
                "reason": "No path_loss_map.npy produced by BenchmarkRunner",
            })
    else:
        comparisons.append({
            "name": "path_loss_map_vs_l1l2l3_composite",
            "passed": False,
            "reason": f"Golden composite not found: {golden_composite_path}",
        })

    # Compare manifest config_hash against golden
    manifest_checks: List[Dict[str, Any]] = []
    if results:
        manifest = results[0]
        golden_config_hash = golden_manifest.get("config_hash", "")
        manifest_checks.append({
            "field": "config_hash",
            "actual": manifest.config_hash,
            "reference": golden_config_hash,
            "passed": True,  # config_hash may differ if config changed; record for audit
            "note": "config_hash comparison is informational; configs may differ between runs",
        })
        manifest_checks.append({
            "field": "data_snapshot_id",
            "actual": manifest.data_snapshot_id,
            "reference": data_snapshot_id,
            "passed": manifest.data_snapshot_id == data_snapshot_id,
        })
        manifest_checks.append({
            "field": "output_file_hashes_populated",
            "actual": bool(manifest.output_file_hashes),
            "reference": True,
            "passed": bool(manifest.output_file_hashes),
        })

    all_passed = all(c["passed"] for c in comparisons) and all(
        c["passed"] for c in manifest_checks
    )

    return {
        "schema_version": "1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "golden_dir": str(GOLDEN_DIR.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "frame_count": len(timestamps),
        "all_passed": all_passed,
        "array_comparisons": comparisons,
        "manifest_checks": manifest_checks,
        "rel_tol": REL_TOL,
        "abs_tol": ABS_TOL,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden-scene regression analysis")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "benchmarks" / "baselines" / "regression_run"))
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    print(f"Running golden-scene regression analysis...")
    print(f"  config:     {config_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  frame_list: {FRAME_LIST_PATH}")

    try:
        report = run_regression(config_path, output_dir)
    except Exception as exc:
        import traceback
        print(f"\nRegression run failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport written to: {REPORT_PATH}")
    print(f"all_passed: {report.get('all_passed', False)}")

    for comp in report.get("array_comparisons", []):
        status = "PASS" if comp["passed"] else "FAIL"
        if "max_relative_error" in comp:
            print(f"  [{status}] {comp['name']}: "
                  f"max_rel_err={comp['max_relative_error']:.2e} "
                  f"max_abs_err={comp['max_absolute_error_db']:.4f} dB")
        else:
            print(f"  [{status}] {comp['name']}: {comp.get('reason', '')}")

    return 0 if report.get("all_passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())
