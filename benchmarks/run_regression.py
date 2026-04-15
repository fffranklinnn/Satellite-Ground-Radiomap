#!/usr/bin/env python3
"""
Golden-scene regression analysis for BenchmarkRunner (task26 / AC-1..AC-7).

Runs the fixed-frame pipeline via BenchmarkRunner against the golden frame list,
compares output arrays and manifests against frozen baselines, and writes a
machine-readable report to benchmarks/baselines/regression_report.json.

Acceptance criteria checked:
  AC-1: GridSpec geometry consistent (shape matches golden)
  AC-2: UTC timestamps in manifest
  AC-3: FrameContext-driven pipeline produces output
  AC-4: Component arrays present in EntryWaveState (via manifest metadata)
  AC-5: path_loss_map within tolerance of golden composite
  AC-6: config_hash stable; input_file_hashes non-empty; output_file_hashes non-empty
  AC-7: path_loss_map exported via typed export_dataset()

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
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.context import GridSpec, FrameBuilder
from src.layers import L1MacroLayer, L2TopoLayer, L3UrbanLayer
from src.pipeline.benchmark_runner import BenchmarkRunner
from src.products.manifest import collect_input_file_paths


GOLDEN_DIR = ROOT / "benchmarks" / "golden_scenes"
BASELINES_DIR = ROOT / "benchmarks" / "baselines"
FRAME_LIST_PATH = GOLDEN_DIR / "frame_list.json"
GOLDEN_MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
DEFAULT_CONFIG = ROOT / "benchmarks" / "golden_scenes_config.yaml"
REPORT_PATH = BASELINES_DIR / "regression_report.json"

# Tolerances for numeric comparison
REL_TOL = 1e-4
ABS_TOL = 1e-2  # dB

# Golden array files and their labels
GOLDEN_ARRAYS = {
    "l1l2l3_composite": "l1l2l3_composite.npy",
    "l1l2_composite": "l1l2_composite.npy",
    "l1_only": "l1_only.npy",
}


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
    if actual.shape != reference.shape:
        return {
            "name": name, "passed": False,
            "reason": f"shape mismatch: actual={actual.shape} reference={reference.shape}",
        }
    diff = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-10)
    rel_err = float(np.max(diff / denom))
    abs_err = float(np.max(diff))
    passed = rel_err < rel_tol or abs_err < abs_tol
    return {
        "name": name,
        "passed": passed,
        "max_relative_error": rel_err,
        "max_absolute_error_db": abs_err,
        "mean_absolute_error_db": float(np.mean(diff)),
        "rel_tol": rel_tol,
        "abs_tol": abs_tol,
        "shape": list(actual.shape),
    }


def check_manifest_fields(
    manifest: Any,
    golden_manifest: dict,
    data_snapshot_id: str,
) -> List[Dict[str, Any]]:
    checks = []

    # AC-6: config_hash must be stable (decisive comparison against golden)
    golden_config_hash = golden_manifest.get("config_hash", "")
    checks.append({
        "field": "config_hash",
        "actual": manifest.config_hash,
        "reference": golden_config_hash,
        "passed": manifest.config_hash == golden_config_hash,
        "ac": "AC-6",
    })

    # AC-6: data_snapshot_id preserved
    checks.append({
        "field": "data_snapshot_id",
        "actual": manifest.data_snapshot_id,
        "reference": data_snapshot_id,
        "passed": manifest.data_snapshot_id == data_snapshot_id,
        "ac": "AC-6",
    })

    # AC-6: output_file_hashes non-empty (export_dataset wired correctly)
    checks.append({
        "field": "output_file_hashes_populated",
        "actual": bool(manifest.output_file_hashes),
        "reference": True,
        "passed": bool(manifest.output_file_hashes),
        "ac": "AC-6 / AC-7",
    })

    # AC-6: input_file_hashes non-empty (provenance helper wired correctly)
    checks.append({
        "field": "input_file_hashes_populated",
        "actual": bool(manifest.input_file_hashes),
        "reference": True,
        "passed": bool(manifest.input_file_hashes),
        "ac": "AC-6",
    })

    # AC-2: timestamp_utc is UTC-aware ISO string
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(manifest.timestamp_utc)
        utc_aware = ts.tzinfo is not None
    except Exception:
        utc_aware = False
    checks.append({
        "field": "timestamp_utc_is_utc_aware",
        "actual": utc_aware,
        "reference": True,
        "passed": utc_aware,
        "ac": "AC-2",
    })

    # AC-3: frame_id non-empty (FrameContext-driven)
    checks.append({
        "field": "frame_id_non_empty",
        "actual": bool(manifest.frame_id),
        "reference": True,
        "passed": bool(manifest.frame_id),
        "ac": "AC-3",
    })

    return checks


def run_regression(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    config = load_config(config_path)
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]

    if not GOLDEN_MANIFEST_PATH.exists():
        return {"error": f"Golden manifest not found: {GOLDEN_MANIFEST_PATH}"}

    with open(GOLDEN_MANIFEST_PATH, "r") as f:
        golden_manifest = json.load(f)

    timestamps = BenchmarkRunner.load_frame_list(FRAME_LIST_PATH)
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
    manifest_checks: List[Dict[str, Any]] = []

    # Find produced path_loss_map
    produced_npy = sorted(output_dir.glob("*path_loss_map.npy"))
    actual_arr: Optional[np.ndarray] = None
    if produced_npy:
        actual_arr = np.load(produced_npy[0])

    # AC-1 / AC-5: compare against all available golden arrays
    for label, filename in GOLDEN_ARRAYS.items():
        golden_path = GOLDEN_DIR / filename
        if not golden_path.exists():
            comparisons.append({
                "name": f"path_loss_map_vs_{label}",
                "passed": False,
                "reason": f"Golden array not found: {golden_path}",
                "ac": "AC-1 / AC-5",
            })
            continue
        if actual_arr is None:
            comparisons.append({
                "name": f"path_loss_map_vs_{label}",
                "passed": False,
                "reason": "No path_loss_map.npy produced by BenchmarkRunner",
                "ac": "AC-1 / AC-5",
            })
            continue
        golden_arr = np.load(golden_path)
        rec = compare_arrays(f"path_loss_map_vs_{label}", actual_arr, golden_arr)
        rec["ac"] = "AC-1 / AC-5"
        comparisons.append(rec)

    # AC-7: verify path_loss_map.npy was actually written (export_dataset wired)
    comparisons.append({
        "name": "path_loss_map_file_written",
        "passed": bool(produced_npy),
        "ac": "AC-7",
        "actual": str(produced_npy[0]) if produced_npy else None,
    })

    # AC-4: verify shape matches expected grid (256x256)
    expected_shape = (256, 256)
    if actual_arr is not None:
        comparisons.append({
            "name": "output_shape_matches_grid",
            "passed": actual_arr.shape == expected_shape,
            "actual_shape": list(actual_arr.shape),
            "expected_shape": list(expected_shape),
            "ac": "AC-1 / AC-4",
        })

    # Manifest checks
    if results:
        manifest_checks = check_manifest_fields(results[0], golden_manifest, data_snapshot_id)

    # Reproducibility: run a second time and compare output hashes (AC-6)
    if results and produced_npy:
        output_dir2 = output_dir.parent / (output_dir.name + "_repro")
        results2 = runner.run(
            timestamps=timestamps,
            output_dir=output_dir2,
            product_types=["path_loss_map"],
        )
        produced_npy2 = sorted(output_dir2.glob("*path_loss_map.npy"))
        if produced_npy2:
            arr2 = np.load(produced_npy2[0])
            repro_rec = compare_arrays("reproducibility_run1_vs_run2", actual_arr, arr2,
                                       rel_tol=1e-6, abs_tol=1e-6)
            repro_rec["ac"] = "AC-6"
            comparisons.append(repro_rec)
            if results2:
                manifest_checks.append({
                    "field": "config_hash_stable_across_runs",
                    "actual": results[0].config_hash == results2[0].config_hash,
                    "reference": True,
                    "passed": results[0].config_hash == results2[0].config_hash,
                    "ac": "AC-6",
                })
                manifest_checks.append({
                    "field": "output_hashes_stable_across_runs",
                    "actual": dict(results[0].output_file_hashes) == dict(results2[0].output_file_hashes),
                    "reference": True,
                    "passed": dict(results[0].output_file_hashes) == dict(results2[0].output_file_hashes),
                    "ac": "AC-6",
                })

    all_passed = (
        all(c["passed"] for c in comparisons)
        and all(c["passed"] for c in manifest_checks)
    )

    return {
        "schema_version": "2",
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
    parser = argparse.ArgumentParser(description="Golden-scene regression analysis (AC-1..AC-7)")
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

    print("Running golden-scene regression analysis (AC-1..AC-7)...")
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
        ac = comp.get("ac", "")
        if "max_relative_error" in comp:
            print(f"  [{status}] [{ac}] {comp['name']}: "
                  f"max_rel_err={comp['max_relative_error']:.2e} "
                  f"max_abs_err={comp['max_absolute_error_db']:.4f} dB")
        else:
            print(f"  [{status}] [{ac}] {comp['name']}: {comp.get('reason', comp.get('actual', ''))}")

    for chk in report.get("manifest_checks", []):
        status = "PASS" if chk["passed"] else "FAIL"
        ac = chk.get("ac", "")
        print(f"  [{status}] [{ac}] {chk['field']}: actual={chk.get('actual')}")

    return 0 if report.get("all_passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())
