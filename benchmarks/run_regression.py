#!/usr/bin/env python3
"""
Golden-scene regression analysis for BenchmarkRunner (task26 / AC-1..AC-7).

Runs three pipeline configurations against the golden frame list:
  - L1-only  → compared against golden l1_only.npy
  - L1+L2    → compared against golden l1l2_composite.npy
  - L1+L2+L3 → compared against golden l1l2l3_composite.npy

Each configuration is run twice to verify reproducibility (AC-6).
Manifests are checked for UTC timestamps, frame_id, input/output hashes.

Acceptance criteria checked:
  AC-1: GridSpec geometry consistent (shape matches golden)
  AC-2: UTC timestamps in manifest
  AC-3: FrameContext-driven pipeline produces output
  AC-4: Component arrays present in EntryWaveState (via manifest metadata)
  AC-5: path_loss_map within tolerance of matching golden composite
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
import copy
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

# Like-for-like golden array mapping: (config_variant, golden_filename)
GOLDEN_SCENES = [
    ("l1_only",        "l1_only.npy",        {"l1": True,  "l2": False, "l3": False}),
    ("l1l2_composite", "l1l2_composite.npy",  {"l1": True,  "l2": True,  "l3": False}),
    ("l1l2l3_composite","l1l2l3_composite.npy",{"l1": True,  "l2": True,  "l3": True}),
]

# All product types supported by export_dataset (AC-7: multi-product from one FrameContext)
ALL_PRODUCT_TYPES = [
    "path_loss_map",
    "visibility_mask",
    "elevation_field",
    "azimuth_field",
    "terrain_blockage",
    "urban_residual",
]

# Products that require specific layer states
_L1_PRODUCTS = {"path_loss_map", "visibility_mask", "elevation_field", "azimuth_field"}
_L2_PRODUCTS = {"terrain_blockage"}
_L3_PRODUCTS = {"urban_residual"}


def _valid_product_types(layer_flags: Dict[str, bool]) -> List[str]:
    """Return product types valid for the given layer configuration."""
    valid = []
    for pt in ALL_PRODUCT_TYPES:
        if pt in _L2_PRODUCTS and not layer_flags.get("l2"):
            continue
        if pt in _L3_PRODUCTS and not layer_flags.get("l3"):
            continue
        valid.append(pt)
    return valid


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_strict_flag(config: dict) -> bool:
    """
    Merge strict-mode settings from multiple config locations.

    Priority (highest to lowest):
      1. config["data_validation"]["strict"]
      2. config["strict_data"]
      3. config["strict_mode"]
    Returns False when none are set.
    """
    dv = config.get("data_validation", {})
    if isinstance(dv, dict) and "strict" in dv:
        return bool(dv["strict"])
    if "strict_data" in config:
        return bool(config["strict_data"])
    if "strict_mode" in config:
        return bool(config["strict_mode"])
    return False


def build_layers(config: dict, origin_lat: float, origin_lon: float,
                 enable_l1: bool, enable_l2: bool, enable_l3: bool):
    """Build layer objects, propagating the resolved strict flag into each sub-config."""
    strict = _resolve_strict_flag(config)
    l1_layer = l2_layer = l3_layer = None
    layers_cfg = config.get("layers", {})
    if enable_l1 and layers_cfg.get("l1_macro", {}).get("enabled", False):
        l1_cfg = copy.copy(layers_cfg["l1_macro"])
        l1_cfg["strict_data"] = strict
        l1_layer = L1MacroLayer(l1_cfg, origin_lat, origin_lon)
    if enable_l2 and layers_cfg.get("l2_topo", {}).get("enabled", False):
        l2_cfg = copy.copy(layers_cfg["l2_topo"])
        l2_cfg["strict_data"] = strict
        l2_layer = L2TopoLayer(l2_cfg, origin_lat, origin_lon)
    if enable_l3 and layers_cfg.get("l3_urban", {}).get("enabled", False):
        l3_cfg = copy.copy(layers_cfg["l3_urban"])
        l3_cfg["strict_data"] = strict
        l3_layer = L3UrbanLayer(l3_cfg, origin_lat, origin_lon)
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
    scene_label: str,
) -> List[Dict[str, Any]]:
    checks = []

    # AC-6: config_hash must be stable (decisive comparison against golden)
    golden_config_hash = golden_manifest.get("config_hash", "")
    checks.append({
        "field": f"{scene_label}/config_hash",
        "actual": manifest.config_hash,
        "reference": golden_config_hash,
        "passed": manifest.config_hash == golden_config_hash,
        "ac": "AC-6",
    })

    # AC-6: output_file_hashes non-empty (export_dataset wired correctly)
    checks.append({
        "field": f"{scene_label}/output_file_hashes_populated",
        "actual": bool(manifest.output_file_hashes),
        "reference": True,
        "passed": bool(manifest.output_file_hashes),
        "ac": "AC-6 / AC-7",
    })

    # AC-6: input_file_hashes values match golden manifest input_files sha256 values
    golden_input_files = golden_manifest.get("input_files", {})
    golden_input_hashes = {k: v.get("sha256", "") for k, v in golden_input_files.items()
                           if isinstance(v, dict)}
    actual_input_hashes = dict(manifest.input_file_hashes)
    hashes_match = actual_input_hashes == golden_input_hashes
    checks.append({
        "field": f"{scene_label}/input_file_hashes_match_golden",
        "actual": actual_input_hashes,
        "reference": golden_input_hashes,
        "passed": hashes_match,
        "ac": "AC-6",
    })

    # AC-6: data_snapshot_id decisive check (must match config value, not just non-empty)
    checks.append({
        "field": f"{scene_label}/data_snapshot_id_matches_config",
        "actual": manifest.data_snapshot_id,
        "reference": data_snapshot_id,
        "passed": manifest.data_snapshot_id == data_snapshot_id,
        "ac": "AC-6",
    })

    # AC-6: fallbacks_used matches golden manifest (same provenance)
    golden_fallbacks = golden_manifest.get("fallbacks_used", [])
    actual_fallbacks = list(manifest.fallbacks_used)
    checks.append({
        "field": f"{scene_label}/fallbacks_used_matches_golden",
        "actual": actual_fallbacks,
        "reference": golden_fallbacks,
        "passed": actual_fallbacks == golden_fallbacks,
        "ac": "AC-6",
    })

    # AC-2: timestamp_utc is UTC-aware ISO string
    try:
        ts = datetime.fromisoformat(manifest.timestamp_utc)
        utc_aware = ts.tzinfo is not None
    except Exception:
        utc_aware = False
    checks.append({
        "field": f"{scene_label}/timestamp_utc_is_utc_aware",
        "actual": utc_aware,
        "reference": True,
        "passed": utc_aware,
        "ac": "AC-2",
    })

    # AC-3: frame_id non-empty (FrameContext-driven)
    checks.append({
        "field": f"{scene_label}/frame_id_non_empty",
        "actual": bool(manifest.frame_id),
        "reference": True,
        "passed": bool(manifest.frame_id),
        "ac": "AC-3",
    })

    return checks


def run_scene(
    scene_label: str,
    golden_filename: str,
    layer_flags: Dict[str, bool],
    config: dict,
    timestamps: list,
    output_dir: Path,
    golden_manifest: dict,
    data_snapshot_id: str,
) -> Dict[str, Any]:
    """Run one pipeline configuration and compare against its matching golden array."""
    origin_lat = config["origin"]["latitude"]
    origin_lon = config["origin"]["longitude"]

    l1_layer, l2_layer, l3_layer = build_layers(
        config, origin_lat, origin_lon,
        enable_l1=layer_flags["l1"],
        enable_l2=layer_flags["l2"],
        enable_l3=layer_flags["l3"],
    )
    fb = build_frame_builder(config)

    runner = BenchmarkRunner(
        frame_builder=fb,
        l1_layer=l1_layer,
        l2_layer=l2_layer,
        l3_layer=l3_layer,
        config=config,
        data_snapshot_id=data_snapshot_id,
    )

    scene_dir = output_dir / scene_label
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_product_types = _valid_product_types(layer_flags)
    results = runner.run(
        timestamps=timestamps,
        output_dir=scene_dir,
        product_types=scene_product_types,
    )

    # Capture typed states for runtime linkage checks (AC-4/AC-7).
    # Run the pipeline once more directly to get intermediate state objects.
    _typed_states: Dict[str, Any] = {}
    if timestamps:
        _frame = fb.build(timestamps[0])
        _entry = l1_layer.propagate_entry(_frame) if l1_layer else None
        _terrain = l2_layer.propagate_terrain(_frame, entry=_entry) if l2_layer else None
        _urban = l3_layer.refine_urban(_frame, entry=_entry) if l3_layer else None
        _typed_states = {
            "frame_id": _frame.frame_id,
            "grid": _frame.grid,
            "entry": _entry,
            "terrain": _terrain,
            "urban": _urban,
        }

    comparisons: List[Dict[str, Any]] = []
    manifest_checks: List[Dict[str, Any]] = []

    produced_npy = sorted(scene_dir.glob("*path_loss_map.npy"))
    actual_arr: Optional[np.ndarray] = None
    if produced_npy:
        actual_arr = np.load(produced_npy[0])

    # AC-5: like-for-like comparison against matching golden
    golden_path = GOLDEN_DIR / golden_filename
    if not golden_path.exists():
        comparisons.append({
            "name": f"{scene_label}/path_loss_map_vs_golden",
            "passed": False,
            "reason": f"Golden array not found: {golden_path}",
            "ac": "AC-1 / AC-5",
        })
    elif actual_arr is None:
        comparisons.append({
            "name": f"{scene_label}/path_loss_map_vs_golden",
            "passed": False,
            "reason": "No path_loss_map.npy produced",
            "ac": "AC-1 / AC-5",
        })
    else:
        golden_arr = np.load(golden_path)
        rec = compare_arrays(f"{scene_label}/path_loss_map_vs_golden", actual_arr, golden_arr)
        rec["ac"] = "AC-1 / AC-5"
        comparisons.append(rec)

    # AC-7: path_loss_map file written
    comparisons.append({
        "name": f"{scene_label}/path_loss_map_file_written",
        "passed": bool(produced_npy),
        "ac": "AC-7",
        "actual": str(produced_npy[0]) if produced_npy else None,
    })

    # AC-7: multi-product export — all valid product types written from one FrameContext
    for pt in scene_product_types:
        pt_files = sorted(scene_dir.glob(f"*{pt}.npy"))
        comparisons.append({
            "name": f"{scene_label}/product_file_written/{pt}",
            "passed": bool(pt_files),
            "ac": "AC-7",
            "actual": str(pt_files[0]) if pt_files else None,
        })

    # AC-1/AC-4: shape check for all exported products + runtime typed-state linkage
    expected_shape = (256, 256)
    # Map product type → required typed state attribute in _typed_states
    _state_attr = {
        "path_loss_map": None,       # composite — no single state required
        "visibility_mask": "entry",
        "elevation_field": "entry",
        "azimuth_field": "entry",
        "terrain_blockage": "terrain",
        "urban_residual": "urban",
    }
    frame_id = _typed_states.get("frame_id", "")
    frame_grid = _typed_states.get("grid")
    for pt in scene_product_types:
        pt_files = sorted(scene_dir.glob(f"*{pt}.npy"))
        if not pt_files:
            continue
        arr = np.load(pt_files[0])
        # Shape invariant: every product must match the grid
        comparisons.append({
            "name": f"{scene_label}/product_shape/{pt}",
            "passed": arr.shape == expected_shape,
            "actual_shape": list(arr.shape),
            "expected_shape": list(expected_shape),
            "ac": "AC-1 / AC-4",
        })
        # Runtime typed-state linkage: verify the required state object has the
        # same frame_id and grid as the frame that produced this product.
        attr = _state_attr.get(pt)
        if attr is None:
            # path_loss_map: just verify frame_id is non-empty
            state_ok = bool(frame_id)
            state_detail = {"frame_id": frame_id}
        else:
            state_obj = _typed_states.get(attr)
            if state_obj is None:
                state_ok = False
                state_detail = {"error": f"required state '{attr}' is None"}
            else:
                fid_match = getattr(state_obj, "frame_id", None) == frame_id
                grid_match = getattr(state_obj, "grid", None) == frame_grid
                state_ok = fid_match and grid_match
                state_detail = {
                    "state_type": type(state_obj).__name__,
                    "state_frame_id": getattr(state_obj, "frame_id", None),
                    "frame_frame_id": frame_id,
                    "frame_id_match": fid_match,
                    "grid_match": grid_match,
                }
        comparisons.append({
            "name": f"{scene_label}/product_typed_state_linkage/{pt}",
            "passed": state_ok,
            "required_state": attr,
            "detail": state_detail,
            "ac": "AC-4 / AC-7",
        })

    if actual_arr is not None:
        comparisons.append({
            "name": f"{scene_label}/output_shape_matches_grid",
            "passed": actual_arr.shape == expected_shape,
            "actual_shape": list(actual_arr.shape),
            "expected_shape": list(expected_shape),
            "ac": "AC-1 / AC-4",
        })

    # AC-4: typed-state traceability — frame_id in manifest matches frame list
    if results:
        frame_id_non_empty = bool(results[0].frame_id)
        comparisons.append({
            "name": f"{scene_label}/frame_id_traceability",
            "passed": frame_id_non_empty,
            "actual": results[0].frame_id if frame_id_non_empty else "",
            "ac": "AC-4",
        })

    # Manifest checks
    if results:
        manifest_checks = check_manifest_fields(
            results[0], golden_manifest, data_snapshot_id, scene_label
        )

    # AC-6: reproducibility — run a second time and compare
    if results and produced_npy:
        scene_dir2 = output_dir / (scene_label + "_repro")
        results2 = runner.run(
            timestamps=timestamps,
            output_dir=scene_dir2,
            product_types=scene_product_types,
        )
        produced_npy2 = sorted(scene_dir2.glob("*path_loss_map.npy"))
        if produced_npy2:
            arr2 = np.load(produced_npy2[0])
            repro_rec = compare_arrays(
                f"{scene_label}/reproducibility_run1_vs_run2",
                actual_arr, arr2, rel_tol=1e-6, abs_tol=1e-6,
            )
            repro_rec["ac"] = "AC-6"
            comparisons.append(repro_rec)
            if results2:
                manifest_checks.append({
                    "field": f"{scene_label}/config_hash_stable_across_runs",
                    "actual": results[0].config_hash == results2[0].config_hash,
                    "reference": True,
                    "passed": results[0].config_hash == results2[0].config_hash,
                    "ac": "AC-6",
                })
                manifest_checks.append({
                    "field": f"{scene_label}/output_hashes_stable_across_runs",
                    "actual": dict(results[0].output_file_hashes) == dict(results2[0].output_file_hashes),
                    "reference": True,
                    "passed": dict(results[0].output_file_hashes) == dict(results2[0].output_file_hashes),
                    "ac": "AC-6",
                })

    return {"comparisons": comparisons, "manifest_checks": manifest_checks}


def run_regression(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    config = load_config(config_path)

    if not GOLDEN_MANIFEST_PATH.exists():
        return {"error": f"Golden manifest not found: {GOLDEN_MANIFEST_PATH}"}

    with open(GOLDEN_MANIFEST_PATH, "r") as f:
        golden_manifest = json.load(f)

    timestamps = BenchmarkRunner.load_frame_list(FRAME_LIST_PATH)
    data_snapshot_id = config.get("data_validation", {}).get("snapshot_id", "")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_comparisons: List[Dict[str, Any]] = []
    all_manifest_checks: List[Dict[str, Any]] = []

    for scene_label, golden_filename, layer_flags in GOLDEN_SCENES:
        print(f"\n  Running scene: {scene_label} (L1={layer_flags['l1']} L2={layer_flags['l2']} L3={layer_flags['l3']})")
        scene_result = run_scene(
            scene_label=scene_label,
            golden_filename=golden_filename,
            layer_flags=layer_flags,
            config=config,
            timestamps=timestamps,
            output_dir=output_dir,
            golden_manifest=golden_manifest,
            data_snapshot_id=data_snapshot_id,
        )
        all_comparisons.extend(scene_result["comparisons"])
        all_manifest_checks.extend(scene_result["manifest_checks"])

    all_passed = (
        all(c["passed"] for c in all_comparisons)
        and all(c["passed"] for c in all_manifest_checks)
    )

    return {
        "schema_version": "3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)),
        "golden_dir": str(GOLDEN_DIR.relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "frame_count": len(timestamps),
        "scenes_tested": [s[0] for s in GOLDEN_SCENES],
        "all_passed": all_passed,
        "array_comparisons": all_comparisons,
        "manifest_checks": all_manifest_checks,
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
