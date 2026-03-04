#!/usr/bin/env python3
"""
Batch runner for Xi'an city-scale full-physics radiomap experiments.

This script orchestrates repeated runs of scripts/generate_xian_city_radiomap.py
across time and parameter sweeps (frequency, rain rate), while recording an
experiment index for reproducibility.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITY_SCRIPT = PROJECT_ROOT / "scripts" / "generate_xian_city_radiomap.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run Xi'an city radiomap experiments with parameter sweeps."
    )
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml",
                        help="Base config YAML path")
    parser.add_argument("--tile-list", type=str, default="data/l3_urban/xian/tile_list_xian_60.csv",
                        help="Tile list CSV path forwarded to city script")
    parser.add_argument("--output-root", type=str, default="output/city_batch_experiments",
                        help="Root output directory for all experiments")
    parser.add_argument("--start", type=str, default="2025-01-01T00:00:00",
                        help="Batch start time (ISO, UTC assumed if tz missing)")
    parser.add_argument("--end", type=str, default="2025-01-01T23:00:00",
                        help="Batch end time (ISO, UTC assumed if tz missing)")
    parser.add_argument("--step-hours", type=float, default=1.0,
                        help="Time step in hours")
    parser.add_argument("--freqs-ghz", type=str, default="14.5",
                        help='Comma-separated frequencies in GHz, e.g. "10,14.5,26.5"')
    parser.add_argument("--rain-rates", type=str, default="0",
                        help='Comma-separated rain rates in mm/h, e.g. "0,5,25,50"')
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure DPI forwarded to city script")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional debug cap for tile count")
    parser.add_argument("--save-components", action="store_true",
                        help="Also save city-wide L2/L3 mosaics per run")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent experiments")
    parser.add_argument("--gpu-ids", type=str, default="",
                        help='Optional GPU IDs for round-robin assignment, e.g. "0,1,2,3"')
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip runs when target composite NPY already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned commands without executing")
    parser.add_argument("--norad-id", action="append", default=None,
                        help="Limit candidate satellites to NORAD ID; can be repeated")
    return parser.parse_args()


def _parse_iso_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_float_list(text: str) -> List[float]:
    vals = []
    for chunk in text.split(","):
        x = chunk.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("At least one numeric value is required.")
    return vals


def _parse_gpu_ids(text: str) -> List[str]:
    out: List[str] = []
    for chunk in text.split(","):
        token = chunk.strip()
        if token:
            out.append(token)
    return out


def _time_grid(start: datetime, end: datetime, step_hours: float) -> List[datetime]:
    if step_hours <= 0:
        raise ValueError("--step-hours must be > 0")
    step = timedelta(hours=step_hours)
    out: List[datetime] = []
    t = start
    while t <= end:
        out.append(t)
        t += step
    return out


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_cfg(base_cfg: Dict, freq_ghz: float, rain_rate: float) -> Dict:
    cfg = json.loads(json.dumps(base_cfg))

    layers = cfg.setdefault("layers", {})
    l1 = layers.setdefault("l1_macro", {})
    l2 = layers.setdefault("l2_topo", {})

    l1["frequency_ghz"] = float(freq_ghz)
    l1["rain_rate_mm_h"] = float(rain_rate)
    l2["frequency_ghz"] = float(freq_ghz)

    return cfg


def _stat_npy(path: Path) -> Dict[str, object]:
    arr = np.load(path, mmap_mode="r")
    return {
        "shape": list(arr.shape),
        "loss_mean": float(np.nanmean(arr)),
        "loss_std": float(np.nanstd(arr)),
        "loss_min": float(np.nanmin(arr)),
        "loss_max": float(np.nanmax(arr)),
    }


def _fmt_seconds(sec: float) -> str:
    sec_i = int(max(sec, 0.0))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _print_progress(done: int,
                    total: int,
                    started_at: float,
                    ok: int,
                    failed: int,
                    skipped: int,
                    last_exp_id: str = "",
                    last_status: str = "",
                    gpu_id: Optional[str] = None) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    rate = done / elapsed
    remain = max(total - done, 0)
    eta_sec = (remain / rate) if rate > 1e-12 else float("inf")
    eta_str = "N/A" if not np.isfinite(eta_sec) else _fmt_seconds(eta_sec)
    pct = (100.0 * done / total) if total > 0 else 100.0
    print(
        f"[progress] {done}/{total} ({pct:.1f}%) | ok={ok} failed={failed} skipped={skipped} "
        f"| rate={rate:.3f}/s | eta={eta_str} | last={last_exp_id} "
        f"status={last_status} gpu={gpu_id if gpu_id is not None else '-'}"
    )


def _run_single_experiment(task: Dict[str, Any]) -> Dict[str, object]:
    row = dict(task["row"])
    cmd: List[str] = list(task["cmd"])
    log_path = Path(task["log_path"])
    expected_npy = Path(task["expected_npy"])
    gpu_id: Optional[str] = task.get("gpu_id")

    run_t0 = time.time()
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["SGMRM_GPU_ID"] = str(gpu_id)

    try:
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write(f"# gpu_id={gpu_id if gpu_id is not None else 'cpu/default'}\n")
            log_f.write(f"# cmd={' '.join(cmd)}\n\n")
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                env=env,
            )

        row["duration_sec"] = round(time.time() - run_t0, 3)
        row["return_code"] = int(proc.returncode)
        if proc.returncode != 0:
            row["status"] = "failed"
            row["error"] = f"non-zero return code: {proc.returncode}"
        elif not expected_npy.exists():
            row["status"] = "failed"
            row["error"] = "missing expected output npy"
        else:
            row["status"] = "ok"
            row.update(_stat_npy(expected_npy))
    except Exception as exc:
        row["duration_sec"] = round(time.time() - run_t0, 3)
        row["status"] = "failed"
        row["error"] = repr(exc)

    return row


def main() -> None:
    args = parse_args()

    base_cfg_path = _resolve_path(args.config)
    tile_list_path = _resolve_path(args.tile_list)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "configs").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(base_cfg_path)
    freqs = _parse_float_list(args.freqs_ghz)
    rain_rates = _parse_float_list(args.rain_rates)
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    start = _parse_iso_utc(args.start)
    end = _parse_iso_utc(args.end)
    times = _time_grid(start, end, args.step_hours)

    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")

    plan = list(itertools.product(freqs, rain_rates, times))
    total = len(plan)
    print(f"[batch-city] total experiments: {total}")
    print(f"[batch-city] freq sweep (GHz): {freqs}")
    print(f"[batch-city] rain sweep (mm/h): {rain_rates}")
    print(f"[batch-city] time points: {len(times)} | {start.isoformat()} -> {end.isoformat()}")
    print(f"[batch-city] workers: {args.workers}")
    print(f"[batch-city] gpu_ids: {gpu_ids if gpu_ids else 'not specified'}")
    if gpu_ids and args.workers > len(gpu_ids):
        print(
            "[batch-city] warning: workers > number of gpu_ids, "
            "some GPU IDs will be shared."
        )

    index_rows: List[Dict[str, object]] = []
    run_tasks: List[Dict[str, Any]] = []
    t0_all = time.time()
    progress_t0 = time.time()

    for i, (freq, rain, ts) in enumerate(plan, start=1):
        stamp = ts.strftime("%Y%m%dT%H%M%SZ")
        freq_tag = str(freq).replace(".", "p")
        rain_tag = str(rain).replace(".", "p")
        exp_id = f"exp_f{freq_tag}GHz_r{rain_tag}mmh_t{stamp}"
        exp_dir = output_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"xian_city_f{freq_tag}_r{rain_tag}"
        expected_npy = exp_dir / f"{prefix}_{stamp}_composite.npy"
        expected_png = exp_dir / f"{prefix}_{stamp}.png"

        row: Dict[str, object] = {
            "planned_idx": i,
            "exp_id": exp_id,
            "timestamp_utc": ts.isoformat(),
            "frequency_ghz": float(freq),
            "rain_rate_mm_h": float(rain),
            "gpu_id": None,
            "status": "pending",
            "output_dir": str(exp_dir),
            "composite_npy": str(expected_npy),
            "composite_png": str(expected_png),
            "duration_sec": None,
            "return_code": None,
            "shape": None,
            "loss_mean": None,
            "loss_std": None,
            "loss_min": None,
            "loss_max": None,
            "error": "",
        }

        if args.skip_existing and expected_npy.exists():
            row["status"] = "skipped_existing"
            try:
                row.update(_stat_npy(expected_npy))
            except Exception as exc:
                row["error"] = f"stat failed: {exc}"
            index_rows.append(row)
            print(f"[{i}/{total}] skip existing: {exp_id}")
            continue

        cfg = _build_cfg(base_cfg, freq_ghz=freq, rain_rate=rain)
        cfg_path = output_root / "configs" / f"{exp_id}.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        cmd = [
            sys.executable,
            str(CITY_SCRIPT),
            "--config", str(cfg_path),
            "--tile-list", str(tile_list_path),
            "--timestamp", ts.replace(tzinfo=None).isoformat(),
            "--output-dir", str(exp_dir),
            "--output-prefix", prefix,
            "--dpi", str(args.dpi),
        ]

        if args.max_tiles is not None:
            cmd.extend(["--max-tiles", str(args.max_tiles)])
        if args.save_components:
            cmd.append("--save-components")
        if args.norad_id:
            for nid in args.norad_id:
                cmd.extend(["--norad-id", str(nid)])

        assigned_gpu: Optional[str] = None
        if gpu_ids:
            assigned_gpu = gpu_ids[(i - 1) % len(gpu_ids)]
            row["gpu_id"] = assigned_gpu

        print(f"[{i}/{total}] run: {exp_id}")
        if args.dry_run:
            row["status"] = "dry_run"
            row["error"] = " ".join(cmd)
            index_rows.append(row)
            continue

        log_path = output_root / "logs" / f"{exp_id}.log"
        run_tasks.append({
            "row": row,
            "cmd": cmd,
            "log_path": str(log_path),
            "expected_npy": str(expected_npy),
            "gpu_id": assigned_gpu,
        })

    ok = sum(1 for r in index_rows if r["status"] == "ok")
    failed = sum(1 for r in index_rows if r["status"] == "failed")
    skipped = sum(1 for r in index_rows if r["status"] in ("skipped_existing", "dry_run"))
    done = len(index_rows)

    if total > 0 and done > 0:
        _print_progress(done, total, progress_t0, ok, failed, skipped)

    if not args.dry_run and run_tasks:
        if args.workers == 1:
            for task in run_tasks:
                row = _run_single_experiment(task)
                index_rows.append(row)
                done += 1
                if row["status"] == "ok":
                    ok += 1
                elif row["status"] == "failed":
                    failed += 1
                else:
                    skipped += 1
                print(
                    f"[{row['planned_idx']}/{total}] done: {row['exp_id']} | "
                    f"status={row['status']} | gpu={row['gpu_id']} | dt={row['duration_sec']}s"
                )
                _print_progress(
                    done, total, progress_t0, ok, failed, skipped,
                    last_exp_id=str(row.get("exp_id", "")),
                    last_status=str(row.get("status", "")),
                    gpu_id=row.get("gpu_id"),
                )
        else:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_run_single_experiment, t) for t in run_tasks]
                for fut in cf.as_completed(futures):
                    row = fut.result()
                    index_rows.append(row)
                    done += 1
                    if row["status"] == "ok":
                        ok += 1
                    elif row["status"] == "failed":
                        failed += 1
                    else:
                        skipped += 1
                    print(
                        f"[parallel {done}/{total}] done: {row['exp_id']} | "
                        f"status={row['status']} | gpu={row['gpu_id']} | dt={row['duration_sec']}s"
                    )
                    _print_progress(
                        done, total, progress_t0, ok, failed, skipped,
                        last_exp_id=str(row.get("exp_id", "")),
                        last_status=str(row.get("status", "")),
                        gpu_id=row.get("gpu_id"),
                    )

    index_rows.sort(key=lambda r: int(r.get("planned_idx", 0)))

    # Persist index
    csv_path = output_root / "experiment_index.csv"
    json_path = output_root / "experiment_index.json"

    if index_rows:
        fields = list(index_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index_rows)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(index_rows, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in index_rows if r["status"] == "ok")
    failed = sum(1 for r in index_rows if r["status"] == "failed")
    skipped_existing = sum(1 for r in index_rows if r["status"] == "skipped_existing")
    dry = sum(1 for r in index_rows if r["status"] == "dry_run")

    print("=" * 88)
    print("Batch city experiments finished")
    print(f"  total         : {len(index_rows)}")
    print(f"  ok            : {ok}")
    print(f"  failed        : {failed}")
    print(f"  skipped       : {skipped_existing}")
    print(f"  dry-run       : {dry}")
    print(f"  elapsed_sec   : {round(time.time() - t0_all, 3)}")
    print(f"  index_csv     : {csv_path}")
    print(f"  index_json    : {json_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
