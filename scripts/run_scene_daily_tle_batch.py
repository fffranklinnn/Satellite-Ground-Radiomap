#!/usr/bin/env python3
"""
Run a multi-day scene experiment with day-specific TLE files.

This is meant for validation runs where orbital inputs should change per day.
Each day is executed as an independent sub-run under its own output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESET_MAP = {
    "xian_urban": PROJECT_ROOT / "configs" / "presets" / "xian_urban.yaml",
    "qinling_mountain": PROJECT_ROOT / "configs" / "presets" / "qinling_mountain.yaml",
    "huashan_mountain": PROJECT_ROOT / "configs" / "presets" / "huashan_mountain.yaml",
    "loess_plateau": PROJECT_ROOT / "configs" / "presets" / "loess_plateau.yaml",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a daily-TLE multi-day scene experiment.")
    parser.add_argument("--scene", default="xian_urban", choices=sorted(PRESET_MAP.keys()))
    parser.add_argument("--start-date", type=str, default="2025-05-01")
    parser.add_argument("--end-date", type=str, default="2025-05-07")
    parser.add_argument("--step-seconds", type=float, default=30.0)
    parser.add_argument("--output-root", type=str, default="output/weekly_experiments")
    parser.add_argument("--gpu-id", type=str, default=None, help="GPU id to expose via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--allow-missing-data", action="store_true", default=True)
    parser.add_argument("--no-allow-missing-data", action="store_false", dest="allow_missing_data")
    parser.add_argument("--max-frames-per-day", type=int, default=None)
    return parser.parse_args(argv)


def _parse_date(text: str) -> date:
    return date.fromisoformat(text)


def _ensure_utc_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _run(cmd: list[str], *, gpu_id: Optional[str] = None) -> None:
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    if gpu_id is not None and str(gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()
        env["SGMRM_GPU_ID"] = str(gpu_id).strip()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _daily_tle_path(day: date) -> Path:
    return PROJECT_ROOT / "data" / "starlink-2025-tle" / f"{day.isoformat()}.tle"


def _build_daily_config(base_cfg: dict, day: date, scene: str, output_dir: Path) -> dict:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("scene", {})
    cfg["scene"]["profile"] = base_cfg.get("scene", {}).get("profile") or (
        "urban_flat" if scene == "xian_urban" else "mountain_rural"
    )
    cfg.setdefault("time", {})
    cfg["time"]["start"] = f"{day.isoformat()}T00:00:00"
    cfg["time"]["end"] = f"{day.isoformat()}T23:59:30"
    cfg["time"]["timezone"] = "UTC"
    cfg.setdefault("output", {})
    cfg["output"]["directory"] = str(output_dir)
    cfg.setdefault("layers", {}).setdefault("l1_macro", {})
    cfg["layers"]["l1_macro"]["tle_file"] = str(_daily_tle_path(day))
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    start_day = _parse_date(args.start_date)
    end_day = _parse_date(args.end_date)
    if end_day < start_day:
        raise SystemExit("--end-date must be >= --start-date")

    preset_path = PRESET_MAP[args.scene]
    base_cfg = _load_yaml(preset_path)
    output_root = Path(args.output_root) / args.scene
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "scene": args.scene,
        "preset_config": str(preset_path),
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "step_seconds": float(args.step_seconds),
        "gpu_id": args.gpu_id,
        "allow_missing_data": bool(args.allow_missing_data),
        "max_frames_per_day": args.max_frames_per_day,
        "days": [],
    }

    day = start_day
    while day <= end_day:
        tle_path = _daily_tle_path(day)
        if not tle_path.exists():
            raise FileNotFoundError(f"Missing daily TLE: {tle_path}")

        day_dir = output_root / day.isoformat()
        daily_cfg = _build_daily_config(base_cfg, day, args.scene, day_dir)
        daily_cfg_path = day_dir / "run_config.yaml"
        _save_yaml(daily_cfg_path, daily_cfg)
        shutil.copy2(preset_path, day_dir / "preset_config.yaml")

        start_utc = _ensure_utc_iso(datetime.combine(day, datetime.min.time()))
        end_utc = _ensure_utc_iso(datetime(day.year, day.month, day.day, 23, 59, 30))
        step_minutes = float(args.step_seconds) / 60.0

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_multisat_timeseries_radiomap.py"),
            "--config",
            str(daily_cfg_path),
            "--start",
            start_utc,
            "--end",
            end_utc,
            "--step-minutes",
            str(step_minutes),
            "--output-dir",
            str(day_dir),
        ]
        if args.max_frames_per_day is not None:
            cmd.extend(["--max-frames", str(args.max_frames_per_day)])
        if args.allow_missing_data:
            cmd.append("--allow-missing-data")

        _run(cmd, gpu_id=args.gpu_id)
        summary["days"].append(
            {
                "date": day.isoformat(),
                "tle_file": str(tle_path),
                "output_dir": str(day_dir),
                "run_config": str(daily_cfg_path),
            }
        )
        day += timedelta(days=1)

    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
