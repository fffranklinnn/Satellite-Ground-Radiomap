#!/usr/bin/env python3
"""
Small-batch scene validation runner.

Runs a short multi-satellite timeseries for one preset scene config so you can
quickly inspect whether the radiomap output looks reasonable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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


def parse_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_scene: Optional[str] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short scene smoke test.")
    parser.add_argument(
        "--scene",
        required=default_scene is None,
        default=default_scene,
        choices=sorted(PRESET_MAP.keys()),
    )
    parser.add_argument("--start", type=str, default="2025-05-01T12:00:00")
    parser.add_argument("--end", type=str, default="2025-05-01T12:09:30")
    parser.add_argument("--step-minutes", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--output-root", type=str, default="output/smoke_tests")
    parser.add_argument("--gpu-id", type=str, default=None,
                        help="GPU id to expose via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--save-run-config", action="store_true", default=True,
                        help="Write run_config.yaml and run_manifest.json to the output directory")
    parser.add_argument("--no-save-run-config", action="store_false", dest="save_run_config")
    parser.add_argument("--allow-missing-data", action="store_true", default=True)
    parser.add_argument("--no-allow-missing-data", action="store_false", dest="allow_missing_data")
    return parser.parse_args(argv)


def run(cmd: list[str], *, gpu_id: Optional[str] = None) -> None:
    print("[cmd]", " ".join(cmd))
    env = os.environ.copy()
    if gpu_id is not None and str(gpu_id).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()
        env["SGMRM_GPU_ID"] = str(gpu_id).strip()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _ensure_utc_iso(text: str) -> str:
    value = str(text).strip()
    if not value:
        return value
    if value.endswith("Z") or "+" in value[10:] or "-" in value[10:]:
        return value
    return value + "Z"


def _write_run_record(
    output_dir: Path,
    *,
    scene: str,
    preset_config: Path,
    args: argparse.Namespace,
    start_utc: str,
    end_utc: str,
    gpu_id: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(preset_config, output_dir / "preset_config.yaml")
    git_commit = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        git_commit = ""
    run_config = {
        "scene": scene,
        "preset_config": str(preset_config),
        "output_root": str(Path(args.output_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "gpu_id": gpu_id,
        "start": start_utc,
        "end": end_utc,
        "step_minutes": float(args.step_minutes),
        "max_frames": int(args.max_frames),
        "allow_missing_data": bool(args.allow_missing_data),
        "save_run_config": bool(args.save_run_config),
        "git_commit": git_commit,
        "cli_args": vars(args),
    }
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    run_manifest = {
        "scene": scene,
        "preset_config": str(preset_config),
        "gpu_id": gpu_id,
        "start": start_utc,
        "end": end_utc,
        "step_minutes": float(args.step_minutes),
        "max_frames": int(args.max_frames),
        "allow_missing_data": bool(args.allow_missing_data),
        "output_root": str(Path(args.output_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "git_commit": git_commit,
        "command": [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_scene_smoke.py"),
            "--scene",
            scene,
            "--start",
            args.start,
            "--end",
            args.end,
            "--step-minutes",
            str(args.step_minutes),
            "--max-frames",
            str(args.max_frames),
            "--output-root",
            args.output_root,
            *(
                ["--gpu-id", str(gpu_id)]
                if gpu_id is not None and str(gpu_id).strip()
                else []
            ),
            "--allow-missing-data" if args.allow_missing_data else "--no-allow-missing-data",
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    default_scene: Optional[str] = None,
) -> int:
    args = parse_args(argv, default_scene=default_scene)
    config = PRESET_MAP[args.scene]
    output_dir = Path(args.output_root) / args.scene
    start_utc = _ensure_utc_iso(args.start)
    end_utc = _ensure_utc_iso(args.end)

    if args.save_run_config:
        _write_run_record(
            output_dir,
            scene=args.scene,
            preset_config=config,
            args=args,
            start_utc=start_utc,
            end_utc=end_utc,
            gpu_id=args.gpu_id,
        )

    run([
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(config),
        "--check-data-only",
    ], gpu_id=args.gpu_id)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_multisat_timeseries_radiomap.py"),
        "--config",
        str(config),
        "--start",
        start_utc,
        "--end",
        end_utc,
        "--step-minutes",
        str(args.step_minutes),
        "--max-frames",
        str(args.max_frames),
        "--output-dir",
        str(output_dir),
    ]
    if args.allow_missing_data:
        cmd.append("--allow-missing-data")
    run(cmd, gpu_id=args.gpu_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
