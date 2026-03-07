#!/usr/bin/env python3
"""
Post-process Xi'an hourly city-wide radiomap outputs.

Inputs:
  - batch_city_experiments.py output root (24 hourly runs)
  - per-frame NPY files: composite, l2, l3

Outputs:
  - Per-frame PNG for each component:
      composite / l1_link / l2_terrain / l3_urban
  - Time-collage panel PNG (e.g., 4x6 for 24 hours) for each component
  - GIF for each component
  - Summary JSON with discovered frames and visualization scales
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


TS_PATTERN = re.compile(r"(\d{8}T\d{6}Z)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-component time-series PNG/panel/GIF from Xi'an batch outputs."
    )
    parser.add_argument("--batch-root", type=str, required=True,
                        help="Output root produced by batch_city_experiments.py")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Post-process output dir (default: <batch-root>/timeseries_visuals)")
    parser.add_argument("--panel-cols", type=int, default=6,
                        help="Number of columns in time panel")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="GIF frame rate")
    parser.add_argument("--dpi", type=int, default=160,
                        help="PNG DPI")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap for debugging")
    parser.add_argument("--scale-batch-root", type=str, default=None,
                        help="Optional batch root used only for global color-scale calculation. "
                             "Use this to keep color bars consistent across multiple subsets "
                             "(e.g., different frequencies).")
    parser.add_argument("--percentile-low", type=float, default=1.0,
                        help="Lower percentile for color scaling")
    parser.add_argument("--percentile-high", type=float, default=99.0,
                        help="Upper percentile for color scaling")
    return parser.parse_args()


def parse_timestamp_from_text(text: str) -> str:
    m = TS_PATTERN.search(text)
    return m.group(1) if m else ""


def parse_timestamp_for_sort(ts_text: str) -> Tuple[int, str]:
    if not ts_text:
        return (10**18, "")
    key = ts_text.replace("T", "").replace("Z", "")
    try:
        return (int(key), ts_text)
    except Exception:
        return (10**18, ts_text)


def discover_frames(batch_root: Path, max_frames: Optional[int]) -> List[Dict[str, str | Path]]:
    index_json = batch_root / "experiment_index.json"
    frames: List[Dict[str, str | Path]] = []

    if index_json.exists():
        rows = json.loads(index_json.read_text(encoding="utf-8"))
        for row in rows:
            if row.get("status") != "ok":
                continue
            comp_path = Path(str(row.get("composite_npy", "")))
            if not comp_path.exists():
                continue
            stamp = parse_timestamp_from_text(comp_path.name)
            l2_path = comp_path.with_name(comp_path.name.replace("_composite.npy", "_l2.npy"))
            l3_path = comp_path.with_name(comp_path.name.replace("_composite.npy", "_l3.npy"))
            frames.append({
                "timestamp": stamp,
                "composite": comp_path,
                "l2": l2_path,
                "l3": l3_path,
            })
    else:
        for exp_dir in sorted(batch_root.glob("exp_*")):
            if not exp_dir.is_dir():
                continue
            comp_files = sorted(exp_dir.glob("*_composite.npy"))
            if not comp_files:
                continue
            comp_path = comp_files[0]
            stamp = parse_timestamp_from_text(comp_path.name)
            l2_path = comp_path.with_name(comp_path.name.replace("_composite.npy", "_l2.npy"))
            l3_path = comp_path.with_name(comp_path.name.replace("_composite.npy", "_l3.npy"))
            frames.append({
                "timestamp": stamp,
                "composite": comp_path,
                "l2": l2_path,
                "l3": l3_path,
            })

    frames.sort(key=lambda x: parse_timestamp_for_sort(str(x["timestamp"])))
    if max_frames is not None and max_frames > 0:
        frames = frames[:int(max_frames)]
    return frames


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_component_scales(
    frames: List[Dict[str, str | Path]],
    p_low: float,
    p_high: float,
) -> Dict[str, Tuple[float, float]]:
    values: Dict[str, List[np.ndarray]] = {
        "composite": [],
        "l1_link": [],
        "l2_terrain": [],
        "l3_urban": [],
    }

    for f in frames:
        comp = np.load(f["composite"])
        values["composite"].append(comp[np.isfinite(comp)])

        if f["l2"].exists() and f["l3"].exists():
            l2 = np.load(f["l2"])
            l3 = np.load(f["l3"])
            l1 = comp - l2 - l3
            values["l1_link"].append(l1[np.isfinite(l1)])
            values["l2_terrain"].append(l2[np.isfinite(l2)])
            values["l3_urban"].append(l3[np.isfinite(l3)])

    scales: Dict[str, Tuple[float, float]] = {}
    for k, chunks in values.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks)
        if arr.size == 0:
            continue
        vmin = float(np.percentile(arr, p_low))
        vmax = float(np.percentile(arr, p_high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if vmin == vmax:
                vmax = vmin + 1e-6
        scales[k] = (vmin, vmax)
    return scales


def save_frame_png(
    arr: np.ndarray,
    title: str,
    out_file: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Loss (dB)", rotation=270, labelpad=14)
    txt = (
        f"mean={float(np.nanmean(arr)):.3f} dB\n"
        f"std={float(np.nanstd(arr)):.3f} dB\n"
        f"min={float(np.nanmin(arr)):.3f} dB\n"
        f"max={float(np.nanmax(arr)):.3f} dB"
    )
    ax.text(
        0.015,
        0.015,
        txt,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
    )
    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_time_panel(
    image_files: List[Path],
    labels: List[str],
    out_file: Path,
    panel_cols: int,
    title: str,
    dpi: int,
) -> None:
    n = len(image_files)
    cols = max(1, panel_cols)
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.ravel(axes))

    for idx, ax in enumerate(axes_list):
        if idx >= n:
            ax.axis("off")
            continue
        img = imageio.imread(image_files[idx])
        ax.imshow(img)
        ax.set_title(labels[idx], fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_gif(image_files: List[Path], out_file: Path, fps: float) -> None:
    with imageio.get_writer(out_file, mode="I", fps=float(fps)) as writer:
        for p in image_files:
            writer.append_data(imageio.imread(p))


def main() -> None:
    args = parse_args()

    batch_root = Path(args.batch_root)
    if not batch_root.is_absolute():
        batch_root = (Path(__file__).resolve().parents[1] / batch_root).resolve()
    if not batch_root.exists():
        raise FileNotFoundError(f"batch root not found: {batch_root}")

    out_dir = Path(args.output_dir) if args.output_dir else (batch_root / "timeseries_visuals")
    if not out_dir.is_absolute():
        out_dir = (Path(__file__).resolve().parents[1] / out_dir).resolve()
    ensure_dir(out_dir)

    frames = discover_frames(batch_root=batch_root, max_frames=args.max_frames)
    if not frames:
        raise RuntimeError("No valid frames discovered in batch output.")

    have_components = all(f["l2"].exists() and f["l3"].exists() for f in frames)
    if not have_components:
        print("[WARN] Some frames are missing l2/l3 npy; only composite will be visualized for those frames.")

    scale_batch_root: Path = batch_root
    scale_frames = frames
    if args.scale_batch_root:
        scale_batch_root = Path(args.scale_batch_root)
        if not scale_batch_root.is_absolute():
            scale_batch_root = (Path(__file__).resolve().parents[1] / scale_batch_root).resolve()
        if not scale_batch_root.exists():
            raise FileNotFoundError(f"scale batch root not found: {scale_batch_root}")
        scale_frames = discover_frames(batch_root=scale_batch_root, max_frames=None)
        if not scale_frames:
            raise RuntimeError(f"No valid frames discovered in scale batch root: {scale_batch_root}")

    scales = compute_component_scales(
        frames=scale_frames,
        p_low=float(args.percentile_low),
        p_high=float(args.percentile_high),
    )
    if "composite" not in scales:
        raise RuntimeError("Failed to compute color scale for composite maps.")

    component_info = [
        ("composite", "Composite Radiomap", "viridis"),
        ("l1_link", "L1 Link Loss", "magma"),
        ("l2_terrain", "L2 Terrain Loss", "cividis"),
        ("l3_urban", "L3 Urban Loss", "plasma"),
    ]

    written: Dict[str, List[Path]] = {k: [] for k, _, _ in component_info}
    written_labels: Dict[str, List[str]] = {k: [] for k, _, _ in component_info}

    for i, frame in enumerate(frames, start=1):
        stamp = str(frame.get("timestamp", "") or "")
        frame_label = stamp if stamp else f"frame_{i:03d}"

        comp_path = Path(str(frame["composite"]))
        l2_path = Path(str(frame["l2"]))
        l3_path = Path(str(frame["l3"]))

        comp = np.load(comp_path).astype(np.float32)
        l2 = np.load(l2_path).astype(np.float32) if l2_path.exists() else None
        l3 = np.load(l3_path).astype(np.float32) if l3_path.exists() else None
        l1 = (comp - l2 - l3).astype(np.float32) if (l2 is not None and l3 is not None) else None

        arr_map = {
            "composite": comp,
            "l1_link": l1,
            "l2_terrain": l2,
            "l3_urban": l3,
        }

        if l1 is not None:
            l1_npy_dir = out_dir / "l1_link" / "npy"
            ensure_dir(l1_npy_dir)
            np.save(l1_npy_dir / f"{i:03d}_{stamp or f'frame_{i:03d}'}.npy", l1.astype(np.float32, copy=False))

        for comp_key, comp_title, cmap in component_info:
            arr = arr_map[comp_key]
            if arr is None or comp_key not in scales:
                continue
            comp_dir = out_dir / comp_key / "frames"
            ensure_dir(comp_dir)
            out_png = comp_dir / f"{i:03d}_{stamp or f'frame_{i:03d}'}.png"
            vmin, vmax = scales[comp_key]
            save_frame_png(
                arr=arr,
                title=f"{comp_title} | {stamp}",
                out_file=out_png,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                dpi=int(args.dpi),
            )
            written[comp_key].append(out_png)
            written_labels[comp_key].append(frame_label)

        if i == 1 or i % 4 == 0 or i == len(frames):
            print(f"[post] rendered frame {i}/{len(frames)}")

    for comp_key, comp_title, _ in component_info:
        files = written.get(comp_key, [])
        if not files:
            continue
        panel_out = out_dir / f"{comp_key}_panel.png"
        gif_out = out_dir / f"{comp_key}.gif"
        save_time_panel(
            image_files=files,
            labels=written_labels[comp_key],
            out_file=panel_out,
            panel_cols=int(args.panel_cols),
            title=f"{comp_title} Time Series",
            dpi=int(args.dpi),
        )
        save_gif(
            image_files=files,
            out_file=gif_out,
            fps=float(args.fps),
        )
        print(f"[post] saved panel: {panel_out}")
        print(f"[post] saved gif  : {gif_out}")

    summary = {
        "batch_root": str(batch_root),
        "scale_batch_root": str(scale_batch_root),
        "frame_count": len(frames),
        "scale_frame_count": len(scale_frames),
        "scales": {
            k: {"vmin": v[0], "vmax": v[1]} for k, v in scales.items()
        },
        "components_written": {
            k: len(v) for k, v in written.items()
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Post-processing finished")
    print(f"  batch_root : {batch_root}")
    print(f"  out_dir    : {out_dir}")
    print(f"  frames     : {len(frames)}")
    print(f"  summary    : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
