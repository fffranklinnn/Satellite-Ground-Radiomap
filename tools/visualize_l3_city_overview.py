"""
City-wide overview visualization for L3 tile cache.
城市级 L3 tile cache 总览可视化。

Outputs:
- one combined panel image (Height / Occupancy / Coverage)
- three standalone images (Height, Occupancy, Coverage)
- one summary JSON
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


_GRID_TILE_ID_RE = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate city-wide overview images from L3 tile cache."
    )
    parser.add_argument("--tile-list", type=Path, required=True, help="CSV with tile_id column.")
    parser.add_argument("--tiles-root", type=Path, required=True, help="Root directory of tile caches.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-tag", type=str, default="city_overview")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--native-pixel-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save standalone maps in native scene pixels (no axes/colorbar).",
    )
    parser.add_argument(
        "--save-native-combined",
        action="store_true",
        help="Also save a native-pixel combined strip image (Height|Occ|Coverage).",
    )
    return parser.parse_args()


def _load_tile_arrays(tile_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    h_path = tile_dir / "H.npy"
    occ_path = tile_dir / "Occ.npy"
    if not h_path.exists():
        raise FileNotFoundError(f"Missing file: {h_path}")
    height_m = np.load(h_path).astype(np.float32)
    if occ_path.exists():
        occ = np.load(occ_path).astype(np.uint8)
    else:
        occ = (height_m > 0).astype(np.uint8)
    return height_m, occ


def _parse_grid_tile_id(tile_id: str) -> Tuple[int, int]:
    match = _GRID_TILE_ID_RE.match(tile_id)
    if not match:
        raise ValueError(f"tile_id must match '<prefix>_<i>_<j>', got: {tile_id}")
    return int(match.group(2)), int(match.group(3))


def _save_single_map(
    data: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
    im = ax.imshow(data, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _save_single_map_native(
    data: np.ndarray,
    output_path: Path,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    plt.imsave(
        output_path,
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
    )


def _to_rgba(data: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = plt.get_cmap(cmap)(norm(data))
    return (rgba * 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.tile_list)
    if "tile_id" not in df.columns:
        raise ValueError("tile-list CSV must contain tile_id column.")

    parsed: List[Tuple[str, int, int]] = []
    for tile_id in df["tile_id"].astype(str):
        i, j = _parse_grid_tile_id(tile_id)
        parsed.append((tile_id, i, j))

    if not parsed:
        raise RuntimeError("No tile ids found in tile list.")

    min_i = min(i for _, i, _ in parsed)
    max_i = max(i for _, i, _ in parsed)
    min_j = min(j for _, _, j in parsed)
    max_j = max(j for _, _, j in parsed)

    sample_h = None
    for tile_id, _, _ in parsed:
        h_path = args.tiles_root / tile_id / "H.npy"
        if h_path.exists():
            sample_h = np.load(h_path)
            break
    if sample_h is None:
        raise FileNotFoundError("No H.npy found under tiles-root for provided tile list.")

    tile_px_h, tile_px_w = sample_h.shape
    rows = (max_j - min_j + 1) * tile_px_h
    cols = (max_i - min_i + 1) * tile_px_w

    scene_h = np.zeros((rows, cols), dtype=np.float32)
    scene_occ = np.zeros((rows, cols), dtype=np.uint8)
    scene_cov = np.zeros((rows, cols), dtype=np.uint8)

    missing: List[str] = []
    loaded = 0
    for tile_id, i, j in parsed:
        tile_dir = args.tiles_root / tile_id
        h_path = tile_dir / "H.npy"
        occ_path = tile_dir / "Occ.npy"
        if not h_path.exists():
            missing.append(tile_id)
            continue

        height_m = np.load(h_path).astype(np.float32)
        occ = np.load(occ_path).astype(np.uint8) if occ_path.exists() else (height_m > 0).astype(np.uint8)

        r0 = (max_j - j) * tile_px_h
        c0 = (i - min_i) * tile_px_w
        r1 = r0 + tile_px_h
        c1 = c0 + tile_px_w

        scene_h[r0:r1, c0:c1] = height_m
        scene_occ[r0:r1, c0:c1] = occ
        scene_cov[r0:r1, c0:c1] = 1
        loaded += 1

    nonzero = scene_h[scene_h > 0]
    if nonzero.size > 0:
        h_vmin = float(np.percentile(nonzero, 2))
        h_vmax = float(np.percentile(nonzero, 98))
        if h_vmax <= h_vmin:
            h_vmin = float(nonzero.min())
            h_vmax = float(nonzero.max())
    else:
        h_vmin, h_vmax = 0.0, 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_height = args.output_dir / f"{args.scene_tag}_height.png"
    out_occ = args.output_dir / f"{args.scene_tag}_occ.png"
    out_cov = args.output_dir / f"{args.scene_tag}_coverage.png"
    out_combined = args.output_dir / f"{args.scene_tag}_combined.png"
    out_combined_native = args.output_dir / f"{args.scene_tag}_combined_native.png"
    out_summary = args.output_dir / f"{args.scene_tag}_summary.json"

    if args.native_pixel_output:
        _save_single_map_native(
            data=scene_h,
            output_path=out_height,
            cmap="terrain",
            vmin=h_vmin,
            vmax=h_vmax,
        )
        _save_single_map_native(
            data=scene_occ,
            output_path=out_occ,
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
        )
        _save_single_map_native(
            data=scene_cov,
            output_path=out_cov,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        _save_single_map(
            data=scene_h,
            title="City Overview: Height (m)",
            output_path=out_height,
            cmap="terrain",
            vmin=h_vmin,
            vmax=h_vmax,
            dpi=args.dpi,
        )
        _save_single_map(
            data=scene_occ,
            title="City Overview: Occupancy",
            output_path=out_occ,
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            dpi=args.dpi,
        )
        _save_single_map(
            data=scene_cov,
            title="City Overview: Tile Coverage",
            output_path=out_cov,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            dpi=args.dpi,
        )

    fig, axes = plt.subplots(1, 3, figsize=(24, 9), constrained_layout=True)
    im0 = axes[0].imshow(scene_h, cmap="terrain", origin="upper", vmin=h_vmin, vmax=h_vmax)
    axes[0].set_title("Height (m)")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(scene_occ, cmap="gray_r", origin="upper", vmin=0, vmax=1)
    axes[1].set_title("Occupancy")
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("y (px)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(scene_cov, cmap="viridis", origin="upper", vmin=0, vmax=1)
    axes[2].set_title("Coverage")
    axes[2].set_xlabel("x (px)")
    axes[2].set_ylabel("y (px)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(args.scene_tag)
    fig.savefig(out_combined, dpi=args.dpi)
    plt.close(fig)

    combined_native_written = False
    if args.save_native_combined:
        h_rgba = _to_rgba(scene_h, cmap="terrain", vmin=h_vmin, vmax=h_vmax)
        occ_rgba = _to_rgba(scene_occ, cmap="gray_r", vmin=0.0, vmax=1.0)
        cov_rgba = _to_rgba(scene_cov, cmap="viridis", vmin=0.0, vmax=1.0)
        strip = np.concatenate([h_rgba, occ_rgba, cov_rgba], axis=1)
        plt.imsave(out_combined_native, strip, origin="upper")
        combined_native_written = True

    summary = {
        "scene_tag": args.scene_tag,
        "tile_list": str(args.tile_list),
        "tiles_root": str(args.tiles_root),
        "num_tiles_requested": int(len(parsed)),
        "num_tiles_loaded": int(loaded),
        "num_tiles_missing": int(len(missing)),
        "missing_tiles": missing,
        "grid_i_range": [int(min_i), int(max_i)],
        "grid_j_range": [int(min_j), int(max_j)],
        "grid_shape_tiles": [int(max_j - min_j + 1), int(max_i - min_i + 1)],
        "tile_px": [int(tile_px_h), int(tile_px_w)],
        "scene_shape_px": [int(rows), int(cols)],
        "native_pixel_output": bool(args.native_pixel_output),
        "out_height_png": str(out_height),
        "out_occ_png": str(out_occ),
        "out_coverage_png": str(out_cov),
        "out_combined_png": str(out_combined),
        "out_combined_native_png": str(out_combined_native) if combined_native_written else None,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
