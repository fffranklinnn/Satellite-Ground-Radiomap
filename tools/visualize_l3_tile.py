"""
Visualize one L3 tile or a stitched multi-tile scene (2D + 3D).
L3 单 tile 或多 tile 拼接场景可视化（2D + 3D）。

Purpose / 作用:
- Render L3 urban cache into interpretable images.
- 将 L3 城市缓存渲染为可解释图像。
- Support both single tile (256x256 m) and stitched scenes (e.g. 4/9/16 tiles).
- 同时支持单 tile（256x256m）和拼接场景（4/9/16 等）。

Main capabilities / 核心功能:
- Load `H.npy`/`Occ.npy` for one tile or a square mosaic around an anchor tile id.
- 读取单 tile 或以锚点 tile 为中心（或左上）的方形拼接区域。
- Compute NLoS mask and derived L3 loss map using the same logic as L3 runtime.
- 复用 L3 逻辑计算 NLoS 与损耗图。
- Export:
  - 2D panel (Height / Occupancy / NLoS / Loss)
  - 3D surface (height with NLoS overlay)
  - summary JSON (shape, counts, tile layout, output paths)
- 输出：
  - 2D 四联图（高度/占据/NLoS/损耗）
  - 3D 高度叠加 NLoS 图
  - summary JSON（尺寸、统计、tile 布局与输出路径）

Interfaces / 接口:
- CLI entrypoint: `python tools/visualize_l3_tile.py ...`
- 命令行入口：`python tools/visualize_l3_tile.py ...`

Relationship / 与其他脚本关系:
- Input tile cache is produced by `tools/build_l3_tile_cache.py`.
- 输入 tile cache 来自 `tools/build_l3_tile_cache.py`。
- Core NLoS computation is imported from `src/layers/l3_urban.py`.
- 核心 NLoS 计算复用 `src/layers/l3_urban.py`。
- Batch rendering wrapper is `tools/batch_visualize_l3_tiles.py`.
- 批量渲染由 `tools/batch_visualize_l3_tiles.py` 调用本脚本内函数。

Examples / 调用示例:
```powershell
# Single tile / 单 tile
conda run -n radiodiff python tools/visualize_l3_tile.py `
  --tile-id xian_47202_15894 `
  --tiles-root data/l3_urban/xian/tiles_60 `
  --output-dir data/l3_urban/xian/visuals

# 3x3 stitched scene / 3x3 拼接场景（9 tiles）
conda run -n radiodiff python tools/visualize_l3_tile.py `
  --tile-id xian_47202_15894 `
  --tiles-root data/l3_urban/xian/tiles_60 `
  --output-dir data/l3_urban/xian/visuals `
  --mosaic-count 9 --anchor center
```
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layers.l3_urban import compute_nlos_mask

_GRID_TILE_ID_RE = re.compile(r"^(.*)_(-?\d+)_(-?\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D/3D visualization for one tile or stitched tile scenes."
    )
    parser.add_argument("--tile-id", type=str, required=True, help="Anchor tile id under tiles root.")
    parser.add_argument("--tiles-root", type=Path, default=Path("data/l3_urban/tiles"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/l3_urban/visuals"))
    parser.add_argument(
        "--incident-dir",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 1.0],
        metavar=("EAST", "NORTH", "UP"),
        help="Incident ENU vector for NLoS/loss visualization.",
    )
    parser.add_argument("--nlos-loss-db", type=float, default=20.0)
    parser.add_argument("--occ-loss-db", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--mosaic-count",
        type=int,
        default=1,
        help="Total number of stitched tiles; must be a square number (1,4,9,16,...).",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        choices=["auto", "center", "topleft"],
        default="auto",
        help="Anchor semantics when mosaic-count > 1. auto=center for odd side, topleft for even side.",
    )
    parser.add_argument(
        "--strict-mosaic",
        action="store_true",
        help="Fail if any tile in the requested mosaic is missing. Default fills missing tiles with zeros.",
    )
    return parser.parse_args()


def _load_tile_arrays(tile_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    h_path = tile_dir / "H.npy"
    occ_path = tile_dir / "Occ.npy"
    if not h_path.exists():
        raise FileNotFoundError(f"Missing file: {h_path}")
    h = np.load(h_path).astype(np.float32)
    if occ_path.exists():
        occ = np.load(occ_path).astype(np.uint8)
    else:
        occ = (h > 0).astype(np.uint8)
    return h, occ


def _parse_grid_tile_id(tile_id: str) -> tuple[str, int, int]:
    match = _GRID_TILE_ID_RE.match(tile_id)
    if not match:
        raise ValueError(
            "tile-id must match '<prefix>_<i>_<j>' for stitched visualization, "
            f"got: {tile_id}"
        )
    prefix, i_str, j_str = match.groups()
    return prefix, int(i_str), int(j_str)


def _resolve_anchor_mode(side: int, anchor: str) -> str:
    if anchor == "auto":
        return "center" if side % 2 == 1 else "topleft"
    if anchor == "center" and side % 2 == 0:
        raise ValueError("anchor=center only supports odd side mosaic (e.g. 3x3, 5x5).")
    return anchor


def _build_mosaic_layout(tile_id: str, mosaic_side: int, anchor: str) -> List[List[str]]:
    prefix, i_anchor, j_anchor = _parse_grid_tile_id(tile_id)
    mode = _resolve_anchor_mode(mosaic_side, anchor)

    if mode == "center":
        half = mosaic_side // 2
        left_i = i_anchor - half
        top_j = j_anchor + half
    else:  # topleft
        left_i = i_anchor
        top_j = j_anchor

    layout: List[List[str]] = []
    for r in range(mosaic_side):
        row: List[str] = []
        jj = top_j - r
        for c in range(mosaic_side):
            ii = left_i + c
            row.append(f"{prefix}_{ii}_{jj}")
        layout.append(row)
    return layout


def _assemble_scene(
    tiles_root: Path,
    anchor_tile_id: str,
    mosaic_side: int,
    anchor: str,
    strict_mosaic: bool,
) -> tuple[np.ndarray, np.ndarray, List[List[str]], List[str], int]:
    if mosaic_side == 1:
        tile_dir = tiles_root / anchor_tile_id
        height_m, occ = _load_tile_arrays(tile_dir)
        return height_m, occ, [[anchor_tile_id]], [], int(height_m.shape[0])

    layout = _build_mosaic_layout(anchor_tile_id, mosaic_side, anchor)
    tile_ids = [tile for row in layout for tile in row]

    tile_cache = {}
    base_shape = None
    missing: List[str] = []

    for tile_id in tile_ids:
        tile_dir = tiles_root / tile_id
        if tile_dir.exists():
            h, occ = _load_tile_arrays(tile_dir)
            tile_cache[tile_id] = (h, occ)
            if base_shape is None:
                base_shape = h.shape
        else:
            missing.append(tile_id)

    if base_shape is None:
        raise FileNotFoundError("No tile data found for the requested mosaic.")
    if strict_mosaic and missing:
        raise FileNotFoundError(f"Missing {len(missing)} tiles in mosaic: {missing[:5]}")

    zero_h = np.zeros(base_shape, dtype=np.float32)
    zero_occ = np.zeros(base_shape, dtype=np.uint8)

    stitched_h_rows = []
    stitched_occ_rows = []
    for row in layout:
        h_row = []
        occ_row = []
        for tile_id in row:
            h, occ = tile_cache.get(tile_id, (zero_h, zero_occ))
            h_row.append(h)
            occ_row.append(occ)
        stitched_h_rows.append(np.concatenate(h_row, axis=1))
        stitched_occ_rows.append(np.concatenate(occ_row, axis=1))

    scene_h = np.concatenate(stitched_h_rows, axis=0)
    scene_occ = np.concatenate(stitched_occ_rows, axis=0)
    return scene_h, scene_occ, layout, missing, int(base_shape[0])


def _map_loss_db(
    nlos_mask: np.ndarray,
    occ: np.ndarray,
    nlos_loss_db: float,
    occ_loss_db: float | None,
) -> np.ndarray:
    loss_db = np.zeros(nlos_mask.shape, dtype=np.float32)
    loss_db[nlos_mask] = float(nlos_loss_db)
    if occ_loss_db is not None:
        occ_mask = occ.astype(bool)
        loss_db[occ_mask] = np.maximum(loss_db[occ_mask], float(occ_loss_db))
    return loss_db


def _draw_tile_grid(ax, layout: List[List[str]], tile_px: int) -> None:
    rows = len(layout)
    cols = len(layout[0]) if rows > 0 else 0
    h = rows * tile_px
    w = cols * tile_px

    for r in range(1, rows):
        y = r * tile_px - 0.5
        ax.axhline(y, color="white", linewidth=0.6, alpha=0.9)
    for c in range(1, cols):
        x = c * tile_px - 0.5
        ax.axvline(x, color="white", linewidth=0.6, alpha=0.9)

    for r, row in enumerate(layout):
        for c, tile_id in enumerate(row):
            cx = c * tile_px + tile_px * 0.5
            cy = r * tile_px + tile_px * 0.5
            ax.text(
                cx,
                cy,
                tile_id,
                color="white",
                fontsize=6,
                ha="center",
                va="center",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
            )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def _plot_2d(
    tile_id: str,
    output_path: Path,
    height_m: np.ndarray,
    occ: np.ndarray,
    nlos_mask: np.ndarray,
    loss_db: np.ndarray,
    incident_dir: Sequence[float],
    dpi: int,
    layout: List[List[str]] | None = None,
    tile_px: int = 256,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    im0 = axes[0, 0].imshow(height_m, cmap="terrain", origin="upper")
    axes[0, 0].set_title("Height H (m)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(occ, cmap="gray_r", vmin=0, vmax=1, origin="upper")
    axes[0, 1].set_title("Occupancy Occ")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(nlos_mask.astype(np.uint8), cmap="magma", vmin=0, vmax=1, origin="upper")
    axes[1, 0].set_title("NLoS Mask")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(loss_db, cmap="inferno", origin="upper")
    axes[1, 1].set_title("L3 Loss (dB)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        if layout is not None and (len(layout) > 1 or len(layout[0]) > 1):
            _draw_tile_grid(ax, layout=layout, tile_px=tile_px)

    title = (
        f"L3 Scene {tile_id} | incident_dir=[{incident_dir[0]:.2f},"
        f" {incident_dir[1]:.2f}, {incident_dir[2]:.2f}]"
    )
    fig.suptitle(title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_3d(
    tile_id: str,
    output_path: Path,
    height_m: np.ndarray,
    nlos_mask: np.ndarray,
    incident_dir: Sequence[float],
    dpi: int,
) -> None:
    rows, cols = height_m.shape
    x = np.arange(cols, dtype=np.float32)
    y = np.arange(rows, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    z = height_m.copy()
    zmax = float(z.max())
    if zmax <= 0:
        zmax = 1.0

    base_colors = cm.terrain(np.clip(z / zmax, 0.0, 1.0))
    overlay = np.zeros_like(base_colors)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.2
    overlay[..., 2] = 0.2
    overlay[..., 3] = nlos_mask.astype(np.float32) * 0.75
    facecolors = base_colors.copy()
    alpha = overlay[..., 3:4]
    facecolors[..., :3] = (1.0 - alpha) * facecolors[..., :3] + alpha * overlay[..., :3]

    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xx,
        yy,
        z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
        rcount=rows,
        ccount=cols,
    )

    ax.set_title(
        f"3D Height + NLoS Overlay | {tile_id}\n"
        f"incident_dir=[{incident_dir[0]:.2f}, {incident_dir[1]:.2f}, {incident_dir[2]:.2f}]"
    )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_zlabel("height (m)")
    ax.view_init(elev=45, azim=-120)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mosaic_side = int(round(math.sqrt(args.mosaic_count)))
    if mosaic_side * mosaic_side != args.mosaic_count:
        raise ValueError("--mosaic-count must be a square number: 1,4,9,16,...")

    height_m, occ, layout, missing_tiles, tile_px = _assemble_scene(
        tiles_root=args.tiles_root,
        anchor_tile_id=args.tile_id,
        mosaic_side=mosaic_side,
        anchor=args.anchor,
        strict_mosaic=args.strict_mosaic,
    )

    incident_dir = [float(v) for v in args.incident_dir]
    nlos_mask = compute_nlos_mask(height_m, incident_dir)
    loss_db = _map_loss_db(
        nlos_mask=nlos_mask,
        occ=occ,
        nlos_loss_db=args.nlos_loss_db,
        occ_loss_db=args.occ_loss_db,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_tag = args.tile_id if mosaic_side == 1 else f"{args.tile_id}_mosaic{mosaic_side}x{mosaic_side}"
    out_2d = args.output_dir / f"{scene_tag}_2d.png"
    out_3d = args.output_dir / f"{scene_tag}_3d.png"
    out_json = args.output_dir / f"{scene_tag}_summary.json"

    _plot_2d(
        tile_id=scene_tag,
        output_path=out_2d,
        height_m=height_m,
        occ=occ,
        nlos_mask=nlos_mask,
        loss_db=loss_db,
        incident_dir=incident_dir,
        dpi=args.dpi,
        layout=layout,
        tile_px=tile_px,
    )
    _plot_3d(
        tile_id=scene_tag,
        output_path=out_3d,
        height_m=height_m,
        nlos_mask=nlos_mask,
        incident_dir=incident_dir,
        dpi=args.dpi,
    )

    summary = {
        "anchor_tile_id": args.tile_id,
        "scene_tag": scene_tag,
        "mosaic_count": int(args.mosaic_count),
        "mosaic_side": int(mosaic_side),
        "tile_layout": layout,
        "missing_tiles": missing_tiles,
        "incident_dir": incident_dir,
        "shape": [int(height_m.shape[0]), int(height_m.shape[1])],
        "height_min_m": float(height_m.min()),
        "height_max_m": float(height_m.max()),
        "occ_count": int((occ > 0).sum()),
        "nlos_count": int(nlos_mask.sum()),
        "loss_min_db": float(loss_db.min()),
        "loss_max_db": float(loss_db.max()),
        "out_2d_png": str(out_2d),
        "out_3d_png": str(out_3d),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
