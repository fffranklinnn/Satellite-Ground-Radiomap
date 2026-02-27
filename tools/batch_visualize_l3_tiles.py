"""
Batch renderer for per-tile L3 2D/3D visualizations.
L3 单 tile 2D/3D 可视化批量导出脚本。

Purpose / 作用:
- Export 2D and 3D visualization for many tile ids from a tile list.
- 从 tile 列表批量导出每个 tile 的 2D 和 3D 图。

Main capabilities / 核心功能:
- Iterate tile ids from CSV.
- 从 CSV 读取并遍历 tile_id。
- Render Height/Occ/NLoS/Loss 2D panel and 3D surface per tile.
- 为每个 tile 生成高度/占据/NLoS/损耗的 2D 图及 3D 图。
- Save one summary JSON per tile and one batch summary JSON.
- 保存每 tile summary 以及全局 batch summary。

Interfaces / 接口:
- CLI entrypoint: `python tools/batch_visualize_l3_tiles.py ...`
- 命令行入口：`python tools/batch_visualize_l3_tiles.py ...`

Relationship / 与其他脚本关系:
- Reads tile cache produced by `tools/build_l3_tile_cache.py`.
- 读取 `tools/build_l3_tile_cache.py` 产出的 tile 缓存。
- Reuses plotting helpers in `tools/visualize_l3_tile.py`.
- 复用 `tools/visualize_l3_tile.py` 中的绘图函数。

Example / 调用示例:
```powershell
conda run -n radiodiff python tools/batch_visualize_l3_tiles.py `
  --tile-list data/l3_urban/xian/tile_list_xian_60.csv `
  --tiles-root data/l3_urban/xian/tiles_60 `
  --output-dir data/l3_urban/xian/visuals_60 `
  --incident-dir 1 0 1
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layers.l3_urban import compute_nlos_mask
from tools.visualize_l3_tile import _load_tile_arrays, _map_loss_db, _plot_2d, _plot_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export 2D/3D visuals for L3 tiles.")
    parser.add_argument("--tile-list", type=Path, required=True, help="CSV with tile_id column.")
    parser.add_argument("--tiles-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--incident-dir",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 1.0],
        metavar=("EAST", "NORTH", "UP"),
    )
    parser.add_argument("--nlos-loss-db", type=float, default=20.0)
    parser.add_argument("--occ-loss-db", type=float, default=30.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional cap for debug.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.tile_list)
    if "tile_id" not in df.columns:
        raise ValueError("tile-list CSV must contain tile_id column.")

    tile_ids = df["tile_id"].astype(str).tolist()
    if args.max_tiles is not None:
        tile_ids = tile_ids[: args.max_tiles]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    incident_dir = [float(v) for v in args.incident_dir]
    summaries = []
    total = len(tile_ids)
    for idx, tile_id in enumerate(tile_ids, start=1):
        tile_dir = args.tiles_root / tile_id
        if not tile_dir.exists():
            continue

        height_m, occ = _load_tile_arrays(tile_dir)
        nlos_mask = compute_nlos_mask(height_m, incident_dir)
        loss_db = _map_loss_db(
            nlos_mask=nlos_mask,
            occ=occ,
            nlos_loss_db=args.nlos_loss_db,
            occ_loss_db=args.occ_loss_db,
        )

        out_2d = args.output_dir / f"{tile_id}_2d.png"
        out_3d = args.output_dir / f"{tile_id}_3d.png"
        out_json = args.output_dir / f"{tile_id}_summary.json"

        _plot_2d(
            tile_id=tile_id,
            output_path=out_2d,
            height_m=height_m,
            occ=occ,
            nlos_mask=nlos_mask,
            loss_db=loss_db,
            incident_dir=incident_dir,
            dpi=args.dpi,
        )
        _plot_3d(
            tile_id=tile_id,
            output_path=out_3d,
            height_m=height_m,
            nlos_mask=nlos_mask,
            incident_dir=incident_dir,
            dpi=args.dpi,
        )

        summary = {
            "tile_id": tile_id,
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
        summaries.append(summary)

        if idx % 10 == 0 or idx == total:
            print(f"processed {idx}/{total}")

    batch_summary = {
        "num_requested": total,
        "num_exported": len(summaries),
        "incident_dir": incident_dir,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "batch_summary.json").write_text(
        json.dumps(batch_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(batch_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
