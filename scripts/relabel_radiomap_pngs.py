#!/usr/bin/env python3
"""
Re-render existing radiomap PNGs from saved NPY/JSON artifacts.

This keeps the underlying numeric data unchanged and only refreshes the
visualization label/style (for example, using loss-mode with clearer colorbar text).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import plot_radio_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render existing radiomap PNG outputs")
    parser.add_argument("roots", nargs="+", help="Experiment/output roots to relabel")
    parser.add_argument("--display-mode", choices=["loss", "score"], default="loss",
                        help="Visualization mode for regenerated PNGs")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI")
    return parser.parse_args()


def iter_frame_jsons(root: Path) -> Iterable[Path]:
    frame_dir = root / "frame_json"
    if frame_dir.exists():
        yield from sorted(frame_dir.glob("*.json"))


def iter_per_sat_jsons(root: Path) -> Iterable[Path]:
    per_sat_root = root / "per_satellite"
    if per_sat_root.exists():
        yield from sorted(per_sat_root.glob("*/*.json"))


def render_from_payload(root: Path, payload: Dict, png_rel: str, npy_rel: str, *,
                        title: str, cmap: str, dpi: int, display_mode: str) -> None:
    png_path = root / png_rel
    npy_path = root / npy_rel
    arr = np.load(npy_path)

    origin = payload.get("origin", {})
    product = payload.get("product_grid", {})
    origin_lat = origin.get("lat")
    origin_lon = origin.get("lon")
    coverage_km = product.get("coverage_km")

    plot_radio_map(
        loss_map=arr,
        title=title,
        output_file=str(png_path),
        cmap=cmap,
        display_mode=display_mode,
        show_colorbar=True,
        show_stats=True,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=coverage_km,
        dpi=dpi,
    )


def relabel_frame(root: Path, frame_json: Path, *, cmap: str, dpi: int, display_mode: str) -> None:
    payload = json.loads(frame_json.read_text())
    artifacts = payload["artifacts"]
    ts = payload["timestamp_utc"].replace("+00:00", " UTC")
    title = (
        "Multi-Satellite Time-Series Radiomap\n"
        f"{ts} | fusion={payload['fusion']['mode']} | sats={payload['satellite_count_used']}"
    )
    render_from_payload(
        root,
        payload,
        artifacts["png"],
        artifacts["npy"],
        title=title,
        cmap=cmap,
        dpi=dpi,
        display_mode=display_mode,
    )
    payload.setdefault("fusion", {})["display_mode"] = display_mode
    frame_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relabel_per_sat(root: Path, sat_json: Path, *, cmap: str, dpi: int, display_mode: str) -> None:
    payload = json.loads(sat_json.read_text())
    art = payload["artifact"]
    sat = payload["satellite"]
    ts = payload["timestamp_utc"].replace("+00:00", " UTC")
    title = (
        "Single-Satellite Radiomap\n"
        f"{ts} | norad={sat.get('norad_id')} | "
        f"az={sat.get('azimuth_deg', float('nan')):.2f} | "
        f"el={sat.get('elevation_deg', float('nan')):.2f}"
    )
    # Per-satellite payloads do not store origin/product info, so infer from parent frame JSON.
    frame_json_name = payload["frame_id"] + ".json"
    frame_json = root / "frame_json" / frame_json_name
    if frame_json.exists():
        frame_payload = json.loads(frame_json.read_text())
        payload["origin"] = frame_payload.get("origin", {})
        payload["product_grid"] = frame_payload.get("product_grid", {})

    render_from_payload(
        root,
        payload,
        art["png"],
        art["npy"],
        title=title,
        cmap=cmap,
        dpi=dpi,
        display_mode=display_mode,
    )
    sat_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    roots = [Path(p).resolve() for p in args.roots]
    for root in roots:
        if not root.exists():
            print(f"[skip] missing root: {root}")
            continue
        frame_jsons = list(iter_frame_jsons(root))
        per_sat_jsons = list(iter_per_sat_jsons(root))
        print(f"[root] {root} | frames={len(frame_jsons)} | per_sat={len(per_sat_jsons)}")
        for fp in frame_jsons:
            relabel_frame(root, fp, cmap=args.cmap, dpi=args.dpi, display_mode=args.display_mode)
        for sp in per_sat_jsons:
            relabel_per_sat(root, sp, cmap=args.cmap, dpi=args.dpi, display_mode=args.display_mode)


if __name__ == "__main__":
    main()
