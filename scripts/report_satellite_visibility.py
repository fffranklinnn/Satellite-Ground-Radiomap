#!/usr/bin/env python3
"""
Report visible satellites over a location for a time range.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.layers import L1MacroLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report satellite visibility over time.")
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml")
    parser.add_argument("--origin-lat", type=float, default=None)
    parser.add_argument("--origin-lon", type=float, default=None)
    parser.add_argument("--start", type=str, default="2025-01-01T00:00:00")
    parser.add_argument("--end", type=str, default="2025-01-01T23:00:00")
    parser.add_argument("--step-hours", type=float, default=1.0)
    parser.add_argument("--min-elevation-deg", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def _parse_iso_utc(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _top_sat_summary(items: List[Dict], top_k: int) -> str:
    if not items:
        return "-"
    subset = items[:max(top_k, 1)]
    return "; ".join(
        f"{x['norad_id']}(el={x['elevation_deg']:.1f} az={x['azimuth_deg']:.1f})"
        for x in subset
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = _load_config(cfg_path)

    origin_cfg = cfg.get("origin", {})
    origin_lat = float(args.origin_lat if args.origin_lat is not None else origin_cfg.get("latitude", 34.3416))
    origin_lon = float(args.origin_lon if args.origin_lon is not None else origin_cfg.get("longitude", 108.9398))

    l1_cfg = cfg.get("layers", {}).get("l1_macro", {})
    l1 = L1MacroLayer(l1_cfg, origin_lat, origin_lon)

    start = _parse_iso_utc(args.start)
    end = _parse_iso_utc(args.end)
    step = timedelta(hours=float(args.step_hours))

    rows: List[Dict[str, object]] = []
    t = start
    while t <= end:
        visible = l1.get_visible_satellites(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            timestamp=t,
            min_elevation_deg=args.min_elevation_deg,
        )
        best = visible[0] if visible else None
        rows.append({
            "timestamp_utc": t.isoformat(),
            "visible_count": len(visible),
            "best_norad": best["norad_id"] if best else "",
            "best_el_deg": f"{best['elevation_deg']:.3f}" if best else "",
            "best_az_deg": f"{best['azimuth_deg']:.3f}" if best else "",
            "top_list": _top_sat_summary(visible, args.top_k),
        })
        t += step

    print("=" * 80)
    print(f"Location: lat={origin_lat:.6f}, lon={origin_lon:.6f}")
    print(f"Range   : {start.isoformat()} -> {end.isoformat()} (step={args.step_hours}h)")
    print(f"Min el  : {args.min_elevation_deg:.1f} deg")
    print("=" * 80)
    for row in rows:
        print(
            f"{row['timestamp_utc']} | N={row['visible_count']:>3} | "
            f"best={row['best_norad']} el={row['best_el_deg']} az={row['best_az_deg']} | "
            f"{row['top_list']}"
        )

    if args.output_csv:
        out_path = Path(args.output_csv)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved: {out_path}")


if __name__ == "__main__":
    main()

