#!/usr/bin/env python3
"""
Generate L1 Macro Layer radio maps for 2025-01-01.

Produces:
  output/l1_maps/l1_loss_00utc.png   — 00:00 UTC
  output/l1_maps/l1_loss_06utc.png   — 06:00 UTC
  output/l1_maps/l1_loss_12utc.png   — 12:00 UTC
  output/l1_maps/l1_loss_18utc.png   — 18:00 UTC
  output/l1_maps/l1_loss_comparison.png — 2×2 panel

Usage:
    conda run -n SatelliteRM python scripts/generate_l1_map.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.layers.l1_macro import L1MacroLayer
from src.utils.plotter import plot_radio_map

# ── Configuration ────────────────────────────────────────────────────────────

ORIGIN_LAT = 39.9042   # Beijing
ORIGIN_LON = 116.4074
COVERAGE_KM = 256.0

L1_CONFIG = {
    'grid_size': 256,
    'coverage_km': COVERAGE_KM,
    'resolution_m': 1000.0,
    'frequency_ghz': 10.0,
    'satellite_altitude_km': 550.0,
    'tec': 10.0,
    'rain_rate_mm_h': 0.0,
    'ionex_file': 'data/l1_space/data/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz',
    'era5_file':  'data/l1_space/data/data_stream-oper_stepType-instant.nc',
    'tle_file':   'data/2025_0101.tle',
}

TIMESTAMPS = [
    datetime(2025, 1, 1,  0, 0, 0),
    datetime(2025, 1, 1,  6, 0, 0),
    datetime(2025, 1, 1, 12, 0, 0),
    datetime(2025, 1, 1, 18, 0, 0),
]

OUTPUT_DIR = Path('output/l1_maps')

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing L1 layer...")
    layer = L1MacroLayer(L1_CONFIG, ORIGIN_LAT, ORIGIN_LON)
    print(f"  IONEX loaded: {layer.ionex is not None}")
    print(f"  ERA5  loaded: {layer.era5  is not None}")

    maps = {}
    for ts in TIMESTAMPS:
        label = ts.strftime('%H') + 'utc'
        print(f"Computing {ts.strftime('%Y-%m-%d %H:%M UTC')} ...")
        loss = layer.compute(ts)
        maps[label] = loss

        valid = loss[~np.isnan(loss)]
        print(f"  shape={loss.shape}  min={valid.min():.2f}  "
              f"max={valid.max():.2f}  mean={valid.mean():.2f} dB")

        plot_radio_map(
            loss,
            title=f"L1 Macro Loss — {ts.strftime('%Y-%m-%d %H:%M UTC')}",
            output_file=str(OUTPUT_DIR / f"l1_loss_{label}.png"),
            cmap='viridis_r',
            origin_lat=ORIGIN_LAT,
            origin_lon=ORIGIN_LON,
            coverage_km=COVERAGE_KM,
            show_stats=True,
            dpi=150,
        )

    # ── 2×2 comparison panel ─────────────────────────────────────────────────
    print("Generating comparison panel...")
    labels = [ts.strftime('%H') + 'utc' for ts in TIMESTAMPS]
    titles = [ts.strftime('%H:%M UTC') for ts in TIMESTAMPS]

    all_valid = np.concatenate([maps[l][~np.isnan(maps[l])] for l in labels])
    vmin, vmax = all_valid.min(), all_valid.max()

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle('L1 Macro Layer Loss — 2025-01-01 (10 GHz / 550 km LEO)',
                 fontsize=15, fontweight='bold')

    lat_extent = COVERAGE_KM / 2.0 / 111.0
    lon_extent = COVERAGE_KM / 2.0 / (111.0 * np.cos(np.radians(ORIGIN_LAT)))
    n = 256
    n_ticks = 5
    tick_px   = np.linspace(0, n - 1, n_ticks)
    lat_vals  = np.linspace(ORIGIN_LAT + lat_extent, ORIGIN_LAT - lat_extent, n_ticks)
    lon_vals  = np.linspace(ORIGIN_LON - lon_extent, ORIGIN_LON + lon_extent, n_ticks)

    for ax, label, title in zip(axes.flat, labels, titles):
        loss = maps[label]
        im = ax.imshow(loss, cmap='viridis_r', vmin=vmin, vmax=vmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(tick_px)
        ax.set_xticklabels([f"{v:.1f}°E" for v in lon_vals], fontsize=7)
        ax.set_yticks(tick_px)
        ax.set_yticklabels([f"{v:.1f}°N" for v in lat_vals], fontsize=7)

        valid = loss[~np.isnan(loss)]
        stats = (f"mean={valid.mean():.2f}\nstd={valid.std():.2f}\n"
                 f"min={valid.min():.2f}\nmax={valid.max():.2f} dB")
        ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=7,
                va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Loss (dB)', shrink=0.6)
    plt.tight_layout()
    out = OUTPUT_DIR / 'l1_loss_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison panel to {out}")
    print("Done.")


if __name__ == '__main__':
    main()
