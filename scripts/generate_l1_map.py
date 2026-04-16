#!/usr/bin/env python3
"""
Generate L1 Macro Layer radio maps for validation and analysis.

Produces four sets of outputs:
  output/l1_maps/hourly/          — 24 hourly maps + 4x6 panel
  output/l1_maps/rain_scan/       — rain rate sweep (0/10/50/100 mm/h)
  output/l1_maps/freq_scan/       — frequency sweep (1/3/10/30 GHz)
  output/l1_maps/components/      — FSPL / atm / iono / total decomposition

Usage:
    conda run -n SatelliteRM python scripts/generate_l1_map.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.layers.l1_macro import L1MacroLayer
from src.utils.plotter import plot_radio_map

# ── Shared configuration ──────────────────────────────────────────────────────

ORIGIN_LAT  = 39.9042
ORIGIN_LON  = 116.4074
COVERAGE_KM = 256.0

BASE_CONFIG = {
    'grid_size':             256,
    'coverage_km':           COVERAGE_KM,
    'resolution_m':          1000.0,
    'frequency_ghz':         10.0,
    'satellite_altitude_km': 550.0,
    'tec':                   10.0,
    'rain_rate_mm_h':        0.0,
    'ionex_file': 'data/l1_space/data/cddis_data_2025/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz',
    'era5_file':  'data/l1_space/data/data_stream-oper_stepType-instant.nc',
    'tle_file':   'data/2025_0101.tle',
}

TS_NOON = datetime(2025, 1, 1, 12, 0, 0)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _geo_ticks(n, origin_lat, origin_lon, coverage_km, n_ticks=5):
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    px      = np.linspace(0, n - 1, n_ticks)
    lats    = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, n_ticks)
    lons    = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, n_ticks)
    return px, lats, lons


def _add_geo_ticks(ax, px, lats, lons):
    ax.set_xticks(px)
    ax.set_xticklabels([f"{v:.1f}E" for v in lons], fontsize=7)
    ax.set_yticks(px)
    ax.set_yticklabels([f"{v:.1f}N" for v in lats], fontsize=7)


def _overlay_stats(ax, arr):
    v = arr[~np.isnan(arr)]
    txt = f"mean={v.mean():.1f}\nstd={v.std():.1f}\nmin={v.min():.1f}\nmax={v.max():.1f} dB"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))


# ── 1. Hourly maps ────────────────────────────────────────────────────────────

def run_hourly():
    out = Path('output/l1_maps/hourly')
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Hourly maps (00-23 UTC) ===")
    layer = L1MacroLayer(BASE_CONFIG, ORIGIN_LAT, ORIGIN_LON)
    print(f"  IONEX={layer.ionex is not None}  ERA5={layer.era5 is not None}  TLE={layer.tle_loader is not None}")

    maps = {}
    for h in range(24):
        ts    = datetime(2025, 1, 1, h, 0, 0)
        label = f"{h:02d}utc"
        print(f"  {ts.strftime('%H:%M UTC')} ...", end=' ', flush=True)
        loss = layer.compute(ts)
        maps[label] = loss
        v = loss[~np.isnan(loss)]
        print(f"mean={v.mean():.2f} dB")
        plot_radio_map(loss,
                       title=f"L1 Loss — {ts.strftime('%Y-%m-%d %H:%M UTC')}",
                       output_file=str(out / f"l1_loss_{label}.png"),
                       cmap='viridis_r', origin_lat=ORIGIN_LAT,
                       origin_lon=ORIGIN_LON, coverage_km=COVERAGE_KM,
                       show_stats=True, dpi=120)

    # 4x6 panel
    print("  Generating 4x6 panel...")
    labels = [f"{h:02d}utc" for h in range(24)]
    all_v  = np.concatenate([maps[l][~np.isnan(maps[l])] for l in labels])
    vmin, vmax = all_v.min(), all_v.max()
    px, lats, lons = _geo_ticks(256, ORIGIN_LAT, ORIGIN_LON, COVERAGE_KM)

    fig, axes = plt.subplots(4, 6, figsize=(24, 16))
    fig.suptitle('L1 Macro Layer Loss — 2025-01-01 Hourly (10 GHz / 550 km LEO)',
                 fontsize=14, fontweight='bold')
    for ax, label in zip(axes.flat, labels):
        im = ax.imshow(maps[label], cmap='viridis_r', vmin=vmin, vmax=vmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(label.replace('utc', ':00 UTC'), fontsize=8, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        _overlay_stats(ax, maps[label])
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Loss (dB)', shrink=0.4)
    plt.tight_layout()
    fig.savefig(out / 'l1_loss_hourly_panel.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out / 'l1_loss_hourly_panel.png'}")


# ── 2. Rain rate scan ─────────────────────────────────────────────────────────

def run_rain_scan():
    out = Path('output/l1_maps/rain_scan')
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Rain rate scan (0 / 10 / 50 / 100 mm/h) ===")
    rain_rates = [0, 10, 50, 100]
    maps = {}
    for rr in rain_rates:
        cfg   = {**BASE_CONFIG, 'rain_rate_mm_h': rr}
        layer = L1MacroLayer(cfg, ORIGIN_LAT, ORIGIN_LON)
        layer.era5 = None   # force simplified model so rain term is visible
        loss  = layer.compute(TS_NOON)
        maps[rr] = loss
        v = loss[~np.isnan(loss)]
        print(f"  {rr:3d} mm/h -> mean={v.mean():.3f}  max={v.max():.3f} dB")

    px, lats, lons = _geo_ticks(256, ORIGIN_LAT, ORIGIN_LON, COVERAGE_KM)
    all_v  = np.concatenate([maps[r][~np.isnan(maps[r])] for r in rain_rates])
    vmin, vmax = all_v.min(), all_v.max()

    # Absolute panel
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Rain Rate Scan — L1 Total Loss (10 GHz / 550 km LEO / 12:00 UTC)',
                 fontsize=12, fontweight='bold')
    for ax, rr in zip(axes, rain_rates):
        im = ax.imshow(maps[rr], cmap='hot_r', vmin=vmin, vmax=vmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(f"{rr} mm/h", fontsize=11, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        _overlay_stats(ax, maps[rr])
    fig.colorbar(im, ax=axes.tolist(), label='Loss (dB)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(out / 'rain_scan_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out / 'rain_scan_panel.png'}")

    # Difference panel (relative to 0 mm/h baseline)
    base = maps[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Rain Attenuation Increment vs 0 mm/h (dB)', fontsize=12, fontweight='bold')
    diffs = [maps[rr] - base for rr in [10, 50, 100]]
    dmax  = max(np.nanmax(d) for d in diffs)
    for ax, rr, diff in zip(axes, [10, 50, 100], diffs):
        im = ax.imshow(diff, cmap='Reds', vmin=0, vmax=dmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(f"+{rr} mm/h increment", fontsize=11, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        v = diff[~np.isnan(diff)]
        ax.text(0.02, 0.02,
                f"mean={v.mean():.3f}\nmax={v.max():.3f} dB",
                transform=ax.transAxes, fontsize=7, va='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    fig.colorbar(im, ax=axes.tolist(), label='Additional Loss (dB)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(out / 'rain_scan_diff_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out / 'rain_scan_diff_panel.png'}")


# ── 3. Frequency scan ─────────────────────────────────────────────────────────

def run_freq_scan():
    out = Path('output/l1_maps/freq_scan')
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Frequency scan (1 / 3 / 10 / 30 GHz) ===")
    freqs = [1, 3, 10, 30]
    maps  = {}
    for f in freqs:
        cfg   = {**BASE_CONFIG, 'frequency_ghz': f}
        layer = L1MacroLayer(cfg, ORIGIN_LAT, ORIGIN_LON)
        loss  = layer.compute(TS_NOON)
        maps[f] = loss
        v = loss[~np.isnan(loss)]
        print(f"  {f:2d} GHz -> mean={v.mean():.2f}  min={v.min():.2f}  max={v.max():.2f} dB")

    px, lats, lons = _geo_ticks(256, ORIGIN_LAT, ORIGIN_LON, COVERAGE_KM)

    # Independent color scales (loss range differs ~29 dB across frequencies)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Frequency Scan — L1 Total Loss (550 km LEO / 12:00 UTC)',
                 fontsize=12, fontweight='bold')
    for ax, f in zip(axes, freqs):
        v    = maps[f][~np.isnan(maps[f])]
        im   = ax.imshow(maps[f], cmap='plasma_r',
                         vmin=v.min(), vmax=v.max(),
                         origin='upper', interpolation='nearest')
        ax.set_title(f"{f} GHz", fontsize=11, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        _overlay_stats(ax, maps[f])
        plt.colorbar(im, ax=ax, label='Loss (dB)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(out / 'freq_scan_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out / 'freq_scan_panel.png'}")


# ── 4. Component decomposition ────────────────────────────────────────────────

def run_component_decomposition():
    out = Path('output/l1_maps/components')
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Component decomposition (FSPL / atm / iono / total) ===")
    layer = L1MacroLayer(BASE_CONFIG, ORIGIN_LAT, ORIGIN_LON)
    comps = layer.compute_components(TS_NOON)

    labels = {
        'fspl':  ('Free Space Path Loss (FSPL)', 'Blues_r'),
        'atm':   ('Atmospheric Loss (ERA5-enhanced)', 'Greens_r'),
        'iono':  ('Ionospheric Loss (IONEX TEC)', 'Oranges_r'),
        'total': ('Total L1 Loss', 'viridis_r'),
    }

    for key, (title, cmap) in labels.items():
        arr = comps[key]
        v   = arr[~np.isnan(arr)]
        print(f"  {key:5s}: mean={v.mean():.4f}  min={v.min():.4f}  max={v.max():.4f} dB")
        plot_radio_map(arr,
                       title=f"{title} — 2025-01-01 12:00 UTC",
                       output_file=str(out / f"component_{key}.png"),
                       cmap=cmap, origin_lat=ORIGIN_LAT,
                       origin_lon=ORIGIN_LON, coverage_km=COVERAGE_KM,
                       show_stats=True, dpi=150)

    # 1x4 comparison panel
    print("  Generating component panel...")
    px, lats, lons = _geo_ticks(256, ORIGIN_LAT, ORIGIN_LON, COVERAGE_KM)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('L1 Loss Component Decomposition — 2025-01-01 12:00 UTC (10 GHz / 550 km LEO)',
                 fontsize=12, fontweight='bold')
    for ax, (key, (title, cmap)) in zip(axes, labels.items()):
        arr = comps[key]
        v   = arr[~np.isnan(arr)]
        im  = ax.imshow(arr, cmap=cmap, vmin=v.min(), vmax=v.max(),
                        origin='upper', interpolation='nearest')
        ax.set_title(title, fontsize=9, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        _overlay_stats(ax, arr)
        plt.colorbar(im, ax=ax, label='Loss (dB)', shrink=0.8)
    plt.tight_layout()
    fig.savefig(out / 'component_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {out / 'component_panel.png'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_hourly()
    run_rain_scan()
    run_freq_scan()
    run_component_decomposition()
    print("\nDone. All outputs saved to output/l1_maps/")


if __name__ == '__main__':
    main()
