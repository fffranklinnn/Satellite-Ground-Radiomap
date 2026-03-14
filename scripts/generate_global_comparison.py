#!/usr/bin/env python3
"""
Generate L1 radio maps for 6 globally distributed regions.

Compares how IONEX TEC and ERA5 IWV vary across different latitudes/climates,
showing the physical effects that are invisible in a single 256 km local patch.

Regions:
  Beijing    (39.9N, 116.4E) — mid-latitude baseline
  Singapore  ( 1.3N, 103.8E) — equatorial, high TEC, high IWV
  Moscow     (55.7N,  37.6E) — high latitude, low TEC
  Arctic     (70.0N,  25.0E) — polar TEC disturbance zone
  Sao Paulo  (23.5S,  46.6W) — South Atlantic Anomaly region
  Dubai      (25.2N,  55.3E) — arid, low IWV

Usage:
    conda run -n SatelliteRM python scripts/generate_global_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.layers.l1_macro import L1MacroLayer
from src.utils.plotter import plot_radio_map

# ── Configuration ─────────────────────────────────────────────────────────────

REGIONS = [
    ('Beijing',   39.9042,  116.4074),
    ('Singapore',  1.3521,  103.8198),
    ('Moscow',    55.7558,   37.6173),
    ('Arctic',    70.0000,   25.0000),
    ('Sao Paulo', -23.5505, -46.6333),
    ('Dubai',     25.2048,   55.2708),
]

COVERAGE_KM = 256.0
TS = datetime(2025, 1, 1, 12, 0, 0)

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

OUT = Path('output/global_comparison')

# ── Helpers ───────────────────────────────────────────────────────────────────

def _geo_ticks(n, origin_lat, origin_lon, coverage_km, n_ticks=5):
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    px   = np.linspace(0, n - 1, n_ticks)
    lats = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, n_ticks)
    lons = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, n_ticks)
    return px, lats, lons


def _add_geo_ticks(ax, px, lats, lons):
    ax.set_xticks(px)
    ax.set_xticklabels([f"{v:.1f}E" for v in lons], fontsize=6)
    ax.set_yticks(px)
    ax.set_yticklabels([f"{v:.1f}N" for v in lats], fontsize=6)


def _overlay_stats(ax, arr, extra=''):
    v = arr[~np.isnan(arr)]
    txt = f"mean={v.mean():.2f}\nstd={v.std():.2f}\nmin={v.min():.2f}\nmax={v.max():.2f} dB"
    if extra:
        txt = extra + '\n' + txt
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6,
            va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}   # name -> {'total', 'fspl', 'atm', 'iono', 'tec_mean', 'iwv_mean'}

    print(f"Timestamp: {TS.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'Region':<12} {'Lat':>7} {'Lon':>8}  {'TEC(TECU)':>10}  {'IWV(kg/m2)':>11}  "
          f"{'FSPL(dB)':>9}  {'Atm(dB)':>8}  {'Total(dB)':>10}")
    print('-' * 85)

    for name, lat, lon in REGIONS:
        layer = L1MacroLayer(BASE_CONFIG, lat, lon)
        comps = layer.compute_components(TS)

        # Sample TEC and IWV at region center for reporting
        epoch_sec = TS.hour * 3600
        tec_center = (layer.ionex.get_tec(epoch_sec,
                      np.array([[lat]]), np.array([[lon]]))[0, 0]
                      if layer.ionex else float('nan'))
        iwv_center = (layer.era5.get_iwv(TS.hour,
                      np.array([[lat]]), np.array([[lon]]))[0, 0]
                      if layer.era5 else float('nan'))

        results[name] = {**comps, 'lat': lat, 'lon': lon,
                         'tec_center': tec_center, 'iwv_center': iwv_center}

        v_total = comps['total'][~np.isnan(comps['total'])]
        v_fspl  = comps['fspl'][~np.isnan(comps['fspl'])]
        v_atm   = comps['atm'][~np.isnan(comps['atm'])]

        era5_note = '' if layer.era5 else ' [ERA5 fallback]'
        print(f"{name:<12} {lat:>7.2f} {lon:>8.2f}  {tec_center:>10.2f}  "
              f"{iwv_center:>11.3f}  {v_fspl.mean():>9.2f}  "
              f"{v_atm.mean():>8.4f}  {v_total.mean():>10.2f}{era5_note}")

        # Individual total loss map
        plot_radio_map(comps['total'],
                       title=f"L1 Total Loss — {name} (10 GHz / 550 km / 12:00 UTC)",
                       output_file=str(OUT / f"{name.lower().replace(' ', '_')}_total.png"),
                       cmap='viridis_r', origin_lat=lat, origin_lon=lon,
                       coverage_km=COVERAGE_KM, show_stats=True, dpi=120)

        # Component panel for this region
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle(f'L1 Loss Components — {name} ({lat:.1f}°, {lon:.1f}°) | '
                     f'TEC={tec_center:.1f} TECU  IWV={iwv_center:.2f} kg/m²',
                     fontsize=10, fontweight='bold')
        comp_items = [
            ('fspl',  'FSPL',         'Blues_r'),
            ('atm',   'Atmospheric',  'Greens_r'),
            ('iono',  'Ionospheric',  'Oranges_r'),
            ('total', 'Total',        'viridis_r'),
        ]
        px, lats_t, lons_t = _geo_ticks(256, lat, lon, COVERAGE_KM)
        for ax, (key, title, cmap) in zip(axes, comp_items):
            arr = comps[key]
            v   = arr[~np.isnan(arr)]
            im  = ax.imshow(arr, cmap=cmap, vmin=v.min(), vmax=v.max(),
                            origin='upper', interpolation='nearest')
            ax.set_title(title, fontsize=9, fontweight='bold')
            _add_geo_ticks(ax, px, lats_t, lons_t)
            _overlay_stats(ax, arr)
            plt.colorbar(im, ax=ax, label='dB', shrink=0.8)
        plt.tight_layout()
        fig.savefig(OUT / f"{name.lower().replace(' ', '_')}_components.png",
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── 2×3 global comparison panel (total loss) ──────────────────────────────
    print("\nGenerating 2x3 comparison panel...")
    all_v = np.concatenate([
        results[n]['total'][~np.isnan(results[n]['total'])]
        for n, _, _ in REGIONS
    ])
    vmin, vmax = all_v.min(), all_v.max()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('L1 Total Loss — Global Region Comparison (10 GHz / 550 km LEO / 2025-01-01 12:00 UTC)',
                 fontsize=13, fontweight='bold')
    for ax, (name, lat, lon) in zip(axes.flat, REGIONS):
        arr = results[name]['total']
        im  = ax.imshow(arr, cmap='viridis_r', vmin=vmin, vmax=vmax,
                        origin='upper', interpolation='nearest')
        tec = results[name]['tec_center']
        iwv = results[name]['iwv_center']
        ax.set_title(f"{name}  ({lat:.1f}°, {lon:.1f}°)", fontsize=10, fontweight='bold')
        px, lats_t, lons_t = _geo_ticks(256, lat, lon, COVERAGE_KM)
        _add_geo_ticks(ax, px, lats_t, lons_t)
        _overlay_stats(ax, arr, extra=f"TEC={tec:.1f} TECU\nIWV={iwv:.2f} kg/m²")
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Total Loss (dB)', shrink=0.5)
    plt.tight_layout()
    fig.savefig(OUT / 'global_comparison_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {OUT / 'global_comparison_panel.png'}")
    print("Done.")


if __name__ == '__main__':
    main()
