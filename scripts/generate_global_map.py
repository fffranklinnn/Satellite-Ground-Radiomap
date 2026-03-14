#!/usr/bin/env python3
"""
Generate a global radio propagation map at 0.5-degree resolution.

Loads Starlink TLE data and computes the best-satellite (minimum loss) L1
loss map over the entire globe. For each ground pixel, finds the visible
satellite with the highest elevation angle and computes FSPL + atmospheric
+ ionospheric loss to that satellite.

Output:
  output/global_map/global_fspl.png
  output/global_map/global_atm.png
  output/global_map/global_iono.png
  output/global_map/global_total.png
  output/global_map/global_panel.png

Usage:
    conda run -n SatelliteRM python scripts/generate_global_map.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.core.physics import (
    free_space_path_loss,
    atmospheric_loss,
    ionospheric_loss,
)
from src.utils.ionex_loader import IonexLoader
from src.utils.tle_loader import TleLoader

# ── Configuration ─────────────────────────────────────────────────────────────

FREQ_GHZ      = 10.0
FREQ_GHZ_IONO = 1.0    # ionospheric loss visible below 3 GHz (zero at 10 GHz)
TS            = datetime(2025, 1, 1, 12, 0, 0)
RESOLUTION    = 0.5    # degrees per pixel
R_E           = 6371.0 # km
MIN_ELEV_DEG  = 5.0    # minimum elevation angle for a satellite to be "visible"

TLE_FILE   = 'data/2025_0101.tle'
IONEX_FILE = 'data/l1_space/data/cddis_data_2025/UPC0OPSRAP_20250010000_01D_15M_GIM.INX.gz'
OUT        = Path('output/global_map')

# ── Geometry ──────────────────────────────────────────────────────────────────

# ── Geometry ──────────────────────────────────────────────────────────────────

def calc_geometry(pixel_lats, pixel_lons, sat_lat, sat_lon, sat_alt_km):
    """
    Vectorized slant range and elevation angle from ground pixels to one satellite.

    Returns:
        slant_km  : (H, W) km
        elev_deg  : (H, W) degrees
    """
    lat1 = np.radians(pixel_lats)
    lon1 = np.radians(pixel_lons)
    lat2 = np.radians(sat_lat)
    lon2 = np.radians(sat_lon)

    cos_psi = np.clip(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1),
        -1.0, 1.0)
    psi = np.arccos(cos_psi)

    r_sat    = R_E + sat_alt_km
    slant_km = np.sqrt(R_E**2 + r_sat**2 - 2 * R_E * r_sat * np.cos(psi))

    sin_el  = (r_sat * np.cos(psi) - R_E) / np.where(slant_km > 0, slant_km, 1.0)
    elev_deg = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    return slant_km, elev_deg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Build global lat/lon grids
    lats_1d = np.arange( 90.0 - RESOLUTION/2, -90.0, -RESOLUTION)
    lons_1d = np.arange(-180.0 + RESOLUTION/2, 180.0,  RESOLUTION)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
    H, W = lats_2d.shape
    print(f"Global grid: {H}x{W} pixels at {RESOLUTION} deg resolution")

    # Load TLE constellation
    print(f"Loading TLE from {TLE_FILE} ...")
    try:
        tle = TleLoader(TLE_FILE, inc_min=52.9, inc_max=53.3)
        sat_lats, sat_lons, sat_alts = tle.get_geodetic(TS)
        # Filter to valid LEO altitudes
        mask = (sat_alts >= 300.0) & (sat_alts <= 700.0)
        sat_lats, sat_lons, sat_alts = sat_lats[mask], sat_lons[mask], sat_alts[mask]
        print(f"  {len(sat_lats)} satellites after altitude filter")
    except Exception as e:
        print(f"  TLE unavailable ({e}), falling back to single zenith satellite")
        sat_lats = np.array([0.0])
        sat_lons = np.array([0.0])
        sat_alts = np.array([550.0])

    # IONEX TEC
    print("Loading IONEX TEC...")
    try:
        ionex    = IonexLoader(IONEX_FILE)
        epoch_sec = TS.hour * 3600 + TS.minute * 60 + TS.second
        tec      = ionex.get_tec(epoch_sec, lats_2d, lons_2d)
        print(f"  TEC range: {tec.min():.1f} - {tec.max():.1f} TECU (mean={tec.mean():.1f})")
    except Exception as e:
        print(f"  IONEX unavailable ({e}), using default TEC=10 TECU")
        tec = np.full_like(lats_2d, 10.0)

    # Best-satellite loss: iterate over constellation, keep minimum total loss per pixel
    print(f"Computing best-satellite loss over {len(sat_lats)} satellites ...")
    best_total = np.full((H, W), np.inf)
    best_fspl  = np.full((H, W), np.nan)
    best_atm   = np.full((H, W), np.nan)
    best_elev  = np.full((H, W), np.nan)

    BATCH = 50
    for start in range(0, len(sat_lats), BATCH):
        sl = slice(start, start + BATCH)
        for slat, slon, salt in zip(sat_lats[sl], sat_lons[sl], sat_alts[sl]):
            slant, elev = calc_geometry(lats_2d, lons_2d, slat, slon, salt)
            visible = elev > MIN_ELEV_DEG
            if not np.any(visible):
                continue
            fspl = free_space_path_loss(slant, FREQ_GHZ)
            atm  = atmospheric_loss(elev, FREQ_GHZ, rain_rate_mm_h=0.0)
            tot  = np.where(visible, fspl + atm, np.inf)
            improved = tot < best_total
            best_total = np.where(improved, tot,  best_total)
            best_fspl  = np.where(improved, fspl, best_fspl)
            best_atm   = np.where(improved, atm,  best_atm)
            best_elev  = np.where(improved, elev, best_elev)
        if (start // BATCH) % 5 == 0:
            covered = np.isfinite(best_total).sum()
            print(f"  [{start+BATCH}/{len(sat_lats)}] covered pixels: "
                  f"{covered}/{H*W} ({100*covered/(H*W):.1f}%)")

    # Pixels with no visible satellite -> NaN
    best_total = np.where(np.isinf(best_total), np.nan, best_total)

    # Ionospheric loss at 1 GHz using best-satellite elevation
    iono = np.where(np.isfinite(best_elev),
                    ionospheric_loss(FREQ_GHZ_IONO, tec), np.nan)

    covered = np.isfinite(best_total).sum()
    print(f"Coverage: {covered}/{H*W} pixels ({100*covered/(H*W):.1f}%)")
    for label, arr in [('FSPL', best_fspl), ('Atm', best_atm),
                        ('Iono', iono), ('Total', best_total)]:
        v = arr[~np.isnan(arr)]
        if len(v):
            print(f"  {label:5s}: mean={v.mean():.3f}  std={v.std():.3f}  "
                  f"min={v.min():.3f}  max={v.max():.3f} dB")

    # ── Plotting ──────────────────────────────────────────────────────────────

    extent = [-180, 180, -90, 90]
    maps = [
        ('global_fspl',  best_fspl,  f'Best-Sat FSPL ({FREQ_GHZ} GHz)',              'Blues_r'),
        ('global_atm',   best_atm,   f'Best-Sat Atmospheric Loss ({FREQ_GHZ} GHz)',   'Greens_r'),
        ('global_iono',  iono,       f'Ionospheric Loss ({FREQ_GHZ_IONO} GHz, IONEX)','Oranges_r'),
        ('global_total', best_total, f'Best-Sat Total Loss ({FREQ_GHZ} GHz)',          'viridis_r'),
    ]

    sat_info = f"Starlink TLE {TS.strftime('%Y-%m-%d %H:%M UTC')} | {len(sat_lats)} sats"
    for fname, arr, title, cmap in maps:
        fig, ax = plt.subplots(figsize=(16, 8))
        v  = arr[~np.isnan(arr)]
        im = ax.imshow(arr, cmap=cmap, vmin=v.min(), vmax=v.max(),
                       extent=extent, origin='upper',
                       interpolation='bilinear', aspect='auto')
        ax.set_title(f"{title}\n{sat_info}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (deg)', fontsize=10)
        ax.set_ylabel('Latitude (deg)', fontsize=10)
        ax.set_xticks(range(-180, 181, 30))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label('Loss (dB)', fontsize=10)
        ax.text(0.01, 0.02,
                f"mean={v.mean():.2f}  std={v.std():.2f}  "
                f"min={v.min():.2f}  max={v.max():.2f} dB",
                transform=ax.transAxes, fontsize=8, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        fig.savefig(OUT / f"{fname}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved -> {OUT / fname}.png")

    # ── 2x2 panel ─────────────────────────────────────────────────────────────
    print("Generating 2x2 panel...")
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    fig.suptitle(f'Global L1 Loss — {sat_info} | '
                 f'FSPL/Atm: {FREQ_GHZ} GHz  Iono: {FREQ_GHZ_IONO} GHz',
                 fontsize=11, fontweight='bold')
    for ax, (_, arr, title, cmap) in zip(axes.flat, maps):
        v  = arr[~np.isnan(arr)]
        im = ax.imshow(arr, cmap=cmap, vmin=v.min(), vmax=v.max(),
                       extent=extent, origin='upper',
                       interpolation='bilinear', aspect='auto')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Longitude (deg)', fontsize=8)
        ax.set_ylabel('Latitude (deg)', fontsize=8)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        plt.colorbar(im, ax=ax, label='dB', shrink=0.8)
        ax.text(0.01, 0.02,
                f"mean={v.mean():.2f}  max={v.max():.2f} dB",
                transform=ax.transAxes, fontsize=7, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    fig.savefig(OUT / 'global_panel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {OUT / 'global_panel.png'}")
    print("Done.")


if __name__ == '__main__':
    main()
