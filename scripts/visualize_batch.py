#!/usr/bin/env python3
"""
Visualize all .npy radiomap files in output/batch/ as PNG images.

For each task group produces:
  - Individual PNG per .npy (same directory, same stem)
  - Summary panel PNG in the task directory

Usage:
    conda run -n SatelliteRM python scripts/visualize_batch.py
    python scripts/visualize_batch.py --batch-dir output/batch
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Task metadata ─────────────────────────────────────────────────────────────
# key: directory prefix (first 3 chars)
# value: (title, cmap, coverage_km, origin_lat, origin_lon, shared_scale)
TASK_META = {
    'G01': ('L1 Macro — Hourly (Xi\'an)',        'viridis_r', 256.0,  34.3416, 108.9398, True),
    'G02': ('L1 Macro — Multi-day (Xi\'an)',      'viridis_r', 256.0,  34.3416, 108.9398, True),
    'G03': ('L1 Macro — Frequency Sweep',         'plasma_r',  256.0,  34.3416, 108.9398, False),
    'G04': ('L1 Macro — Rain Rate Sweep',         'hot_r',     256.0,  34.3416, 108.9398, True),
    'G06': ('L2 Terrain — Angle Sweep (Xi\'an)',  'Blues_r',   25.6,   34.3416, 108.9398, True),
    'G08': ('L3 Urban — Angle Sweep (Xi\'an)',    'Reds',      0.256,  34.3416, 108.9398, True),
}

ELEVATIONS = [15, 30, 45, 60, 75]
AZIMUTHS   = [0, 45, 90, 135, 180, 225, 270, 315]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _geo_ticks(n, origin_lat, origin_lon, coverage_km, n_ticks=5):
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    px   = np.linspace(0, n - 1, n_ticks)
    lats = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, n_ticks)
    lons = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, n_ticks)
    return px, lats, lons


def _add_geo_ticks(ax, px, lats, lons, fontsize=7):
    ax.set_xticks(px)
    ax.set_xticklabels([f"{v:.2f}E" for v in lons], fontsize=fontsize)
    ax.set_yticks(px)
    ax.set_yticklabels([f"{v:.2f}N" for v in lats], fontsize=fontsize)


def _stats_text(arr):
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return "no data"
    return f"μ={v.mean():.1f}\nσ={v.std():.1f}\n[{v.min():.1f},{v.max():.1f}] dB"


def _overlay_stats(ax, arr, fontsize=6):
    ax.text(0.02, 0.02, _stats_text(arr), transform=ax.transAxes,
            fontsize=fontsize, va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))


def _save_individual(arr, out_path, title, cmap, coverage_km,
                     origin_lat, origin_lon, vmin=None, vmax=None):
    """Save a single radiomap as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    v = arr[~np.isnan(arr)]
    _vmin = float(v.min()) if vmin is None and len(v) > 0 else vmin
    _vmax = float(v.max()) if vmax is None and len(v) > 0 else vmax
    if _vmin == _vmax:
        _vmin -= 0.5; _vmax += 0.5

    im = ax.imshow(arr, cmap=cmap, vmin=_vmin, vmax=_vmax,
                   origin='upper', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Loss (dB)', fontsize=9)

    px, lats, lons = _geo_ticks(arr.shape[0], origin_lat, origin_lon, coverage_km)
    _add_geo_ticks(ax, px, lats, lons, fontsize=8)
    _overlay_stats(ax, arr, fontsize=8)

    ax.set_title(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Per-task panel generators ─────────────────────────────────────────────────

def _panel_grid(maps, labels, title, cmap, coverage_km, origin_lat, origin_lon,
                ncols, out_path, shared_scale=True, dpi=120):
    """Generic N-up panel: maps is list of (label, arr)."""
    n = len(maps)
    nrows = (n + ncols - 1) // ncols

    all_v = np.concatenate([a[~np.isnan(a)].ravel() for _, a in maps if a is not None])
    g_vmin = float(all_v.min()) if len(all_v) > 0 else 0
    g_vmax = float(all_v.max()) if len(all_v) > 0 else 1
    if g_vmin == g_vmax:
        g_vmin -= 0.5; g_vmax += 0.5

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.2, nrows * 3.0 + 0.6),
                             squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    px, lats, lons = _geo_ticks(256, origin_lat, origin_lon, coverage_km)

    for idx, (lbl, arr) in enumerate(maps):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        vmin = g_vmin if shared_scale else float(arr[~np.isnan(arr)].min())
        vmax = g_vmax if shared_scale else float(arr[~np.isnan(arr)].max())
        if vmin == vmax:
            vmin -= 0.5; vmax += 0.5
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(lbl, fontsize=8, fontweight='bold')
        _add_geo_ticks(ax, px, lats, lons)
        _overlay_stats(ax, arr)
        if not shared_scale:
            fig.colorbar(im, ax=ax, shrink=0.75, label='dB')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    if shared_scale:
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Loss (dB)', shrink=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Panel -> {out_path}")


def _panel_angle_sweep(maps_dict, title, cmap, coverage_km,
                       origin_lat, origin_lon, out_path, dpi=120):
    """5×8 panel: rows=elevation, cols=azimuth."""
    els = ELEVATIONS
    azs = AZIMUTHS

    all_v = np.concatenate([a[~np.isnan(a)].ravel()
                             for a in maps_dict.values() if a is not None])
    g_vmin = float(all_v.min()) if len(all_v) > 0 else 0
    g_vmax = float(all_v.max()) if len(all_v) > 0 else 1
    if g_vmin == g_vmax:
        g_vmin -= 0.5; g_vmax += 0.5

    fig, axes = plt.subplots(len(els), len(azs),
                             figsize=(len(azs) * 2.8, len(els) * 2.8 + 0.8),
                             squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    px, lats, lons = _geo_ticks(256, origin_lat, origin_lon, coverage_km)

    for ri, el in enumerate(els):
        for ci, az in enumerate(azs):
            ax = axes[ri][ci]
            key = (el, az)
            arr = maps_dict.get(key)
            if arr is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(arr, cmap=cmap, vmin=g_vmin, vmax=g_vmax,
                           origin='upper', interpolation='nearest')
            ax.set_title(f"el{el} az{az}", fontsize=7, fontweight='bold')
            _add_geo_ticks(ax, px, lats, lons, fontsize=5)
            _overlay_stats(ax, arr, fontsize=5)

        # Row label
        axes[ri][0].set_ylabel(f"el={el}°", fontsize=8, fontweight='bold')

    # Column labels
    for ci, az in enumerate(azs):
        axes[0][ci].set_xlabel(f"az={az}°", fontsize=8)
        axes[0][ci].xaxis.set_label_position('top')

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Loss (dB)', shrink=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Panel -> {out_path}")


# ── Task processors ───────────────────────────────────────────────────────────

def process_g01(task_dir, meta):
    """G01: L1 hourly — 24 maps, 4×6 panel."""
    title, cmap, cov, lat, lon, shared = meta
    files = sorted(task_dir.glob('l1_h*.npy'))
    if not files:
        return
    print(f"\n[G01] {len(files)} hourly maps")

    maps = []
    for f in files:
        arr = np.load(f)
        hour = f.stem.replace('l1_h', '')
        lbl  = f"{hour}:00 UTC"
        png  = f.with_suffix('.png')
        _save_individual(arr, png, f"L1 Hourly {lbl}", cmap, cov, lat, lon)
        print(f"  {f.name} -> {png.name}")
        maps.append((lbl, arr))

    _panel_grid(maps, [m[0] for m in maps],
                f"{title} — 2025-01-01",
                cmap, cov, lat, lon, ncols=6,
                out_path=task_dir / 'panel_hourly.png', shared_scale=shared)


def process_g02(task_dir, meta):
    """G02: L1 multi-day — variable count, 3×N panel."""
    title, cmap, cov, lat, lon, shared = meta
    files = sorted(task_dir.glob('l1_doy*.npy'))
    if not files:
        return
    print(f"\n[G02] {len(files)} multi-day maps")

    maps = []
    for f in files:
        arr = np.load(f)
        lbl = f.stem.replace('l1_', '')
        png = f.with_suffix('.png')
        _save_individual(arr, png, f"L1 {lbl}", cmap, cov, lat, lon)
        print(f"  {f.name} -> {png.name}")
        maps.append((lbl, arr))

    ncols = min(5, len(maps))
    _panel_grid(maps, [m[0] for m in maps],
                f"{title}",
                cmap, cov, lat, lon, ncols=ncols,
                out_path=task_dir / 'panel_multiday.png', shared_scale=shared)


def process_g03(task_dir, meta):
    """G03: L1 frequency sweep — 6 bands, independent scales."""
    title, cmap, cov, lat, lon, shared = meta
    files = sorted(task_dir.glob('l1_*.npy'))
    if not files:
        return
    print(f"\n[G03] {len(files)} frequency maps")

    maps = []
    for f in files:
        arr = np.load(f)
        lbl = f.stem.replace('l1_', '')
        png = f.with_suffix('.png')
        _save_individual(arr, png, f"L1 {lbl}", cmap, cov, lat, lon)
        print(f"  {f.name} -> {png.name}")
        maps.append((lbl, arr))

    _panel_grid(maps, [m[0] for m in maps],
                f"{title}",
                cmap, cov, lat, lon, ncols=len(maps),
                out_path=task_dir / 'panel_freq_sweep.png', shared_scale=False)


def process_g04(task_dir, meta):
    """G04: L1 rain sweep — 6 rates, shared scale."""
    title, cmap, cov, lat, lon, shared = meta
    files = sorted(task_dir.glob('l1_rain*.npy'),
                   key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or '0'))
    if not files:
        return
    print(f"\n[G04] {len(files)} rain rate maps")

    maps = []
    for f in files:
        arr = np.load(f)
        lbl = f.stem.replace('l1_', '')
        png = f.with_suffix('.png')
        _save_individual(arr, png, f"L1 {lbl}", cmap, cov, lat, lon)
        print(f"  {f.name} -> {png.name}")
        maps.append((lbl, arr))

    _panel_grid(maps, [m[0] for m in maps],
                f"{title}",
                cmap, cov, lat, lon, ncols=len(maps),
                out_path=task_dir / 'panel_rain_sweep.png', shared_scale=shared)


def process_angle_sweep(task_dir, meta, prefix, panel_name, task_tag):
    """Generic angle sweep processor for G06 (L2) and G08 (L3)."""
    title, cmap, cov, lat, lon, shared = meta
    files = list(task_dir.glob(f'{prefix}_el*_az*.npy'))
    if not files:
        return
    print(f"\n[{task_tag}] {len(files)} angle maps")

    maps_dict = {}
    for f in sorted(files):
        arr  = np.load(f)
        stem = f.stem  # e.g. l2_el45_az180
        parts = stem.split('_')
        el = int(parts[-2].replace('el', ''))
        az = int(parts[-1].replace('az', ''))
        lbl = f"el{el}° az{az}°"
        png = f.with_suffix('.png')
        _save_individual(arr, png, f"{task_tag} {lbl}", cmap, cov, lat, lon)
        print(f"  {f.name} -> {png.name}")
        maps_dict[(el, az)] = arr

    _panel_angle_sweep(maps_dict, f"{title}",
                       cmap, cov, lat, lon,
                       out_path=task_dir / panel_name)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize batch .npy radiomaps as PNG')
    parser.add_argument('--batch-dir', default='output/batch',
                        help='Root batch output directory (default: output/batch)')
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"ERROR: batch dir not found: {batch_dir}")
        sys.exit(1)

    total_png = 0

    for task_dir in sorted(batch_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        key = task_dir.name[:3].upper()
        meta = TASK_META.get(key)
        if meta is None:
            continue

        if key == 'G01':
            process_g01(task_dir, meta)
        elif key == 'G02':
            process_g02(task_dir, meta)
        elif key == 'G03':
            process_g03(task_dir, meta)
        elif key == 'G04':
            process_g04(task_dir, meta)
        elif key == 'G06':
            process_angle_sweep(task_dir, meta, 'l2', 'panel_angle_sweep.png', 'G06-L2')
        elif key == 'G08':
            process_angle_sweep(task_dir, meta, 'l3', 'panel_angle_sweep.png', 'G08-L3')

        total_png += len(list(task_dir.glob('*.png')))

    print(f"\nDone. {total_png} PNG files written under {batch_dir}/")


if __name__ == '__main__':
    import argparse
    main()
