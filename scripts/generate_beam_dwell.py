#!/usr/bin/env python3
"""
波束凝视仿真脚本

选定一颗卫星，在它过境期间持续将波束对准地面中心点（beam dwell/staring），
每隔固定时间步长生成一张 L1+L2 接收功率图，输出图序列和汇总动画。

用法：
  python scripts/generate_beam_dwell.py
  python scripts/generate_beam_dwell.py --norad 57360
  python scripts/generate_beam_dwell.py --norad 57360 --start 2025-01-01T00:00:00 --step 60 --output output/dwell
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['Noto Sans CJK JP', 'DejaVu Sans Mono']
import matplotlib.pyplot as plt

from src.layers.l1_macro import L1MacroLayer
from src.layers.l2_topo import L2TopoLayer
from src.layers.base import LayerContext


# ---------------------------------------------------------------------------
# 找过境弧段：从 start_time 开始扫描，找到该卫星仰角 >= min_el 的连续时段
# ---------------------------------------------------------------------------

def find_pass_window(l1: L1MacroLayer,
                     norad_id: str,
                     origin_lat: float,
                     origin_lon: float,
                     start_time: datetime,
                     min_el_deg: float = 5.0,
                     scan_hours: float = 24.0,
                     scan_step_s: int = 30) -> list:
    """
    扫描 scan_hours 小时，找到 norad_id 卫星仰角 >= min_el_deg 的所有时刻。
    返回 [(datetime, el_deg, az_deg), ...] 列表。
    """
    from skyfield.api import wgs84
    observer = wgs84.latlon(origin_lat, origin_lon)

    # 找到目标卫星
    target_sat = None
    for sat, tg in zip(l1.satellites, l1.tle_groups):
        from src.layers.l1_macro import _get_norad_id
        if _get_norad_id(tg) == str(norad_id):
            target_sat = sat
            break

    if target_sat is None:
        raise ValueError(f"NORAD {norad_id} not found in TLE file")

    from skyfield.api import load
    ts = load.timescale()

    visible = []
    t = start_time
    end_time = start_time + timedelta(hours=scan_hours)

    while t <= end_time:
        sky_t = ts.from_datetime(t)
        diff = target_sat - observer
        topo = diff.at(sky_t)
        alt, az, _ = topo.altaz()
        el = alt.degrees
        if el >= min_el_deg:
            visible.append((t, el, az.degrees))
        t += timedelta(seconds=scan_step_s)

    return visible


# ---------------------------------------------------------------------------
# DEM 高程图
# ---------------------------------------------------------------------------

def _plot_dem(dem: np.ndarray,
              origin_lat: float,
              origin_lon: float,
              coverage_km: float,
              output_file: str,
              dpi: int = 150) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"DEM 高程图  |  覆盖 {coverage_km} km\n"
        f"({origin_lat:.3f}°N, {origin_lon:.3f}°E)",
        fontsize=11, fontweight='bold'
    )
    n = dem.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)

    im = ax.imshow(dem, cmap='terrain', origin='upper', interpolation='bilinear')
    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=8, rotation=30)
    ax.set_yticks(tick_px)
    ax.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('高程 (m)', fontsize=9)
    ax.text(0.02, 0.02,
            f"min={dem.min():.0f} m\nmax={dem.max():.0f} m\nmean={dem.mean():.0f} m",
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[dem] DEM 高程图已保存: {output_file}")


def _plot_dem_rsrp(dem: np.ndarray,
                   rx_power: np.ndarray,
                   rx_sensitivity_dbm: float,
                   origin_lat: float,
                   origin_lon: float,
                   coverage_km: float,
                   timestamp: datetime,
                   el_deg: float,
                   az_deg: float,
                   norad_id: str,
                   frame_idx: int,
                   output_file: str,
                   dpi: int = 120) -> None:
    n = dem.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)

    rx_display = rx_power.copy().astype(float)
    rx_display[rx_power < rx_sensitivity_dbm] = np.nan
    coverage_pct = 100.0 * np.sum(rx_power >= rx_sensitivity_dbm) / rx_power.size
    vmax = float(np.nanmax(rx_display)) if not np.all(np.isnan(rx_display)) else -40.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(
        f"波束凝视 NORAD {norad_id}  |  帧 {frame_idx:03d}  |  "
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"el={el_deg:.1f}°  az={az_deg:.1f}°   "
        f"({origin_lat:.3f}°N, {origin_lon:.3f}°E)   覆盖 {coverage_km} km",
        fontsize=11, fontweight='bold'
    )

    # 左图：DEM
    ax0 = axes[0]
    im0 = ax0.imshow(dem, cmap='terrain', origin='upper', interpolation='bilinear')
    ax0.set_title('DEM 地形高程 (m)', fontsize=10, fontweight='bold')
    ax0.set_xticks(tick_px)
    ax0.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=7, rotation=30)
    ax0.set_yticks(tick_px)
    ax0.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=7)
    cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.set_label('高程 (m)', fontsize=9)
    ax0.text(0.02, 0.02,
             f"min={dem.min():.0f} m\nmax={dem.max():.0f} m",
             transform=ax0.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 右图：接收功率 + DEM 等高线
    ax1 = axes[1]
    cmap_rsrp = plt.cm.jet.copy()
    cmap_rsrp.set_bad(color='#303030')
    im1 = ax1.imshow(rx_display, cmap=cmap_rsrp, vmin=rx_sensitivity_dbm, vmax=vmax,
                     origin='upper', interpolation='nearest')
    ax1.set_title('接收功率 RSRP (dBm)', fontsize=10, fontweight='bold')
    ax1.set_xticks(tick_px)
    ax1.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=7, rotation=30)
    ax1.set_yticks(tick_px)
    ax1.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=7)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('RSRP (dBm)', fontsize=9)
    ax1.text(0.02, 0.02,
             f"覆盖率={coverage_pct:.1f}%\n"
             f"最大={vmax:.1f} dBm\n"
             f"阈值={rx_sensitivity_dbm:.0f} dBm（灰=无覆盖）",
             transform=ax1.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 单帧绘图
# ---------------------------------------------------------------------------

def plot_frame(rx_power: np.ndarray,
               rx_sensitivity_dbm: float,
               origin_lat: float,
               origin_lon: float,
               coverage_km: float,
               timestamp: datetime,
               el_deg: float,
               az_deg: float,
               norad_id: str,
               frame_idx: int,
               output_file: str,
               dpi: int = 120) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    n = rx_power.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))

    display = rx_power.copy().astype(float)
    display[rx_power < rx_sensitivity_dbm] = np.nan

    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='#404040')
    vmax = float(np.nanmax(display)) if not np.all(np.isnan(display)) else -40.0
    vmin = rx_sensitivity_dbm

    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='upper', interpolation='nearest')

    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)
    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=7, rotation=30)
    ax.set_yticks(tick_px)
    ax.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=7)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('接收功率 (dBm)', fontsize=9)

    coverage_pct = 100.0 * np.sum(rx_power >= rx_sensitivity_dbm) / rx_power.size
    ax.text(0.02, 0.02,
            f"el={el_deg:.1f}°  az={az_deg:.1f}°\n"
            f"max={vmax:.1f} dBm\n"
            f"覆盖率={coverage_pct:.1f}%",
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    fig.suptitle(
        f"波束凝视 NORAD {norad_id}  |  帧 {frame_idx:03d}\n"
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  "
        f"({origin_lat:.3f}°N, {origin_lon:.3f}°E)",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(config_path: str,
        norad_id: str,
        start_time: datetime,
        step_s: int,
        output_dir: Path,
        min_el_deg: float = 5.0) -> None:

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    origin_lat = cfg['origin']['latitude']
    origin_lon = cfg['origin']['longitude']
    coverage_km = float(cfg['layers']['l1_macro'].get('coverage_km', 256.0))
    lb = cfg.get('link_budget', {})
    tx_power_dbw      = float(lb.get('tx_power_dbw', 20.0))
    rx_gain_dbi       = float(lb.get('rx_gain_dbi', 33.0))
    rx_sensitivity_dbw = float(lb.get('rx_sensitivity_dbw', -130.0))
    rx_sensitivity_dbm = rx_sensitivity_dbw + 30.0

    print(f"[config] 原点: ({origin_lat}°N, {origin_lon}°E)")
    print(f"[config] 目标卫星: NORAD {norad_id}")
    print(f"[config] 扫描起始: {start_time.isoformat()}")

    print("\n[init] 初始化 L1/L2 层...")
    l1 = L1MacroLayer(cfg['layers']['l1_macro'], origin_lat, origin_lon)
    l1.set_time(start_time)
    l2 = L2TopoLayer(cfg['layers']['l2_topo'], origin_lat, origin_lon)

    print("\n[scan] 扫描过境弧段...")
    visible = find_pass_window(
        l1, norad_id, origin_lat, origin_lon,
        start_time, min_el_deg=min_el_deg,
        scan_hours=24.0, scan_step_s=step_s
    )

    if not visible:
        print(f"[warn] 在 24 小时内未找到 NORAD {norad_id} 仰角 >= {min_el_deg}° 的过境")
        return

    # 找第一个连续弧段
    arc = [visible[0]]
    for i in range(1, len(visible)):
        dt = (visible[i][0] - visible[i-1][0]).total_seconds()
        if dt <= step_s * 2:
            arc.append(visible[i])
        else:
            break  # 只取第一段

    print(f"[scan] 找到过境弧段: {len(arc)} 帧，"
          f"{arc[0][0].strftime('%H:%M:%S')} ~ {arc[-1][0].strftime('%H:%M:%S')} UTC")
    print(f"[scan] 仰角范围: {min(a[1] for a in arc):.1f}° ~ {max(a[1] for a in arc):.1f}°")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 DEM 高程图（只需一次，地形不随时间变化）
    print("\n[dem] 生成 DEM 高程图...")
    dem_patch = l2.get_dem_patch(origin_lat, origin_lon)
    np.save(output_dir / "dem_patch.npy", dem_patch)
    _plot_dem(dem_patch, origin_lat, origin_lon, coverage_km,
              str(output_dir / "dem_patch.png"))

    frame_files = []

    for idx, (ts, el, az) in enumerate(arc):
        print(f"\n[frame {idx:03d}] {ts.strftime('%H:%M:%S')} | el={el:.1f}° az={az:.1f}°")

        context = LayerContext(extras={'target_norad_ids': [norad_id]})
        l1_components = l1.compute_components(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            timestamp=ts,
            context=context,
        )
        l1_map = l1_components['total']
        sat_info = l1_components['satellite']

        l2_context = LayerContext(extras={
            'satellite_elevation_deg':  sat_info['elevation_deg'],
            'satellite_azimuth_deg':    sat_info['azimuth_deg'],
            'satellite_slant_range_km': sat_info['slant_range_km'],
            'satellite_altitude_km':    sat_info['alt_m'] / 1000.0,
        })
        l2_map = l2.compute(
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            timestamp=ts,
            context=l2_context,
        )

        total_loss = (l1_map + l2_map).astype(np.float32)
        rx_power = (tx_power_dbw - total_loss + rx_gain_dbi + 30.0).astype(np.float32)

        ts_str = ts.strftime('%Y%m%dT%H%M%S')
        np.save(output_dir / f"frame_{idx:03d}_{ts_str}.npy", rx_power)

        # 联合图（DEM + 接收功率）
        frame_file = str(output_dir / f"frame_{idx:03d}_{ts_str}.png")
        _plot_dem_rsrp(
            dem_patch, rx_power, rx_sensitivity_dbm,
            origin_lat, origin_lon, coverage_km,
            ts, el, az, norad_id, idx, frame_file
        )
        frame_files.append(frame_file)
        print(f"[frame {idx:03d}] 保存: {frame_file}")

    # 生成 GIF 动画
    try:
        from PIL import Image
        imgs = [Image.open(f) for f in frame_files]
        gif_path = output_dir / f"dwell_{norad_id}.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                     duration=400, loop=0)
        print(f"\n[gif] 动画已保存: {gif_path}")
    except ImportError:
        print("\n[gif] 未安装 Pillow，跳过 GIF 生成（pip install Pillow）")

    l2.close()
    print(f"\n[完成] {len(frame_files)} 帧已保存至: {output_dir}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='波束凝视仿真')
    parser.add_argument('--config', default='configs/mission_config.yaml')
    parser.add_argument('--norad', default='57360', help='目标卫星 NORAD ID')
    parser.add_argument('--start', default='2025-01-01T00:00:00',
                        help='扫描起始时间 ISO 格式')
    parser.add_argument('--step', type=int, default=60,
                        help='时间步长（秒），默认 60s')
    parser.add_argument('--min-el', type=float, default=5.0,
                        help='最低仰角阈值（度），默认 5°')
    parser.add_argument('--output', default='output/dwell')
    args = parser.parse_args()

    from src.context.time_utils import parse_iso_utc
    ts = parse_iso_utc(args.start, strict=False)
    config_path = _ROOT / args.config
    output_dir  = _ROOT / args.output / args.norad

    run(str(config_path), args.norad, ts, args.step, output_dir, args.min_el)


if __name__ == '__main__':
    main()
