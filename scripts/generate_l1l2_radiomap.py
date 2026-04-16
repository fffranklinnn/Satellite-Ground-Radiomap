#!/usr/bin/env python3
"""
L1+L2 大尺度无线电地图生成脚本

在 256 km 尺度（1000 m/pixel）上合成 L1 宏观层与 L2 地形层，
生成大尺度卫星-地面路径损耗地图。

输出：
  - l1_macro.npy / l1_macro.png       — L1 宏观损耗图
  - l2_topo.npy  / l2_topo.png        — L2 地形衍射损耗
  - l1l2_composite.npy / .png         — L1+L2 合成损耗图
  - l1l2_decomposition.png            — 三图对比

用法：
  python scripts/generate_l1l2_radiomap.py
  python scripts/generate_l1l2_radiomap.py --config configs/mission_config.yaml
  python scripts/generate_l1l2_radiomap.py --timestamp 2025-01-01T06:00:00
  python scripts/generate_l1l2_radiomap.py --output output/l1l2
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# 确保项目根目录在 sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['font.monospace'] = ['Noto Sans CJK JP', 'DejaVu Sans Mono']

from src.layers.l1_macro import L1MacroLayer
from src.layers.l2_topo import L2TopoLayer
from src.utils.plotter import plot_radio_map


# ---------------------------------------------------------------------------
# 可视化：接收功率图（RSSI）
# ---------------------------------------------------------------------------

def plot_rx_power(rx_power: np.ndarray,
                  rx_sensitivity_dbm: float,
                  origin_lat: float,
                  origin_lon: float,
                  coverage_km: float,
                  timestamp: datetime,
                  output_file: str,
                  sat_el: float = None,
                  sat_az: float = None,
                  dpi: int = 150) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 7))
    sat_str = f"  |  el={sat_el:.1f}°  az={sat_az:.1f}°" if sat_el is not None else ""
    fig.suptitle(
        f"接收功率 (RSRP)  |  {timestamp.strftime('%Y-%m-%d %H:%M UTC')}{sat_str}\n"
        f"覆盖 {coverage_km} km  @  ({origin_lat:.4f}°N, {origin_lon:.4f}°E)",
        fontsize=11, fontweight='bold'
    )

    n = rx_power.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))

    # 无覆盖区域设为 NaN 显示为灰色
    display = rx_power.copy().astype(float)
    display[rx_power < rx_sensitivity_dbm] = np.nan

    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='#404040')

    vmax = float(np.nanmax(display))
    vmin = rx_sensitivity_dbm

    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='upper', interpolation='nearest')

    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)
    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:.3f}°E" for v in lon_ticks], fontsize=8, rotation=30)
    ax.set_yticks(tick_px)
    ax.set_yticklabels([f"{v:.3f}°N" for v in lat_ticks], fontsize=8)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('接收功率 (dBm)', fontsize=9)

    coverage_pct = 100.0 * np.sum(rx_power >= rx_sensitivity_dbm) / rx_power.size
    ax.text(0.02, 0.02,
            f"max={vmax:.1f} dBm\n"
            f"灵敏度={rx_sensitivity_dbm:.0f} dBm\n"
            f"覆盖率={coverage_pct:.1f}%",
            transform=ax.transAxes, fontsize=8, va='bottom',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[output] 接收功率图已保存: {output_file}")


# ---------------------------------------------------------------------------
# 可视化：DEM 高程图
# ---------------------------------------------------------------------------

def plot_dem(dem: np.ndarray,
             origin_lat: float,
             origin_lon: float,
             coverage_km: float,
             output_file: str,
             dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

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
    print(f"[output] DEM 高程图已保存: {output_file}")


# ---------------------------------------------------------------------------
# 可视化：DEM + 接收功率联合图（左 DEM，右 RSRP 叠加等高线）
# ---------------------------------------------------------------------------

def plot_dem_rsrp(dem: np.ndarray,
                 rx_power: np.ndarray,
                 rx_sensitivity_dbm: float,
                 origin_lat: float,
                 origin_lon: float,
                 coverage_km: float,
                 timestamp: datetime,
                 output_file: str,
                 title_extra: str = "",
                 dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    n = dem.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)

    rx_display = rx_power.copy().astype(float)
    rx_display[rx_power < rx_sensitivity_dbm] = np.nan
    coverage_pct = 100.0 * np.sum(rx_power >= rx_sensitivity_dbm) / rx_power.size

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ts_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')
    extra = f"  |  {title_extra}" if title_extra else ""
    fig.suptitle(
        f"地形 vs 接收信号{extra}\n"
        f"{ts_str}   ({origin_lat:.3f}°N, {origin_lon:.3f}°E)   覆盖 {coverage_km} km",
        fontsize=12, fontweight='bold'
    )

    # --- 左图：DEM 高程 ---
    ax0 = axes[0]
    im0 = ax0.imshow(dem, cmap='terrain', origin='upper', interpolation='bilinear')
    ax0.set_title('DEM 地形高程 (m)', fontsize=11, fontweight='bold')
    ax0.set_xticks(tick_px)
    ax0.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=8, rotation=30)
    ax0.set_yticks(tick_px)
    ax0.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=8)
    cb0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    cb0.set_label('高程 (m)', fontsize=9)
    ax0.text(0.02, 0.02,
             f"min={dem.min():.0f} m\nmax={dem.max():.0f} m",
             transform=ax0.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # --- 右图：接收功率 + DEM 等高线 ---
    ax1 = axes[1]
    cmap_rsrp = plt.cm.jet.copy()
    cmap_rsrp.set_bad(color='#303030')
    vmax = float(np.nanmax(rx_display)) if not np.all(np.isnan(rx_display)) else -40.0
    im1 = ax1.imshow(rx_display, cmap=cmap_rsrp, vmin=rx_sensitivity_dbm, vmax=vmax,
                     origin='upper', interpolation='nearest')
    ax1.set_title('接收功率 RSRP (dBm)', fontsize=11, fontweight='bold')
    ax1.set_xticks(tick_px)
    ax1.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=8, rotation=30)
    ax1.set_yticks(tick_px)
    ax1.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=8)
    cb1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label('RSRP (dBm)', fontsize=9)
    # 无覆盖区域标注
    no_cov_mask = (rx_power < rx_sensitivity_dbm).astype(float)
    no_cov_mask[no_cov_mask == 0] = np.nan
    ax1.imshow(no_cov_mask, cmap='gray', vmin=0, vmax=1,
               origin='upper', interpolation='nearest', alpha=0.0)  # 仅占位，颜色已由 set_bad 处理
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
    print(f"[output] DEM+RSRP 联合图已保存: {output_file}")


# ---------------------------------------------------------------------------
# 可视化：四图对比（L1损耗 / L2损耗 / 总损耗 / 接收功率）
# ---------------------------------------------------------------------------

def plot_decomposition(l1_map: np.ndarray,
                       l2_map: np.ndarray,
                       total_loss: np.ndarray,
                       rx_power: np.ndarray,
                       rx_sensitivity_dbm: float,
                       origin_lat: float,
                       origin_lon: float,
                       coverage_km: float,
                       timestamp: datetime,
                       output_file: str,
                       sat_el: float = None,
                       sat_az: float = None,
                       dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    n = total_loss.shape[0]
    lat_ext = coverage_km / 2.0 / 111.0
    lon_ext = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))
    tick_px = np.linspace(0, n - 1, 5)
    lat_ticks = np.linspace(origin_lat + lat_ext, origin_lat - lat_ext, 5)
    lon_ticks = np.linspace(origin_lon - lon_ext, origin_lon + lon_ext, 5)

    rx_display = rx_power.copy().astype(float)
    rx_display[rx_power < rx_sensitivity_dbm] = np.nan

    panels = [
        ("L1 路径损耗 (dB)",   l1_map,     'viridis', None,  None,  'Loss (dB)'),
        ("L2 地形衍射损耗 (dB)", l2_map,   'hot_r',   None,  None,  'Loss (dB)'),
        ("总损耗 L1+L2 (dB)",  total_loss, 'viridis', None,  None,  'Loss (dB)'),
        ("接收功率 RSRP (dBm)", rx_display, 'jet',     rx_sensitivity_dbm, float(np.nanmax(rx_display)), 'Power (dBm)'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    sat_str = f"  |  el={sat_el:.1f}°  az={sat_az:.1f}°" if sat_el is not None else ""
    fig.suptitle(
        f"L1+L2 无线电地图  |  {timestamp.strftime('%Y-%m-%d %H:%M UTC')}{sat_str}  |  "
        f"覆盖 {coverage_km} km  @  ({origin_lat:.4f}°N, {origin_lon:.4f}°E)",
        fontsize=12, fontweight='bold'
    )

    for ax, (title, data, cmap, vmin, vmax, clabel) in zip(axes, panels):
        _vmin = vmin if vmin is not None else float(np.nanmin(data))
        _vmax = vmax if vmax is not None else float(np.nanmax(data))
        _cmap = plt.cm.get_cmap(cmap).copy()
        _cmap.set_bad(color='#404040')
        im = ax.imshow(data, cmap=_cmap, vmin=_vmin, vmax=_vmax,
                       origin='upper', interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xticks(tick_px)
        ax.set_xticklabels([f"{v:.2f}°E" for v in lon_ticks], fontsize=7, rotation=30)
        ax.set_yticks(tick_px)
        ax.set_yticklabels([f"{v:.2f}°N" for v in lat_ticks], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=clabel)
        ax.text(0.02, 0.02,
                f"mean={np.nanmean(data):.1f}\nmin={np.nanmin(data):.1f}\nmax={np.nanmax(data):.1f}",
                transform=ax.transAxes, fontsize=7, va='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[output] 对比图已保存: {output_file}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(config_path: str,
        timestamp: datetime,
        output_dir: Path) -> None:

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    origin_lat = cfg['origin']['latitude']
    origin_lon = cfg['origin']['longitude']
    coverage_km = float(cfg['layers']['l1_macro'].get('coverage_km', 256.0))

    print(f"[config] 原点: ({origin_lat}°N, {origin_lon}°E)")
    print(f"[config] 时间: {timestamp.isoformat()}")
    print(f"[config] 覆盖: {coverage_km} km")

    # ── 初始化 L1 ──────────────────────────────────────────────────────────
    print("\n[L1] 初始化...")
    l1 = L1MacroLayer(cfg['layers']['l1_macro'], origin_lat, origin_lon)
    l1.set_time(timestamp)

    # ── 初始化 L2 ──────────────────────────────────────────────────────────
    print("\n[L2] 初始化...")
    l2 = L2TopoLayer(cfg['layers']['l2_topo'], origin_lat, origin_lon)

    # ── 计算 L1 ────────────────────────────────────────────────────────────
    print("\n[L1] 计算路径损耗...")
    l1_components = l1.compute_components(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp,
    )
    l1_map = l1_components['total']          # 256×256 @ 256 km
    sat_info = l1_components['satellite']

    # ── 计算 L2（传入卫星几何参数）─────────────────────────────────────────
    print("\n[L2] 计算地形衍射损耗...")
    from src.layers.base import LayerContext
    l2_context = LayerContext(extras={
        'satellite_elevation_deg': sat_info['elevation_deg'],
        'satellite_azimuth_deg':   sat_info['azimuth_deg'],
        'satellite_slant_range_km': sat_info['slant_range_km'],
        'satellite_altitude_km':   sat_info['alt_m'] / 1000.0,
    })
    l2_map = l2.compute(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        timestamp=timestamp,
        context=l2_context,
    )                                        # 256×256 @ 256 km

    # ── dB 域相加（同尺度，无需插值）──────────────────────────────────────
    total_loss = (l1_map + l2_map).astype(np.float32)

    # ── 链路预算：接收功率 = 发射功率 - 总损耗 + 接收天线增益 ──────────────
    lb = cfg.get('link_budget', {})
    tx_power_dbw      = float(lb.get('tx_power_dbw', 20.0))
    rx_gain_dbi       = float(lb.get('rx_gain_dbi', 33.0))
    rx_sensitivity_dbw = float(lb.get('rx_sensitivity_dbw', -130.0))

    rx_power_dbw = (tx_power_dbw - total_loss + rx_gain_dbi).astype(np.float32)
    rx_sensitivity_dbm = rx_sensitivity_dbw + 30.0
    rx_power = rx_power_dbw + 30.0   # dBm
    no_coverage_mask = rx_power < rx_sensitivity_dbm
    coverage_ratio = 1.0 - no_coverage_mask.mean()

    print(f"\n[结果] L1 损耗   — mean={l1_map.mean():.1f} dB, "
          f"min={l1_map.min():.1f}, max={l1_map.max():.1f}")
    print(f"[结果] L2 损耗   — mean={l2_map.mean():.1f} dB, "
          f"min={l2_map.min():.1f}, max={l2_map.max():.1f}")
    print(f"[结果] 总损耗    — mean={total_loss.mean():.1f} dB, "
          f"min={total_loss.min():.1f}, max={total_loss.max():.1f}")
    print(f"[结果] 接收功率  — mean={rx_power[~no_coverage_mask].mean():.1f} dBm, "
          f"min={rx_power[~no_coverage_mask].min():.1f}, "
          f"max={rx_power.max():.1f}  (灵敏度={rx_sensitivity_dbm:.0f} dBm)")
    print(f"[结果] 覆盖率    — {coverage_ratio*100:.1f}%")

    # ── 保存 ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_str = timestamp.strftime('%Y%m%dT%H%M%S')

    # 损耗图
    for name, data in [('l1_loss', l1_map), ('l2_loss', l2_map), ('total_loss', total_loss)]:
        np.save(output_dir / f"{name}_{ts_str}.npy", data)
        plot_radio_map(
            data,
            title=f"{name}  |  {timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            output_file=str(output_dir / f"{name}_{ts_str}.png"),
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            coverage_km=coverage_km,
            dpi=150,
        )

    # 接收功率图（RSSI）
    np.save(output_dir / f"rx_power_{ts_str}.npy", rx_power)
    plot_rx_power(
        rx_power,
        rx_sensitivity_dbm=rx_sensitivity_dbm,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=coverage_km,
        timestamp=timestamp,
        sat_el=sat_info['elevation_deg'],
        sat_az=sat_info['azimuth_deg'],
        output_file=str(output_dir / f"rx_power_{ts_str}.png"),
    )

    plot_decomposition(
        l1_map, l2_map, total_loss, rx_power,
        rx_sensitivity_dbm=rx_sensitivity_dbm,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=coverage_km,
        timestamp=timestamp,
        sat_el=sat_info['elevation_deg'],
        sat_az=sat_info['azimuth_deg'],
        output_file=str(output_dir / f"l1l2_decomposition_{ts_str}.png"),
    )

    # DEM 高程图
    dem_patch = l2.get_dem_patch(origin_lat, origin_lon)
    np.save(output_dir / "dem_patch.npy", dem_patch)
    plot_dem(
        dem_patch,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=coverage_km,
        output_file=str(output_dir / "dem_patch.png"),
    )

    # DEM + 接收功率联合图
    plot_dem_rsrp(
        dem_patch, rx_power,
        rx_sensitivity_dbm=rx_sensitivity_dbm,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        coverage_km=coverage_km,
        timestamp=timestamp,
        output_file=str(output_dir / f"dem_rsrp_{ts_str}.png"),
    )

    l2.close()
    print(f"\n[完成] 所有输出已保存至: {output_dir}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='生成 L1+L2 大尺度无线电地图')
    parser.add_argument('--config', default='configs/mission_config.yaml')
    parser.add_argument('--timestamp', default='2025-01-01T00:00:00',
                        help='仿真时间 ISO 格式，如 2025-01-01T06:00:00')
    parser.add_argument('--output', default='output/l1l2')
    args = parser.parse_args()

    ts = datetime.fromisoformat(args.timestamp).replace(tzinfo=timezone.utc)

    config_path = _ROOT / args.config
    output_dir  = _ROOT / args.output

    run(str(config_path), ts, output_dir)


if __name__ == '__main__':
    main()
