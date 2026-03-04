"""
Plotting and visualization utilities for SG-MRM project.

Provides functions for visualizing radio maps, layer outputs, and
exporting results as PNG images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime


def plot_radio_map(loss_map: np.ndarray,
                   title: str = "Radio Map",
                   output_file: Optional[str] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   cmap: str = 'viridis',
                   show_colorbar: bool = True,
                   show_stats: bool = True,
                   origin_lat: Optional[float] = None,
                   origin_lon: Optional[float] = None,
                   coverage_km: Optional[float] = None,
                   dpi: int = 150) -> None:
    """
    Plot a radio map as a 2D heatmap.

    Args:
        loss_map: 256×256 array of loss values in dB
        title: Plot title
        output_file: Optional output file path (PNG)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        cmap: Colormap name
        show_colorbar: Whether to show colorbar
        show_stats: Whether to overlay min/max/mean/std text box
        origin_lat: Origin latitude (degrees) — enables geographic axis labels
        origin_lon: Origin longitude (degrees) — enables geographic axis labels
        coverage_km: Coverage width in km — enables geographic axis labels
        dpi: Output DPI for saved figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    valid = loss_map[~np.isnan(loss_map)]
    auto_vmin = float(np.nanmin(loss_map)) if vmin is None else vmin
    auto_vmax = float(np.nanmax(loss_map)) if vmax is None else vmax

    im = ax.imshow(loss_map, cmap=cmap, vmin=auto_vmin, vmax=auto_vmax,
                   origin='upper', interpolation='nearest')

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss (dB)', rotation=270, labelpad=20)

    # Geographic axis ticks
    if origin_lat is not None and origin_lon is not None and coverage_km is not None:
        n = loss_map.shape[0]
        lat_extent = coverage_km / 2.0 / 111.0
        lon_extent = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))

        n_ticks = 5
        lat_vals = np.linspace(origin_lat + lat_extent, origin_lat - lat_extent, n_ticks)
        lon_vals = np.linspace(origin_lon - lon_extent, origin_lon + lon_extent, n_ticks)
        tick_px = np.linspace(0, n - 1, n_ticks)

        ax.set_xticks(tick_px)
        ax.set_xticklabels([f"{v:.2f}°E" for v in lon_vals], fontsize=9)
        ax.set_yticks(tick_px)
        ax.set_yticklabels([f"{v:.2f}°N" for v in lat_vals], fontsize=9)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
    else:
        ax.set_xlabel('Pixel X', fontsize=12)
        ax.set_ylabel('Pixel Y', fontsize=12)

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Stats overlay
    if show_stats and len(valid) > 0:
        stats_text = (f"mean={np.mean(valid):.2f} dB\n"
                      f"std={np.std(valid):.2f} dB\n"
                      f"min={np.min(valid):.2f} dB\n"
                      f"max={np.max(valid):.2f} dB")
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved radio map to {output_file}")
    else:
        plt.show()

    plt.close()


def plot_layer_comparison(l1_map: Optional[np.ndarray] = None,
                          l2_map: Optional[np.ndarray] = None,
                          l3_map: Optional[np.ndarray] = None,
                          composite_map: Optional[np.ndarray] = None,
                          output_file: Optional[str] = None,
                          dpi: int = 150) -> None:
    """
    Plot comparison of different layer outputs.

    Args:
        l1_map: L1 layer output
        l2_map: L2 layer output
        l3_map: L3 layer output
        composite_map: Composite map
        output_file: Optional output file path
        dpi: Output DPI
    """
    # Count available maps
    maps = [('L1 Macro', l1_map), ('L2 Terrain', l2_map),
            ('L3 Urban', l3_map), ('Composite', composite_map)]
    available_maps = [(name, m) for name, m in maps if m is not None]

    if not available_maps:
        print("No maps to plot")
        return

    # Create subplots
    n_maps = len(available_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(5 * n_maps, 4))

    if n_maps == 1:
        axes = [axes]

    # Plot each map
    for ax, (name, loss_map) in zip(axes, available_maps):
        im = ax.imshow(loss_map, cmap='viridis', origin='upper')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        plt.colorbar(im, ax=ax, label='Loss (dB)')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved layer comparison to {output_file}")
    else:
        plt.show()

    plt.close()


def _apply_geographic_ticks(ax,
                            shape: Tuple[int, int],
                            origin_lat: Optional[float],
                            origin_lon: Optional[float],
                            coverage_km: Optional[float],
                            n_ticks: int = 5) -> None:
    """Apply geographic axis ticks when origin/coverage are available."""
    if origin_lat is None or origin_lon is None or coverage_km is None:
        ax.set_xlabel('Pixel X', fontsize=11)
        ax.set_ylabel('Pixel Y', fontsize=11)
        return

    n = shape[0]
    lat_extent = coverage_km / 2.0 / 111.0
    lon_extent = coverage_km / 2.0 / (111.0 * np.cos(np.radians(origin_lat)))

    lat_vals = np.linspace(origin_lat + lat_extent, origin_lat - lat_extent, n_ticks)
    lon_vals = np.linspace(origin_lon - lon_extent, origin_lon + lon_extent, n_ticks)
    tick_px = np.linspace(0, n - 1, n_ticks)

    ax.set_xticks(tick_px)
    ax.set_xticklabels([f"{v:.5f}°E" for v in lon_vals], fontsize=8)
    ax.set_yticks(tick_px)
    ax.set_yticklabels([f"{v:.5f}°N" for v in lat_vals], fontsize=8)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)


def plot_full_radiomap_paper(composite_map: np.ndarray,
                             output_file: str,
                             origin_lat: float,
                             origin_lon: float,
                             coverage_km: float = 0.256,
                             timestamp: Optional[datetime] = None,
                             title: str = "Full-Physics Composite Radio Map",
                             l1_map: Optional[np.ndarray] = None,
                             l2_map: Optional[np.ndarray] = None,
                             l3_map: Optional[np.ndarray] = None,
                             iono_map: Optional[np.ndarray] = None,
                             atm_map: Optional[np.ndarray] = None,
                             terrain_mask: Optional[np.ndarray] = None,
                             urban_nlos_mask: Optional[np.ndarray] = None,
                             urban_occ_mask: Optional[np.ndarray] = None,
                             note_lines: Optional[Dict[str, Any]] = None,
                             show_decomposition: bool = False,
                             decomposition_output_file: Optional[str] = None,
                             cmap: str = "viridis",
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             dpi: int = 300) -> None:
    """
    Plot a publication-quality composite map with optional physical annotations.

    The main figure is a single composite map. Optional decomposition figure
    includes L1/L2/L3/Composite as 2x2 subplots.
    """
    valid = composite_map[~np.isnan(composite_map)]
    if valid.size == 0:
        raise ValueError("composite_map has no valid values.")

    auto_vmin = float(np.nanmin(composite_map)) if vmin is None else vmin
    auto_vmax = float(np.nanmax(composite_map)) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=(11, 8.5))
    im = ax.imshow(
        composite_map,
        cmap=cmap,
        vmin=auto_vmin,
        vmax=auto_vmax,
        origin='upper',
        interpolation='nearest'
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Composite Loss (dB)', rotation=270, labelpad=18)

    legend_handles = []
    if terrain_mask is not None:
        ax.contour(terrain_mask.astype(float), levels=[0.5], colors='black',
                   linewidths=1.1, linestyles='--')
        legend_handles.append(Line2D([0], [0], color='black', lw=1.1, ls='--',
                                     label='Terrain Occlusion'))
    if urban_nlos_mask is not None:
        ax.contour(urban_nlos_mask.astype(float), levels=[0.5], colors='red',
                   linewidths=1.0, linestyles='-')
        legend_handles.append(Line2D([0], [0], color='red', lw=1.0, ls='-',
                                     label='Urban NLoS'))
    if urban_occ_mask is not None:
        ax.contour(urban_occ_mask.astype(float), levels=[0.5], colors='orange',
                   linewidths=1.0, linestyles=':')
        legend_handles.append(Line2D([0], [0], color='orange', lw=1.0, ls=':',
                                     label='Building Occupancy'))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9, fontsize=8)

    header = title
    if timestamp is not None:
        header = f"{title}\n{timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    ax.set_title(header, fontsize=13, fontweight='bold')
    _apply_geographic_ticks(ax, composite_map.shape, origin_lat, origin_lon, coverage_km)

    stats_lines = [
        f"mean={np.nanmean(composite_map):.2f} dB",
        f"std={np.nanstd(composite_map):.2f} dB",
        f"min={np.nanmin(composite_map):.2f} dB",
        f"max={np.nanmax(composite_map):.2f} dB",
    ]
    if iono_map is not None:
        stats_lines.append(f"iono mean/max={np.nanmean(iono_map):.3f}/{np.nanmax(iono_map):.3f} dB")
    if atm_map is not None:
        stats_lines.append(f"atm mean/max={np.nanmean(atm_map):.3f}/{np.nanmax(atm_map):.3f} dB")
    if terrain_mask is not None:
        stats_lines.append(f"terrain occluded={100.0*np.mean(terrain_mask):.2f}%")
    if urban_nlos_mask is not None:
        stats_lines.append(f"urban NLoS={100.0*np.mean(urban_nlos_mask):.2f}%")
    if urban_occ_mask is not None:
        stats_lines.append(f"urban occupied={100.0*np.mean(urban_occ_mask):.2f}%")
    if note_lines:
        for k, v in note_lines.items():
            stats_lines.append(f"{k}: {v}")

    ax.text(
        0.015, 0.015, "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=8,
        va='bottom',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.82)
    )

    plt.tight_layout()
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if show_decomposition:
        decomp_path = decomposition_output_file
        if decomp_path is None:
            out_path = Path(output_file)
            decomp_path = str(out_path.with_name(f"{out_path.stem}_decomposition{out_path.suffix}"))

        panel_maps = [
            ("L1 Macro", l1_map),
            ("L2 Terrain", l2_map),
            ("L3 Urban", l3_map),
            ("Composite", composite_map),
        ]
        fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax_i, (name, data) in zip(axes.ravel(), panel_maps):
            if data is None:
                ax_i.text(0.5, 0.5, f"{name}\nN/A", ha='center', va='center', fontsize=11)
                ax_i.axis('off')
                continue

            valid_i = data[~np.isnan(data)]
            local_vmin = float(np.nanmin(valid_i))
            local_vmax = float(np.nanmax(valid_i))
            im_i = ax_i.imshow(data, cmap=cmap, vmin=local_vmin, vmax=local_vmax,
                               origin='upper', interpolation='nearest')
            ax_i.set_title(name, fontsize=11, fontweight='bold')
            _apply_geographic_ticks(ax_i, data.shape, origin_lat, origin_lon, coverage_km)
            plt.colorbar(im_i, ax=ax_i, fraction=0.046, pad=0.04, label='Loss (dB)')
            ax_i.text(
                0.02, 0.02,
                f"mean={np.nanmean(data):.2f}\nstd={np.nanstd(data):.2f}\n"
                f"min={np.nanmin(data):.2f}\nmax={np.nanmax(data):.2f}",
                transform=ax_i.transAxes,
                fontsize=7,
                va='bottom',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        fig2.suptitle("Layer Decomposition", fontsize=13, fontweight='bold')
        plt.tight_layout()
        Path(decomp_path).parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(decomp_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig2)


def export_radio_map_png(loss_map: np.ndarray,
                         output_file: str,
                         normalize: bool = True) -> None:
    """
    Export radio map as a simple PNG image (256×256 pixels).

    This creates a grayscale image where pixel intensity represents loss.

    Args:
        loss_map: 256×256 array of loss values in dB
        output_file: Output PNG file path
        normalize: Whether to normalize values to 0-255 range
    """
    # Normalize to 0-255 range
    if normalize:
        min_val = np.min(loss_map)
        max_val = np.max(loss_map)
        if max_val > min_val:
            normalized = ((loss_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(loss_map, dtype=np.uint8)
    else:
        normalized = np.clip(loss_map, 0, 255).astype(np.uint8)

    # Save as PNG
    plt.imsave(output_file, normalized, cmap='gray', vmin=0, vmax=255)
    print(f"Exported radio map to {output_file}")


def plot_time_series(timestamps: list,
                     loss_values: list,
                     title: str = "Loss Over Time",
                     ylabel: str = "Loss (dB)",
                     output_file: Optional[str] = None,
                     dpi: int = 150) -> None:
    """
    Plot time series of loss values.

    Args:
        timestamps: List of datetime objects
        loss_values: List of loss values
        title: Plot title
        ylabel: Y-axis label
        output_file: Optional output file path
        dpi: Output DPI
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(timestamps, loss_values, marker='o', linestyle='-', linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved time series plot to {output_file}")
    else:
        plt.show()

    plt.close()


def create_animation_frames(loss_maps: list,
                            timestamps: list,
                            output_dir: str,
                            prefix: str = "frame") -> None:
    """
    Create animation frames from a sequence of radio maps.

    Args:
        loss_maps: List of 256×256 loss map arrays
        timestamps: List of corresponding timestamps
        output_dir: Output directory for frames
        prefix: Filename prefix for frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find global min/max for consistent colormap
    all_maps = np.array(loss_maps)
    vmin = np.min(all_maps)
    vmax = np.max(all_maps)

    for i, (loss_map, timestamp) in enumerate(zip(loss_maps, timestamps)):
        output_file = output_path / f"{prefix}_{i:04d}.png"
        title = f"Radio Map - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        plot_radio_map(loss_map, title=title, output_file=str(output_file),
                      vmin=vmin, vmax=vmax, dpi=100)

    print(f"Created {len(loss_maps)} animation frames in {output_dir}")


def plot_statistics(loss_map: np.ndarray,
                    title: str = "Loss Distribution",
                    output_file: Optional[str] = None,
                    dpi: int = 150) -> None:
    """
    Plot statistical distribution of loss values.

    Args:
        loss_map: 256×256 array of loss values
        title: Plot title
        output_file: Optional output file path
        dpi: Output DPI
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(loss_map.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax1.set_title('Loss Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Loss (dB)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Statistics text
    stats_text = f"""
    Mean: {np.mean(loss_map):.2f} dB
    Median: {np.median(loss_map):.2f} dB
    Std Dev: {np.std(loss_map):.2f} dB
    Min: {np.min(loss_map):.2f} dB
    Max: {np.max(loss_map):.2f} dB
    """
    ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axis('off')
    ax2.set_title('Statistics', fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"Saved statistics plot to {output_file}")
    else:
        plt.show()

    plt.close()
