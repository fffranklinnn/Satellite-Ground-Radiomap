"""
Plotting and visualization utilities for SG-MRM project.

Provides functions for visualizing radio maps, layer outputs, and
exporting results as PNG images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Optional, Tuple
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
