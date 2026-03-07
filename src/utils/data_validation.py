"""
Data integrity checks for SG-MRM runtime configuration.

The checks are intentionally lightweight and deterministic so they can be
called from CLI entry points before expensive simulation jobs start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml


def load_yaml_config(path: Path) -> Dict:
    """Load a YAML config file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_project_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    """Resolve a potentially relative path against project root."""
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (project_root / p)


def _check_tle_has_entries(path: Path) -> Tuple[bool, str]:
    """Check whether a TLE file has at least one valid line1/line2 pair."""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except Exception as exc:
        return False, f"cannot read TLE file: {exc}"

    valid_pairs = 0
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            valid_pairs += 1
    if valid_pairs <= 0:
        return False, "no valid TLE line1/line2 pair found"
    return True, f"{valid_pairs} valid TLE pairs"


def _load_nc_variable_names(path: Path) -> Tuple[Optional[Set[str]], Optional[str]]:
    """Load variable names from NetCDF using netCDF4 or xarray."""
    try:
        import netCDF4 as nc  # type: ignore

        ds = nc.Dataset(path, "r")
        vars_set = set(ds.variables.keys())
        ds.close()
        return vars_set, None
    except Exception as exc_nc:
        try:
            import xarray as xr  # type: ignore

            ds = xr.open_dataset(path)
            vars_set = set(ds.variables.keys())
            ds.close()
            return vars_set, None
        except Exception as exc_xr:
            return None, f"netcdf read failed (netCDF4={exc_nc}; xarray={exc_xr})"


def _check_ionex_header(path: Path) -> Tuple[bool, str]:
    """Check basic readability and header signature of an IONEX file."""
    import gzip

    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="ascii", errors="replace") as f:
            chunk = "".join([next(f) for _ in range(40)])
    except StopIteration:
        return False, "file too short"
    except Exception as exc:
        return False, f"cannot read IONEX file: {exc}"

    if "IONEX VERSION / TYPE" not in chunk:
        return False, "missing IONEX header signature"
    return True, "header ok"


def _find_l3_tiles_with_height(tile_root: Path, max_scan: int = 2000) -> int:
    """
    Count tile folders containing H.npy.

    max_scan avoids extremely slow scans on very large folders while still
    giving useful integrity signals.
    """
    count = 0
    scanned = 0
    for child in tile_root.iterdir():
        if scanned >= max_scan:
            break
        scanned += 1
        if child.is_dir() and (child / "H.npy").exists():
            count += 1
    return count


def validate_data_integrity(config: Dict, project_root: Path, strict: bool = False) -> Dict:
    """
    Validate data dependencies declared in runtime config.

    Returns a report dict with keys: strict/errors/warnings/checks.
    """
    report = {
        "strict": bool(strict),
        "errors": [],
        "warnings": [],
        "checks": [],
    }

    def add_error(code: str, message: str, path: Optional[Path] = None) -> None:
        issue = {"code": code, "message": message}
        if path is not None:
            issue["path"] = str(path)
        report["errors"].append(issue)

    def add_warning(code: str, message: str, path: Optional[Path] = None) -> None:
        issue = {"code": code, "message": message}
        if path is not None:
            issue["path"] = str(path)
        report["warnings"].append(issue)

    def add_check(name: str, ok: bool, message: str, path: Optional[Path] = None) -> None:
        row = {"name": name, "ok": bool(ok), "message": message}
        if path is not None:
            row["path"] = str(path)
        report["checks"].append(row)

    layers = config.get("layers", {})

    # ---- L1 checks ----------------------------------------------------------
    l1_cfg = layers.get("l1_macro", {})
    l1_enabled = bool(l1_cfg.get("enabled", True))
    if l1_enabled:
        tle_file = l1_cfg.get("tle_file")
        if tle_file is None and isinstance(l1_cfg.get("tle"), dict):
            tle_file = l1_cfg["tle"].get("file")
        tle_path = resolve_project_path(project_root, tle_file)
        if tle_path is None:
            add_error("l1.tle.missing", "L1 enabled but tle_file is not configured")
        elif not tle_path.exists():
            add_error("l1.tle.not_found", "L1 TLE file does not exist", tle_path)
        else:
            ok, msg = _check_tle_has_entries(tle_path)
            add_check("l1_tle_parse", ok, msg, tle_path)
            if not ok:
                add_error("l1.tle.invalid", msg, tle_path)

        ionex_path = resolve_project_path(project_root, l1_cfg.get("ionex_file"))
        if ionex_path is None:
            add_warning("l1.ionex.unset", "IONEX is not configured; runtime will use fallback TEC")
        elif not ionex_path.exists():
            level = add_error if strict else add_warning
            level("l1.ionex.not_found", "Configured IONEX file does not exist", ionex_path)
        else:
            ok, msg = _check_ionex_header(ionex_path)
            add_check("l1_ionex_header", ok, msg, ionex_path)
            if not ok:
                level = add_error if strict else add_warning
                level("l1.ionex.invalid", msg, ionex_path)

        era5_path = resolve_project_path(project_root, l1_cfg.get("era5_file"))
        if era5_path is None:
            add_warning("l1.era5.unset", "ERA5 is not configured; runtime will use simplified atmospheric model")
        elif not era5_path.exists():
            level = add_error if strict else add_warning
            level("l1.era5.not_found", "Configured ERA5 file does not exist", era5_path)
        else:
            vars_set, err = _load_nc_variable_names(era5_path)
            if vars_set is None:
                level = add_error if strict else add_warning
                level("l1.era5.read_failed", err or "cannot read ERA5 NetCDF variables", era5_path)
            else:
                required = {"q", "pressure_level", "latitude", "longitude", "valid_time"}
                missing = sorted(required - vars_set)
                if missing:
                    add_error(
                        "l1.era5.missing_required_vars",
                        f"ERA5 missing required vars: {', '.join(missing)}",
                        era5_path,
                    )
                    add_check("l1_era5_vars_required", False, f"missing {missing}", era5_path)
                else:
                    add_check("l1_era5_vars_required", True, "required vars present", era5_path)

                recommended = {"z", "r", "t"}
                missing_rec = sorted(recommended - vars_set)
                if missing_rec:
                    add_warning(
                        "l1.era5.missing_recommended_vars",
                        f"ERA5 missing recommended vars: {', '.join(missing_rec)}",
                        era5_path,
                    )

    # ---- L2 checks ----------------------------------------------------------
    l2_cfg = layers.get("l2_topo", {})
    l2_enabled = bool(l2_cfg.get("enabled", True))
    if l2_enabled:
        dem_path = resolve_project_path(project_root, l2_cfg.get("dem_file"))
        if dem_path is None:
            level = add_error if strict else add_warning
            level("l2.dem.unset", "L2 enabled but dem_file is not configured")
        elif not dem_path.exists():
            level = add_error if strict else add_warning
            level("l2.dem.not_found", "Configured DEM file does not exist", dem_path)
        else:
            add_check("l2_dem_exists", True, "DEM file found", dem_path)

    # ---- L3 checks ----------------------------------------------------------
    l3_cfg = layers.get("l3_urban", {})
    l3_enabled = bool(l3_cfg.get("enabled", True))
    if l3_enabled:
        tile_root = resolve_project_path(project_root, l3_cfg.get("tile_cache_root"))
        if tile_root is None:
            add_error("l3.tiles.unset", "L3 enabled but tile_cache_root is not configured")
        elif not tile_root.exists():
            add_error("l3.tiles.not_found", "L3 tile cache root does not exist", tile_root)
        elif not tile_root.is_dir():
            add_error("l3.tiles.not_dir", "L3 tile cache root is not a directory", tile_root)
        else:
            h_count = _find_l3_tiles_with_height(tile_root)
            if h_count <= 0:
                add_error("l3.tiles.empty", "No tile folders with H.npy found under tile cache root", tile_root)
                add_check("l3_tile_cache_scan", False, "no H.npy tile found", tile_root)
            else:
                add_check("l3_tile_cache_scan", True, f"found >= {h_count} tiles with H.npy", tile_root)

    return report


def format_data_validation_report(report: Dict) -> str:
    """Format data validation report to human-readable multiline text."""
    lines: List[str] = []
    lines.append(
        "[data-check] strict={strict} errors={errors} warnings={warnings}".format(
            strict=report.get("strict", False),
            errors=len(report.get("errors", [])),
            warnings=len(report.get("warnings", [])),
        )
    )

    checks = report.get("checks", [])
    if checks:
        lines.append("[data-check] checks:")
        for row in checks:
            mark = "OK" if row.get("ok", False) else "FAIL"
            suffix = f" | {row['path']}" if row.get("path") else ""
            lines.append(f"  - {mark} {row.get('name')}: {row.get('message')}{suffix}")

    errors = report.get("errors", [])
    if errors:
        lines.append("[data-check] errors:")
        for e in errors:
            suffix = f" | {e['path']}" if e.get("path") else ""
            lines.append(f"  - {e.get('code')}: {e.get('message')}{suffix}")

    warnings = report.get("warnings", [])
    if warnings:
        lines.append("[data-check] warnings:")
        for w in warnings:
            suffix = f" | {w['path']}" if w.get("path") else ""
            lines.append(f"  - {w.get('code')}: {w.get('message')}{suffix}")

    return "\n".join(lines)
