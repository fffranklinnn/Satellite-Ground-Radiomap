"""Unit tests for GridSpec spatial contract (task6 / AC-1)."""

import math
import pytest
from src.context.grid_spec import GridSpec


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_from_legacy_args_basic():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    assert gs.center_lat == pytest.approx(34.0)
    assert gs.center_lon == pytest.approx(108.0)
    assert gs.width_m == pytest.approx(256_000.0)
    assert gs.height_m == pytest.approx(256_000.0)
    assert gs.nx == 256
    assert gs.ny == 256
    assert gs.dx_m == pytest.approx(1000.0)
    assert gs.dy_m == pytest.approx(1000.0)
    assert gs.anchor == "center"
    assert gs.pixel_registration == "center"
    assert gs.row_order == "north_to_south"
    assert gs.col_order == "west_to_east"


def test_from_sw_corner_center_offset():
    """SW-corner factory must shift center by half coverage."""
    sw_lat, sw_lon = 34.0, 108.0
    coverage_km = 25.6
    gs = GridSpec.from_sw_corner(sw_lat, sw_lon, coverage_km, 256, 256)

    half_km = coverage_km / 2.0
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * math.cos(math.radians(sw_lat + half_km * lat_per_km)))

    expected_center_lat = sw_lat + half_km * lat_per_km
    expected_center_lon = sw_lon + half_km * lon_per_km

    assert gs.center_lat == pytest.approx(expected_center_lat, abs=1e-9)
    assert gs.center_lon == pytest.approx(expected_center_lon, abs=1e-9)


def test_anchor_must_be_center():
    with pytest.raises(ValueError, match="anchor='center'"):
        GridSpec(
            crs="WGS84", anchor="sw_corner",
            center_lat=34.0, center_lon=108.0,
            width_m=256_000, height_m=256_000,
            nx=256, ny=256, dx_m=1000, dy_m=1000,
            pixel_registration="center",
            row_order="north_to_south", col_order="west_to_east",
        )


def test_pixel_registration_must_be_center():
    with pytest.raises(ValueError, match="pixel_registration='center'"):
        GridSpec(
            crs="WGS84", anchor="center",
            center_lat=34.0, center_lon=108.0,
            width_m=256_000, height_m=256_000,
            nx=256, ny=256, dx_m=1000, dy_m=1000,
            pixel_registration="area",
            row_order="north_to_south", col_order="west_to_east",
        )


# ---------------------------------------------------------------------------
# Round-trip: pixel_to_world -> world_to_pixel (error < 1e-9 deg)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row,col", [
    (0.0, 0.0),
    (127.5, 127.5),
    (255.0, 255.0),
    (0.0, 255.0),
    (255.0, 0.0),
    (64.0, 192.0),
])
def test_pixel_world_round_trip(row, col):
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    lat, lon = gs.pixel_to_world(row, col)
    row2, col2 = gs.world_to_pixel(lat, lon)
    assert abs(row2 - row) < 1e-9
    assert abs(col2 - col) < 1e-9


@pytest.mark.parametrize("lat,lon", [
    (34.0, 108.0),          # center
    (35.15, 108.0),         # north edge
    (32.85, 108.0),         # south edge
    (34.0, 109.35),         # east edge
    (34.0, 106.65),         # west edge
])
def test_world_pixel_round_trip(lat, lon):
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    row, col = gs.world_to_pixel(lat, lon)
    lat2, lon2 = gs.pixel_to_world(row, col)
    assert abs(lat2 - lat) < 1e-9
    assert abs(lon2 - lon) < 1e-9


# ---------------------------------------------------------------------------
# Center pixel maps to grid center
# ---------------------------------------------------------------------------

def test_center_pixel_is_grid_center():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    lat, lon = gs.pixel_to_world(gs.ny / 2.0, gs.nx / 2.0)
    assert abs(lat - gs.center_lat) < 1e-9
    assert abs(lon - gs.center_lon) < 1e-9


def test_world_to_pixel_center():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    row, col = gs.world_to_pixel(gs.center_lat, gs.center_lon)
    assert abs(row - gs.ny / 2.0) < 1e-9
    assert abs(col - gs.nx / 2.0) < 1e-9


# ---------------------------------------------------------------------------
# bbox
# ---------------------------------------------------------------------------

def test_bbox_symmetry():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    south, west, north, east = gs.bbox()
    assert north > south
    assert east > west
    # Center must be midpoint
    assert abs((north + south) / 2 - gs.center_lat) < 1e-6
    assert abs((east + west) / 2 - gs.center_lon) < 1e-6


def test_sw_corner_matches_bbox():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    sw_lat, sw_lon = gs.sw_corner()
    south, west, _, _ = gs.bbox()
    assert sw_lat == pytest.approx(south)
    assert sw_lon == pytest.approx(west)


# ---------------------------------------------------------------------------
# from_sw_corner -> sw_corner round-trip
# ---------------------------------------------------------------------------

def test_sw_corner_round_trip():
    """from_sw_corner then sw_corner() should recover the original SW corner."""
    sw_lat, sw_lon = 34.0, 108.0
    gs = GridSpec.from_sw_corner(sw_lat, sw_lon, 25.6, 256, 256)
    recovered_lat, recovered_lon = gs.sw_corner()
    # Tolerance: the approximation in from_sw_corner uses 111 km/deg
    assert abs(recovered_lat - sw_lat) < 1e-4
    assert abs(recovered_lon - sw_lon) < 1e-4


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------

def test_gridspec_is_frozen():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    with pytest.raises(Exception):
        gs.center_lat = 35.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr_contains_coverage():
    gs = GridSpec.from_legacy_args(34.0, 108.0, 256.0, 256, 256)
    r = repr(gs)
    assert "256.000km" in r
    assert "34.0000N" in r
