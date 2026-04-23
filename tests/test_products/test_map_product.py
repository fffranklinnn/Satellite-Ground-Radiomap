"""
MapProduct and dataset traceability tests (AC-5).

Tests:
- Multi-product export from single frame
- Missing manifest raises error
- Traceability dict contains required fields
"""

import numpy as np
import pytest

from src.context.grid_spec import GridSpec
from src.products.manifest import ProductManifest, ProvenanceBlock
from src.products.map_product import MapProduct, MissingManifestError


CENTER_LAT, CENTER_LON = 34.26, 108.94


def _grid(role="product"):
    return GridSpec.from_legacy_args(CENTER_LAT, CENTER_LON, 25.6, 64, 64, role=role)


def _manifest():
    return ProductManifest(
        frame_id="f1", timestamp_utc="2025-01-01T00:00:00+00:00",
        config_hash="ch", data_snapshot_id="snap",
        provenance=ProvenanceBlock(benchmark_id="b1", software_version="v1"),
    )


class TestMapProduct:
    def test_construction(self):
        g = _grid()
        arr = np.zeros((64, 64), dtype=np.float32)
        mp = MapProduct(
            product_id="p1", product_type="path_loss",
            grid=g, values=arr, units="dB",
            frame_id="f1", manifest=_manifest(),
        )
        assert mp.product_type == "path_loss"

    def test_missing_manifest_raises(self):
        g = _grid()
        arr = np.zeros((64, 64), dtype=np.float32)
        mp = MapProduct(
            product_id="p1", product_type="path_loss",
            grid=g, values=arr, units="dB",
            frame_id="f1", manifest=None,
        )
        with pytest.raises(MissingManifestError):
            mp.validate_manifest()

    def test_shape_mismatch_raises(self):
        g = _grid()
        arr = np.zeros((32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            MapProduct(
                product_id="p1", product_type="path_loss",
                grid=g, values=arr, units="dB",
                frame_id="f1",
            )

    def test_multi_product_from_single_frame(self):
        g = _grid()
        m = _manifest()
        products = [
            MapProduct("p_loss", "path_loss", g, np.ones((64, 64), np.float32) * 150, "dB", "f1", m),
            MapProduct("p_vis", "visibility", g, np.ones((64, 64), bool), "bool", "f1", m),
            MapProduct("p_block", "terrain_blockage", g, np.zeros((64, 64), bool), "bool", "f1", m),
            MapProduct("p_urban", "urban_residual", g, np.ones((64, 64), np.float32) * 5, "dB", "f1", m),
        ]
        assert len(products) == 4
        for p in products:
            p.validate_manifest()

    def test_traceability_dict(self):
        g = _grid()
        m = _manifest()
        mp = MapProduct("p1", "path_loss", g, np.zeros((64, 64), np.float32), "dB", "f1", m)
        td = mp.to_traceability_dict()
        assert td["product_id"] == "p1"
        assert td["frame_id"] == "f1"
        assert td["grid_role"] == "product"
        assert "manifest" in td
        assert td["manifest"]["config_hash"] == "ch"
        assert td["manifest"]["data_snapshot_id"] == "snap"
        assert "provenance" in td["manifest"]
        assert td["manifest"]["provenance"]["benchmark_id"] == "b1"

    def test_traceability_without_manifest_raises(self):
        g = _grid()
        mp = MapProduct("p1", "path_loss", g, np.zeros((64, 64), np.float32), "dB", "f1")
        with pytest.raises(MissingManifestError):
            mp.to_traceability_dict()
