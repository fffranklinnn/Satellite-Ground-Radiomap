"""Unit tests for utils.data_validation module."""

from __future__ import annotations

from pathlib import Path

from src.utils.data_validation import validate_data_integrity


def test_validation_reports_missing_tle_as_error(tmp_path: Path):
    cfg = {
        "layers": {
            "l1_macro": {
                "enabled": True,
                "tle_file": "data/missing.tle",
            },
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        }
    }
    report = validate_data_integrity(config=cfg, project_root=tmp_path, strict=True)
    codes = {item["code"] for item in report["errors"]}
    assert "l1.tle.not_found" in codes


def test_validation_all_layers_disabled_has_no_errors(tmp_path: Path):
    cfg = {
        "layers": {
            "l1_macro": {"enabled": False},
            "l2_topo": {"enabled": False},
            "l3_urban": {"enabled": False},
        }
    }
    report = validate_data_integrity(config=cfg, project_root=tmp_path, strict=True)
    assert report["errors"] == []
