#!/usr/bin/env python3
"""
Validate SG-MRM data dependencies declared by a config file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.data_validation import (  # noqa: E402
    load_yaml_config,
    validate_data_integrity,
    format_data_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SG-MRM data integrity.")
    parser.add_argument("--config", type=str, default="configs/mission_config.yaml",
                        help="Config YAML path")
    parser.add_argument("--strict", action="store_true",
                        help="Treat missing configured optional data as errors")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional JSON report output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    config = load_yaml_config(cfg_path)
    report = validate_data_integrity(config=config, project_root=project_root, strict=bool(args.strict))
    print(format_data_validation_report(report))

    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = project_root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[data-check] json report: {out_path}")

    return 2 if report.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
