#!/usr/bin/env python3
"""Thin wrapper — implementation moved to scripts/legacy/generate_feature_showcase.py."""
import runpy, sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
runpy.run_path(os.path.join(os.path.dirname(__file__), "legacy", "generate_feature_showcase.py"), run_name="__main__")
