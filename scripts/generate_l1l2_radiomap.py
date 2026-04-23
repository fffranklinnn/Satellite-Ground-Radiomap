#!/usr/bin/env python3
"""Thin wrapper — implementation moved to scripts/legacy/generate_l1l2_radiomap.py."""
import runpy, sys, os
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
runpy.run_path(os.path.join(os.path.dirname(__file__), "legacy", "generate_l1l2_radiomap.py"), run_name="__main__")
