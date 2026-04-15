"""
src/context package — unified spatial and temporal contracts.
"""

from .grid_spec import GridSpec
from .coverage_spec import CoverageSpec, BlendPolicy
from .time_utils import StrictModeError, parse_iso_utc, require_utc

__all__ = [
    "GridSpec",
    "CoverageSpec",
    "BlendPolicy",
    "StrictModeError",
    "parse_iso_utc",
    "require_utc",
]
