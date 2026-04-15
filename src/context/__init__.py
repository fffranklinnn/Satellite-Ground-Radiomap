"""
src/context package — unified spatial and temporal contracts.
"""

from .grid_spec import GridSpec
from .coverage_spec import CoverageSpec, BlendPolicy
from .time_utils import StrictModeError, parse_iso_utc, require_utc
from .frame_context import FrameContext, FrameMismatchError
from .layer_states import EntryWaveState, TerrainState, UrbanRefinementState
from .frame_builder import FrameBuilder

__all__ = [
    "GridSpec",
    "CoverageSpec",
    "BlendPolicy",
    "StrictModeError",
    "parse_iso_utc",
    "require_utc",
    "FrameContext",
    "FrameMismatchError",
    "FrameBuilder",
    "EntryWaveState",
    "TerrainState",
    "UrbanRefinementState",
]
