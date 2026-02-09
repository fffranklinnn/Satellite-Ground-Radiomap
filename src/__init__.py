"""SG-MRM: Satellite-Ground Multiscale Radio Map simulation system."""

__version__ = '0.1.0'
__author__ = 'SG-MRM Development Team'

from .core import Grid, free_space_path_loss, atmospheric_loss, ionospheric_loss
from .layers import BaseLayer, L1MacroLayer, L2TopoLayer, L3UrbanLayer
from .engine import RadioMapAggregator
from .utils import setup_logger, get_logger

__all__ = [
    'Grid',
    'free_space_path_loss',
    'atmospheric_loss',
    'ionospheric_loss',
    'BaseLayer',
    'L1MacroLayer',
    'L2TopoLayer',
    'L3UrbanLayer',
    'RadioMapAggregator',
    'setup_logger',
    'get_logger'
]
