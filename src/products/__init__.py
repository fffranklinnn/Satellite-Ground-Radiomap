"""
src/products package — simulation product types and provenance.
"""

from .manifest import ProductManifest
from .projectors import UnknownProductTypeError, project, export_dataset

__all__ = ["ProductManifest", "UnknownProductTypeError", "project", "export_dataset"]
