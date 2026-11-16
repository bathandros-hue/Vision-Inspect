"""Reference-based industrial inspection baseline."""
from .core import Inspection, inspect, pixel_metrics
from .synthetic import synthetic_case

__all__ = ["Inspection", "inspect", "pixel_metrics", "synthetic_case"]
