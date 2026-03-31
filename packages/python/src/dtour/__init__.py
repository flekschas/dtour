"""dtour — guided tours through high-dimensional data."""

from .metrics import MetricResult, compute_metrics
from .palettes import build_color_map
from .spec import add_spec_to_parquet, build_dtour_metadata, read_spec_from_parquet
from .tours import TourResult, le_tour, little_tour, umap_little_tour
from .widget import Widget

__all__ = [
    "MetricResult",
    "TourResult",
    "Widget",
    "add_spec_to_parquet",
    "build_color_map",
    "build_dtour_metadata",
    "compute_metrics",
    "le_tour",
    "little_tour",
    "read_spec_from_parquet",
    "umap_little_tour",
]
