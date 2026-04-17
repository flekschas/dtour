"""dtour — guided tours through high-dimensional data."""

from .metrics import MetricResult, compute_metrics
from .palettes import build_color_map
from .spec import add_spec_to_parquet, build_dtour_metadata, read_spec_from_parquet
from .tours import (
    EmbeddingStep,
    TourResult,
    aligned_umap_tour,
    le_tour,
    little_tour,
    sequential_tour,
    spectrum_tour,
    umap_little_tour,
)
from .widget import Widget

__all__ = [
    "EmbeddingStep",
    "MetricResult",
    "TourResult",
    "Widget",
    "add_spec_to_parquet",
    "aligned_umap_tour",
    "build_color_map",
    "build_dtour_metadata",
    "compute_metrics",
    "le_tour",
    "little_tour",
    "read_spec_from_parquet",
    "sequential_tour",
    "spectrum_tour",
    "umap_little_tour",
]
