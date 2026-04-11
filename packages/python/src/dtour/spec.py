"""Helpers for embedding dtour configuration into Parquet file metadata.

The ``dtour`` key in Parquet key_value_metadata stores a JSON object with
DtourSpec fields (camelCase), an optional ``colorMap``, and an optional
``tour`` with base64-encoded Float32 view matrices.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tours import TourResult

# ── Snake → Camel conversion ────────────────────────────────────────────

_SNAKE_TO_CAMEL: dict[str, str] = {
    "tour_by": "tourBy",
    "tour_position": "tourPosition",
    "tour_playing": "tourPlaying",
    "tour_speed": "tourSpeed",
    "tour_direction": "tourDirection",
    "preview_count": "previewCount",
    "preview_scale": "previewScale",
    "preview_padding": "previewPadding",
    "point_size": "pointSize",
    "point_opacity": "pointOpacity",
    "point_color": "pointColor",
    "camera_pan_x": "cameraPanX",
    "camera_pan_y": "cameraPanY",
    "camera_zoom": "cameraZoom",
    "view_mode": "viewMode",
    "show_legend": "showLegend",
    "show_axes": "showAxes",
    "show_frame_numbers": "showFrameNumbers",
    "show_frame_loadings": "showFrameLoadings",
    "show_tour_description": "showTourDescription",
    "slider_spacing": "sliderSpacing",
    "theme_mode": "themeMode",
}

_CAMEL_TO_SNAKE: dict[str, str] = {v: k for k, v in _SNAKE_TO_CAMEL.items()}

_SPEC_KEYS = set(_SNAKE_TO_CAMEL.values())


def _encode_tour(tour: TourResult) -> dict[str, Any]:
    """Encode a TourResult as a JSON-serializable dict with base64 views."""
    raw_bytes = tour.views_raw
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    result: dict[str, Any] = {
        "nDims": tour.n_dims,
        "nViews": tour.n_views,
        "views": b64,
    }

    if tour.tour_mode is not None:
        result["tourMode"] = tour.tour_mode

    if tour.tour_description is not None:
        result["tourDescription"] = tour.tour_description

    if tour.tour_frame_description is not None:
        result["tourFrameDescription"] = tour.tour_frame_description

    if tour.frame_summaries is not None:
        result["frameSummaries"] = tour.frame_summaries

    if tour.feature_loadings is not None and tour.feature_names is not None:
        loadings = tour.feature_loadings  # (n_components, n_features)
        n_eigenvectors = loadings.shape[0]
        frame_loadings: list[list[list[Any]]] = []
        for i in range(tour.n_views):
            ev_idx = min(i + 1, n_eigenvectors - 1)
            row = loadings[ev_idx]
            top_k = abs(row).argsort()[::-1][:2]
            pairs: list[list[Any]] = []
            for j in top_k:
                name = tour.feature_names[j].rstrip("_")
                pairs.append([name, round(float(row[j]), 6)])
            frame_loadings.append(pairs)
        result["frameLoadings"] = frame_loadings

    return result


def build_dtour_metadata(
    *,
    tour_by: str | None = None,
    tour_position: float | None = None,
    tour_playing: bool | None = None,
    tour_speed: float | None = None,
    tour_direction: str | None = None,
    preview_count: int | None = None,
    preview_scale: float | None = None,
    preview_padding: float | None = None,
    point_size: float | str | None = None,
    point_opacity: float | str | None = None,
    point_color: str | list[float] | None = None,
    camera_pan_x: float | None = None,
    camera_pan_y: float | None = None,
    camera_zoom: float | None = None,
    view_mode: str | None = None,
    show_legend: bool | None = None,
    show_axes: bool | None = None,
    show_frame_numbers: bool | None = None,
    show_frame_loadings: bool | None = None,
    show_tour_description: bool | None = None,
    slider_spacing: str | None = None,
    theme_mode: str | None = None,
    color_map: dict[str, str] | None = None,
    tour: TourResult | None = None,
) -> str:
    """Build a JSON string for the Parquet ``dtour`` key_value_metadata.

    Parameters
    ----------
    tour_by : str, optional
        ``"dimensions"`` or ``"pca"``.
    tour_position : float, optional
        Tour position 0-1.
    tour_playing : bool, optional
        Whether the tour is animating.
    tour_speed : float, optional
        Animation speed multiplier 0.1-5.
    tour_direction : str, optional
        ``"forward"`` or ``"backward"``.
    preview_count : int, optional
        Number of gallery previews (4, 8, 12, or 16).
    preview_scale : float, optional
        Preview size (1, 0.75, or 0.5).
    preview_padding : float, optional
        Padding between previews in px.
    point_size : float or str, optional
        Point size in pixels, or ``"auto"`` for density-adaptive.
    point_opacity : float or str, optional
        Point opacity 0-1, or ``"auto"``.
    point_color : str or list[float], optional
        Column name for color encoding, or ``[r, g, b]`` tuple (0-1).
    camera_pan_x : float, optional
        Horizontal camera pan.
    camera_pan_y : float, optional
        Vertical camera pan.
    camera_zoom : float, optional
        Camera zoom level.
    view_mode : str, optional
        ``"guided"``, ``"manual"``, or ``"grand"``.
    show_legend : bool, optional
        Whether the legend panel is visible.
    show_axes : bool, optional
        Whether the axis biplot is visible in guided mode.
    show_frame_numbers : bool, optional
        Whether frame numbers are shown on preview thumbnails.
    show_frame_loadings : bool, optional
        Whether feature loading pills are shown on preview thumbnails.
    show_tour_description : bool, optional
        Whether the tour description sub-bar is visible.
    slider_spacing : str, optional
        ``"equal"`` or ``"geodesic"``.
    theme_mode : str, optional
        ``"light"``, ``"dark"``, or ``"system"``.
    color_map : dict, optional
        Label → hex color string mapping.
    tour : TourResult, optional
        Tour result to embed (views are base64-encoded).

    Returns
    -------
    str
        JSON string for the Parquet metadata value.
    """
    config: dict[str, Any] = {}

    spec_kwargs: dict[str, Any] = {
        "tour_by": tour_by,
        "tour_position": tour_position,
        "tour_playing": tour_playing,
        "tour_speed": tour_speed,
        "tour_direction": tour_direction,
        "preview_count": preview_count,
        "preview_scale": preview_scale,
        "preview_padding": preview_padding,
        "point_size": point_size,
        "point_opacity": point_opacity,
        "point_color": point_color,
        "camera_pan_x": camera_pan_x,
        "camera_pan_y": camera_pan_y,
        "camera_zoom": camera_zoom,
        "view_mode": view_mode,
        "show_legend": show_legend,
        "show_axes": show_axes,
        "show_frame_numbers": show_frame_numbers,
        "show_frame_loadings": show_frame_loadings,
        "show_tour_description": show_tour_description,
        "slider_spacing": slider_spacing,
        "theme_mode": theme_mode,
    }

    for snake_key, value in spec_kwargs.items():
        if value is not None:
            camel_key = _SNAKE_TO_CAMEL[snake_key]
            config[camel_key] = value

    if color_map is not None:
        config["colorMap"] = color_map

    if tour is not None:
        config["tour"] = _encode_tour(tour)

    return json.dumps(config, separators=(",", ":"))


def add_spec_to_parquet(
    table: object,
    **kwargs: Any,
) -> object:
    """Add dtour spec metadata to an Arrow table.

    Accepts any Arrow-compatible table (pyarrow, polars, arro3, etc.).
    Returns a new table with the ``"dtour"`` key set in schema metadata;
    the original table is unchanged.

    Parameters
    ----------
    table : Arrow-compatible table
        Any object implementing ``__arrow_c_stream__`` (pyarrow Table,
        polars DataFrame, arro3 Table, etc.).
    **kwargs
        All keyword arguments accepted by :func:`build_dtour_metadata`.

    Returns
    -------
    arro3.core.Table
        A new table with embedded dtour configuration.

    Example
    -------
    >>> table = dtour.add_spec_to_parquet(table, point_size=2, tour_by="pca")
    """
    import arro3.core as ac

    if not hasattr(table, "__arrow_c_stream__"):
        raise TypeError(
            f"Expected an Arrow-compatible table, got {type(table).__name__}. "
            "Pass an object with __arrow_c_stream__ "
            "(pyarrow Table, polars DataFrame, arro3 Table, etc.)."
        )

    tbl = ac.Table.from_arrow(table)
    dtour_json = build_dtour_metadata(**kwargs)

    # Merge with existing metadata (preserving other keys like pandas schema)
    existing = dict(tbl.schema.metadata_str) if tbl.schema.metadata else {}
    existing["dtour"] = dtour_json
    new_schema = tbl.schema.with_metadata(existing)

    return tbl.with_schema(new_schema)


def read_spec_from_parquet(table_or_path: object) -> dict[str, Any] | None:
    """Read the dtour spec from an Arrow table or Parquet file path.

    Parameters
    ----------
    table_or_path : Arrow table or str or Path
        An Arrow-compatible table or path to a Parquet file.

    Returns
    -------
    dict or None
        The parsed dtour config dict (camelCase keys), or ``None`` if
        not present.
    """
    from pathlib import Path

    import arro3.core as ac

    if isinstance(table_or_path, (str, Path)):
        import arro3.io

        reader = arro3.io.read_parquet(str(table_or_path))
        tbl = ac.Table.from_arrow(reader)
        kv = tbl.schema.metadata_str if tbl.schema.metadata else None
    elif hasattr(table_or_path, "__arrow_c_stream__"):
        tbl = ac.Table.from_arrow(table_or_path)
        kv = tbl.schema.metadata_str if tbl.schema.metadata else None
    else:
        raise TypeError(f"Expected an Arrow table or file path, got {type(table_or_path).__name__}")

    if not kv:
        return None

    raw = kv.get("dtour")
    if raw is None:
        return None

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
