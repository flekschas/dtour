"""Helpers for embedding dtour configuration into Parquet file metadata.

The ``dtour`` key in Parquet key_value_metadata stores a JSON object with
DtourSpec fields (camelCase), an optional ``pointColorMap``, and an optional
``tour`` with base64-encoded Float32 keyframe bases.
"""

from __future__ import annotations

import base64
import json
import warnings
from typing import TYPE_CHECKING, Any

from .data import _is_numeric_field

if TYPE_CHECKING:
    from .tours import TourResult


def _to_camel(snake: str) -> str:
    """camelCase form of a snake_case name, e.g. ``camera_pan_x`` → ``cameraPanX``."""
    head, *rest = snake.split("_")
    return head + "".join(part.capitalize() for part in rest)


# Renamed keyword arguments: old name → new name and value.
_LEGACY_SPEC_KWARGS: dict[str, tuple[str, Any]] = {
    "preview_scale": (
        "preview_size",
        lambda v: {1: "large", 0.75: "medium", 0.5: "small"}.get(v, v),
    ),
    "show_keyframe_numbers": ("preview_keyframe_numbers", lambda v: "visible" if v else "hidden"),
    "show_keyframe_loadings": ("preview_label_content", lambda v: "auto" if v else "description"),
}


def _rename_legacy_kwargs(kwargs: dict[str, Any], stacklevel: int) -> dict[str, Any]:
    """Map renamed keyword arguments to their new names, with a deprecation warning.

    New names win over old ones. Unknown names pass through unchanged.
    """
    renamed = {k: v for k, v in kwargs.items() if k not in _LEGACY_SPEC_KWARGS}
    for old, (new, to_new) in _LEGACY_SPEC_KWARGS.items():
        if old not in kwargs:
            continue
        warnings.warn(
            f"`{old}` is deprecated; use `{new}` instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        if renamed.get(new) is None:
            renamed[new] = to_new(kwargs[old])
    return renamed


def _encode_tour(
    tour: TourResult,
    tour_dimensions: list[str] | None = None,
) -> dict[str, Any]:
    """Encode a TourResult as a JSON-serializable dict with base64 keyframes."""
    raw_bytes = tour.keyframes_raw
    b64 = base64.b64encode(raw_bytes).decode("ascii")

    # Dimensions name the columns the keyframes project. Feature names only do
    # so when the keyframes project the input features, not tour.embedding.
    dims = tour_dimensions
    if not dims and tour.embedding is None:
        dims = tour.feature_names
    if not dims:
        raise ValueError(
            "Cannot encode tour: dimensions are required. Provide tour_dimensions, "
            f"the names of the {tour.n_dims} columns the tour projects."
        )

    family = tour.tour_family or "hyperdimensional"

    # nViews/nDims are only needed by the parser to decode the base64 blob.
    # nDims must match the basis matrix row count (tour.n_dims), NOT len(dims).
    # The basis payload is stored as `nViews`, `nDims`, and `views`.
    result: dict[str, Any] = {
        "nViews": tour.n_keyframes,
        "nDims": tour.n_dims,
        "views": b64,
        "family": family,
        "dimensions": dims,
    }

    if tour.description is not None:
        result["description"] = tour.description

    if tour.keyframe_descriptions is not None:
        result["keyframeDescriptions"] = tour.keyframe_descriptions

    if tour.feature_loadings is not None and tour.feature_names is not None:
        loadings = tour.feature_loadings  # (n_components, n_features)
        n_eigenvectors = loadings.shape[0]
        keyframe_loadings: list[dict[str, list[Any]]] = []
        for i in range(tour.n_keyframes):
            ev_idx = min(i + 1, n_eigenvectors - 1)
            row = loadings[ev_idx]
            top_k = abs(row).argsort()[::-1][:2]
            names = [tour.feature_names[j].rstrip("_") for j in top_k]
            coeffs = [round(float(row[j]), 6) for j in top_k]
            keyframe_loadings.append(
                {
                    "primary": [names[0], coeffs[0]],
                    "secondary": [names[1], coeffs[1]],
                }
            )
        result["keyframeLoadings"] = keyframe_loadings

    return result


def build_dtour_metadata(
    *,
    tour_by: str | None = None,
    tour_position: float | None = None,
    tour_playing: bool | None = None,
    tour_speed: float | None = None,
    tour_direction: str | None = None,
    preview_count: int | None = None,
    preview_size: str | None = None,
    preview_padding: float | None = None,
    preview_keyframe_numbers: str | None = None,
    preview_label_content: str | None = None,
    preview_label_visibility: str | None = None,
    point_size: float | str | None = None,
    point_opacity: float | str | None = None,
    min_point_size: float | None = None,
    point_color: list[float] | None = None,
    point_color_by: str | list[str] | None = None,
    camera_pan_x: float | None = None,
    camera_pan_y: float | None = None,
    camera_zoom: float | None = None,
    tour_traversal: str | None = None,
    show_legend: bool | None = None,
    show_axes: bool | None = None,
    show_tour_description: bool | None = None,
    tour_slider_spacing: str | None = None,
    tour_slider_visibility: str | None = None,
    theme_mode: str | None = None,
    centering: str | None = None,
    point_color_map: dict[str, str] | None = None,
    point_color_map_2d: str | None = None,
    tour_dimensions: list[str] | None = None,
    tour: TourResult | None = None,
    **legacy_kwargs: Any,
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
        Number of gallery previews (2-32).
    preview_size : str, optional
        ``"auto"``, ``"small"``, ``"medium"``, or ``"large"``.
    preview_padding : float, optional
        Padding between previews in px.
    preview_keyframe_numbers : str, optional
        Keyframe numbers on previews: ``"auto"`` (only when some keyframes
        have no preview), ``"visible"``, or ``"hidden"``.
    preview_label_content : str, optional
        Preview label content: ``"auto"`` (feature loadings when available,
        else the keyframe description), ``"description"``, or ``"loadings"``.
    preview_label_visibility : str, optional
        When preview labels show: ``"auto"`` (``"visible"`` up to 16 previews,
        ``"interactive"`` above), ``"visible"``, ``"interactive"`` (on hover
        and for the current keyframe), or ``"hidden"``.
    point_size : float or str, optional
        Point size in pixels, or ``"auto"`` for density-adaptive.
    point_opacity : float or str, optional
        Point opacity 0-1, or ``"auto"``.
    min_point_size : float, optional
        Smallest point size in pixels (1-20) when ``point_size`` is ``"auto"``.
    point_color : list[float], optional
        Uniform point color as ``[r, g, b]`` tuple (0-1).
    point_color_by : str or list[str], optional
        Column name for per-point color encoding, or an ``[x, y]`` pair of
        numeric columns for a 2D colormap.
    camera_pan_x : float, optional
        Horizontal camera pan.
    camera_pan_y : float, optional
        Vertical camera pan.
    camera_zoom : float, optional
        Camera zoom level.
    tour_traversal : str, optional
        ``"guided"``, ``"manual"``, or ``"grand"``.
    show_legend : bool, optional
        Whether the legend panel is visible.
    show_axes : bool, optional
        Whether the axis biplot is visible in guided mode.
    show_tour_description : bool, optional
        Whether the tour description sub-bar is visible.
    tour_slider_spacing : str, optional
        ``"equal"`` or ``"geodesic"``.
    tour_slider_visibility : str, optional
        ``"visible"``, ``"subtle"``, or ``"hidden"``.
    theme_mode : str, optional
        ``"light"``, ``"dark"``, or ``"system"``.
    centering : str, optional
        ``"midrange"`` (default, ``(min+max)/2``) or ``"mean"`` (center of mass).
    point_color_map : dict, optional
        Label → hex color string mapping.
    point_color_map_2d : str, optional
        2D colormap when ``point_color_by`` is a column pair: ``"schumann"``,
        ``"bremm"``, ``"steiger"``, ``"ziegler"``, ``"teulingfig2"``,
        ``"cubediagonal"``, or ``"oklab_polar"``.
    tour_dimensions : list[str], optional
        Numeric column names that participate in the tour. Written as
        ``tour.dimensions`` in the JSON metadata. When omitted, a *tour*
        that projects its input features uses its ``feature_names``. Tours
        with an ``embedding`` project the embedding columns, so pass their
        names here (:func:`add_spec_to_parquet` infers them from the table).
    tour : TourResult, optional
        Tour result to embed (keyframes are base64-encoded).
    **legacy_kwargs
        Deprecated names: ``preview_scale``, ``show_keyframe_numbers``, and
        ``show_keyframe_loadings``.

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
        "preview_size": preview_size,
        "preview_padding": preview_padding,
        "preview_keyframe_numbers": preview_keyframe_numbers,
        "preview_label_content": preview_label_content,
        "preview_label_visibility": preview_label_visibility,
        "point_size": point_size,
        "point_opacity": point_opacity,
        "min_point_size": min_point_size,
        "point_color": point_color,
        "point_color_by": point_color_by,
        "point_color_map": point_color_map,
        "point_color_map_2d": point_color_map_2d,
        "camera_pan_x": camera_pan_x,
        "camera_pan_y": camera_pan_y,
        "camera_zoom": camera_zoom,
        "tour_traversal": tour_traversal,
        "show_legend": show_legend,
        "show_axes": show_axes,
        "show_tour_description": show_tour_description,
        "tour_slider_spacing": tour_slider_spacing,
        "tour_slider_visibility": tour_slider_visibility,
        "theme_mode": theme_mode,
        "centering": centering,
    }

    for key, value in _rename_legacy_kwargs(legacy_kwargs, stacklevel=2).items():
        if key not in spec_kwargs:
            raise TypeError(f"build_dtour_metadata() got an unexpected keyword argument {key!r}")
        if spec_kwargs[key] is None:
            spec_kwargs[key] = value

    for snake_key, value in spec_kwargs.items():
        if value is not None:
            config[_to_camel(snake_key)] = value

    if tour is not None:
        config["tour"] = _encode_tour(tour, tour_dimensions)

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
    kwargs = _rename_legacy_kwargs(kwargs, stacklevel=2)

    tour = kwargs.get("tour")
    if tour is not None and tour.embedding is not None and not kwargs.get("tour_dimensions"):
        names = tour.embedding_names or []
        if names and set(names) <= set(tbl.column_names):
            dims = names
        else:
            # Without its names, the viewer projects the first n_dims numeric columns
            dims = [f.name for f in tbl.schema if _is_numeric_field(f)][: tour.n_dims]
            if len(dims) < tour.n_dims:
                raise ValueError(
                    f"The tour projects {tour.n_dims} columns but the table has only "
                    f"{len(dims)} numeric columns. Add the tour.embedding columns first."
                )
        kwargs["tour_dimensions"] = dims

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
