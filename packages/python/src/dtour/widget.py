"""anywidget-based Widget for Jupyter / Marimo."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import traitlets as t

from .data import _to_ipc_bytes

if TYPE_CHECKING:
    from .metrics import MetricResult
    from .tours import TourResult

_STATIC = Path(__file__).parent / "static"


class Widget(anywidget.AnyWidget):
    """Interactive dtour scatter widget for Jupyter / Marimo.

    Binary data (Arrow IPC, tour views, metrics) is sent via custom messages
    so it arrives as proper ArrayBuffer/DataView on the JS side — Marimo
    serialises ``Bytes`` traitlets as plain JSON ``number[]`` arrays, making
    them unusable for large binary payloads.

    The JS frontend signals readiness via ``model.send({ type: "ready" })``.
    Python receives it in ``on_msg`` and (re-)sends all binary buffers.

    Parameters
    ----------
    data:
        Any Arrow-compatible object (DataFrame, Arrow table, RecordBatch,
        or raw IPC bytes).  Anything with ``__arrow_c_stream__`` works.
    tour:
        A :class:`~dtour.tours.TourResult` providing basis matrices.

    Example
    -------
    >>> import dtour, numpy as np
    >>> X = np.random.randn(500, 5).astype(np.float32)
    >>> w = dtour.Widget(
    ...     data=dtour.data.from_numpy(X),
    ...     tour=dtour.little_tour(X),
    ... )
    >>> w
    """

    _esm = _STATIC / "widget.js"
    # CSS is inlined into the JS bundle and injected into the Shadow DOM
    # at runtime — no separate _css file needed.

    # ── DtourSpec fields (flat traitlets, snake_case) ────────────────────
    tour_by = t.Enum(["dimensions", "pca", "parameter"], default_value="dimensions").tag(sync=True)
    tour_position = t.Float(0.0).tag(sync=True)
    tour_playing = t.Bool(False).tag(sync=True)
    tour_speed = t.Float(1.0).tag(sync=True)
    tour_direction = t.Enum(["forward", "backward"], default_value="forward").tag(sync=True)
    preview_count = t.Int(4).tag(sync=True)
    preview_size = t.Enum(["small", "medium", "large"], default_value="large").tag(sync=True)
    preview_padding = t.Float(12.0).tag(sync=True)
    point_size = t.Union(
        [t.Float(), t.Unicode()],
        default_value="auto",
    ).tag(sync=True)
    point_opacity = t.Union(
        [t.Float(), t.Unicode()],
        default_value="auto",
    ).tag(sync=True)
    point_color = t.Union(
        [t.List(t.Float()), t.Unicode()],
        default_value=[0.25, 0.5, 0.9],
    ).tag(sync=True)
    camera_pan_x = t.Float(0.0).tag(sync=True)
    camera_pan_y = t.Float(0.0).tag(sync=True)
    camera_zoom = t.Float(1 / 1.5).tag(sync=True)
    view_mode = t.Enum(["guided", "manual", "grand"], default_value="guided").tag(sync=True)
    show_legend = t.Bool(True).tag(sync=True)
    show_frame_loadings = t.Bool(True).tag(sync=True)
    show_tour_description = t.Bool(False).tag(sync=True)
    theme = t.Enum(["light", "dark", "system"], default_value="dark").tag(sync=True)
    metric_bar_width = t.Union(
        [t.Int(), t.Unicode()],
        default_value="full",
    ).tag(sync=True)

    # ── Selection state (bidirectional) ───────────────────────────────────
    selected_labels = t.List(t.Unicode(), default_value=[]).tag(sync=True)
    selected_indices = t.List(t.Int(), default_value=[]).tag(sync=True)

    # ── Color map ────────────────────────────────────────────────────────
    color_map = t.Dict(default_value={}).tag(sync=True)

    # ── Metric track configuration ─────────────────────────────────────
    metric_tracks = t.List(t.Dict(), default_value=[]).tag(sync=True)

    # ── Layout ───────────────────────────────────────────────────────────
    height = t.Int(720).tag(sync=True)

    # ── Validators ───────────────────────────────────────────────────────
    @t.validate("preview_count")
    def _validate_preview_count(self, proposal: t.Bunch) -> int:
        value = proposal["value"]
        if value not in (4, 8, 12, 16):
            raise t.TraitError(f"preview_count must be 4, 8, 12, or 16; got {value}")
        return value

    @t.validate("preview_size")
    def _validate_preview_size(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("small", "medium", "large"):
            raise t.TraitError(f"preview_size must be 'small', 'medium', or 'large'; got {value!r}")
        return value

    @t.validate("tour_direction")
    def _validate_tour_direction(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("forward", "backward"):
            raise t.TraitError(f"tour_direction must be 'forward' or 'backward'; got {value!r}")
        return value

    @t.validate("tour_by")
    def _validate_tour_by(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("dimensions", "pca", "parameter"):
            raise t.TraitError(
                f"tour_by must be 'dimensions', 'pca', or 'parameter'; got {value!r}"
            )
        # Enforce consistency between tour_by and the active tour's tour_mode
        tour = getattr(self, "_tour", None)
        if tour is not None:
            if value == "parameter" and tour.tour_mode != "parameter":
                raise t.TraitError(
                    "tour_by='parameter' requires a parameter tour; "
                    f"the active tour has tour_mode={tour.tour_mode!r}"
                )
            if value != "parameter" and tour.tour_mode == "parameter":
                raise t.TraitError(
                    f"tour_by={value!r} is not allowed when the active tour is a "
                    "parameter tour; tour_by must be 'parameter'"
                )
        return value

    @t.validate("view_mode")
    def _validate_view_mode(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("guided", "manual", "grand"):
            raise t.TraitError(f"view_mode must be 'guided', 'manual', or 'grand'; got {value!r}")
        return value

    @t.validate("theme")
    def _validate_theme(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("light", "dark", "system"):
            raise t.TraitError(f"theme must be 'light', 'dark', or 'system'; got {value!r}")
        return value

    @t.validate("metric_bar_width")
    def _validate_metric_bar_width(self, proposal: t.Bunch) -> int | str:
        value = proposal["value"]
        if isinstance(value, str) and value != "full":
            raise t.TraitError(f"metric_bar_width must be 'full' or a positive int; got {value!r}")
        if isinstance(value, int) and value <= 0:
            raise t.TraitError(f"metric_bar_width must be positive; got {value}")
        return value

    # ── Init ─────────────────────────────────────────────────────────────
    def __init__(self, *, data: object | None = None, tour: TourResult | None = None, **kwargs):
        super().__init__(**kwargs)
        self._data_buf: bytes | None = None
        self._views_buf: bytes | None = None
        self._metrics_buf: bytes | None = None
        self._tour: TourResult | None = None
        self._n_dims: int = 0
        self.on_msg(self._handle_custom_msg)
        if data is not None:
            self.set_data(data)
        if tour is not None:
            self.set_tour(tour)

    # ── Public methods ───────────────────────────────────────────────────
    def set_data(self, data: object) -> None:
        """Load data from any Arrow-compatible source.

        Accepts anything with ``__arrow_c_stream__`` (pandas/polars
        DataFrames, pyarrow/arro3 Tables, etc.), raw ``bytes`` (Arrow IPC),
        or a file path.
        """
        self._data_buf = _to_ipc_bytes(data)
        self.send({"type": "data"}, buffers=[self._data_buf])

    def set_tour(self, tour: TourResult) -> None:
        """Set tour views from a :class:`~dtour.tours.TourResult`."""
        self._views_buf = tour.views_raw
        self._n_dims = tour.n_dims

        msg: dict = {"type": "views", "n_dims": self._n_dims}

        if tour.tour_mode is not None:
            msg["tour_mode"] = tour.tour_mode
        if tour.tour_description is not None:
            msg["tour_description"] = tour.tour_description
        if tour.tour_frame_description is not None:
            msg["tour_frame_description"] = tour.tour_frame_description
        if tour.frame_summaries is not None:
            msg["frame_summaries"] = tour.frame_summaries

        # Encode frame loadings (top-2 per frame) if available
        if tour.feature_loadings is not None and tour.feature_names is not None:
            loadings = tour.feature_loadings
            n_eigenvectors = loadings.shape[0]
            frame_loadings = []
            for i in range(tour.n_views):
                ev_idx = min(i + 1, n_eigenvectors - 1)
                row = loadings[ev_idx]
                top_k = abs(row).argsort()[::-1][:2]
                pairs = [
                    [tour.feature_names[j].rstrip("_"), round(float(row[j]), 6)] for j in top_k
                ]
                frame_loadings.append(pairs)
            msg["frame_loadings"] = frame_loadings

        self.send(msg, buffers=[self._views_buf])

        # Auto-switch tour_by to match the tour type
        if tour.tour_mode == "parameter":
            self.tour_by = "parameter"
        elif self.tour_by == "parameter":
            self.tour_by = "dimensions"

        # Cache the full TourResult for save_spec_to_parquet
        self._tour = tour

    def set_metrics(self, metric_result: MetricResult) -> None:
        """Send quality metrics to the JS frontend for radial chart display."""
        self._metrics_buf = metric_result.to_arrow_ipc()
        self.send({"type": "metrics"}, buffers=[self._metrics_buf])

    def select(self, indices: object) -> None:
        """Select points by index.

        Parameters
        ----------
        indices : array-like of int
            Point indices to select. Accepts numpy arrays, lists, or any
            iterable of non-negative integers. The JS frontend handles
            bit-packing internally.
        """
        import numpy as np

        idx = np.asarray(indices, dtype=np.int32)
        if idx.ndim != 1:
            raise ValueError("indices must be 1-dimensional")
        self.send({"type": "select"}, buffers=[idx.tobytes()])

    def clear_selection(self) -> None:
        """Clear the current point selection."""
        self.send({"type": "clear_selection"})

    def save_spec_to_parquet(self, table: object) -> object:
        """Save the widget's current spec + tour to Parquet file metadata.

        Reads the widget's current traitlet values and embeds them as a
        ``"dtour"`` key in the table's schema metadata.

        Parameters
        ----------
        table : Arrow-compatible table
            Any object with ``__arrow_c_stream__`` (pyarrow Table,
            polars DataFrame, arro3 Table, etc.).

        Returns
        -------
        arro3.core.Table
            A new table with embedded dtour configuration.

        Example
        -------
        >>> annotated = widget.save_spec_to_parquet(table)
        """
        import numpy as np

        from .spec import add_spec_to_parquet

        kwargs: dict = {}

        # Map preview_size (small/medium/large) → previewScale (0.5/0.75/1)
        _size_to_scale = {"small": 0.5, "medium": 0.75, "large": 1}

        # Only include non-default values
        if self.tour_by != "dimensions":
            kwargs["tour_by"] = self.tour_by
        if self.tour_position != 0.0:
            kwargs["tour_position"] = self.tour_position
        if self.tour_playing:
            kwargs["tour_playing"] = self.tour_playing
        if self.tour_speed != 1.0:
            kwargs["tour_speed"] = self.tour_speed
        if self.tour_direction != "forward":
            kwargs["tour_direction"] = self.tour_direction
        if self.preview_count != 4:
            kwargs["preview_count"] = self.preview_count
        if self.preview_size != "large":
            kwargs["preview_scale"] = _size_to_scale[self.preview_size]
        if self.preview_padding != 12.0:
            kwargs["preview_padding"] = self.preview_padding
        if self.point_size != "auto":
            kwargs["point_size"] = self.point_size
        if self.point_opacity != "auto":
            kwargs["point_opacity"] = self.point_opacity
        if self.point_color != [0.25, 0.5, 0.9]:
            kwargs["point_color"] = self.point_color
        if self.camera_pan_x != 0.0:
            kwargs["camera_pan_x"] = self.camera_pan_x
        if self.camera_pan_y != 0.0:
            kwargs["camera_pan_y"] = self.camera_pan_y
        if self.camera_zoom != 1 / 1.5:
            kwargs["camera_zoom"] = self.camera_zoom
        if self.view_mode != "guided":
            kwargs["view_mode"] = self.view_mode
        if not self.show_legend:
            kwargs["show_legend"] = self.show_legend
        if not self.show_frame_loadings:
            kwargs["show_frame_loadings"] = self.show_frame_loadings
        if self.show_tour_description:
            kwargs["show_tour_description"] = self.show_tour_description
        if self.theme != "dark":
            kwargs["theme_mode"] = self.theme

        # Color map
        if self.color_map:
            simple_cm: dict[str, str] = {}
            for label, value in self.color_map.items():
                if isinstance(value, str):
                    simple_cm[label] = value
                elif isinstance(value, dict) and "dark" in value:
                    simple_cm[label] = value["dark"]
            if simple_cm:
                kwargs["color_map"] = simple_cm

        # Embed tour if available (use cached TourResult to preserve metadata)
        if hasattr(self, "_tour") and self._tour is not None:
            kwargs["tour"] = self._tour
        elif self._views_buf is not None and self._n_dims > 0:
            from .tours import TourResult

            floats = np.frombuffer(self._views_buf, dtype=np.float32)
            stride = self._n_dims * 2
            n_views = len(floats) // stride
            views = [
                floats[i * stride : (i + 1) * stride].reshape(self._n_dims, 2)
                for i in range(n_views)
            ]
            kwargs["tour"] = TourResult(
                views=views,
                n_views=n_views,
                n_dims=self._n_dims,
            )

        return add_spec_to_parquet(table, **kwargs)

    # ── Custom message handler ──────────────────────────────────────────
    def _handle_custom_msg(self, data: dict, _buffers: list) -> None:
        """Handle messages from JS (2-arg signature for anywidget on_msg)."""
        if data.get("type") == "ready":
            self._send_all_buffers()

    def _send_all_buffers(self) -> None:
        """(Re-)send all cached binary buffers to the JS frontend."""
        if self._data_buf is not None:
            self.send({"type": "data"}, buffers=[self._data_buf])
        if self._views_buf is not None:
            self.send({"type": "views", "n_dims": self._n_dims}, buffers=[self._views_buf])
        if self._metrics_buf is not None:
            self.send({"type": "metrics"}, buffers=[self._metrics_buf])
