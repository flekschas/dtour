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
    camera_zoom = t.Float(1.0).tag(sync=True)
    view_mode = t.Enum(["guided", "manual", "grand"], default_value="guided").tag(sync=True)
    theme = t.Enum(["light", "dark", "system"], default_value="dark").tag(sync=True)
    metric_bar_width = t.Union(
        [t.Int(), t.Unicode()],
        default_value="full",
    ).tag(sync=True)

    # ── Selection state (JS → Python) ───────────────────────────────────
    selected_labels = t.List(t.Unicode(), default_value=[]).tag(sync=True)

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
        self.send({"type": "views", "n_dims": self._n_dims}, buffers=[self._views_buf])

    def set_metrics(self, metric_result: MetricResult) -> None:
        """Send quality metrics to the JS frontend for radial chart display."""
        self._metrics_buf = metric_result.to_arrow_ipc()
        self.send({"type": "metrics"}, buffers=[self._metrics_buf])

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
