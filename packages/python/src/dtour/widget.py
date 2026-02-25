"""anywidget-based DtourWidget for Jupyter / Marimo."""

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


class DtourWidget(anywidget.AnyWidget):
    """Interactive dtour scatter widget for Jupyter / Marimo.

    Data is sent via custom messages (not binary traitlets) so it works
    reliably in both Jupyter and Marimo.  All programmable settings are
    exposed as individual traitlets that map 1:1 to the JS ``DtourSpec``.

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
    >>> w = dtour.DtourWidget(
    ...     data=dtour.data.from_numpy(X),
    ...     tour=dtour.little_tour(X),
    ... )
    >>> w
    """

    _esm = _STATIC / "widget.js"
    _css = _STATIC / "widget.css"

    # ── DtourSpec fields (flat traitlets, snake_case) ────────────────────
    tour_position = t.Float(0.0).tag(sync=True)
    tour_playing = t.Bool(False).tag(sync=True)
    tour_speed = t.Float(1.0).tag(sync=True)
    tour_direction = t.Unicode("forward").tag(sync=True)
    preview_count = t.Int(4).tag(sync=True)
    preview_padding = t.Float(12.0).tag(sync=True)
    point_size = t.Float(0.012).tag(sync=True)
    point_opacity = t.Float(0.7).tag(sync=True)
    point_color = t.Union(
        [t.List(t.Float()), t.Unicode()],
        default_value=[0.25, 0.5, 0.9],
    ).tag(sync=True)
    camera_pan_x = t.Float(0.0).tag(sync=True)
    camera_pan_y = t.Float(0.0).tag(sync=True)
    camera_zoom = t.Float(1.0).tag(sync=True)

    # ── Layout ───────────────────────────────────────────────────────────
    height = t.Int(600).tag(sync=True)

    # ── Validators ───────────────────────────────────────────────────────
    @t.validate("preview_count")
    def _validate_preview_count(self, proposal: t.Bunch) -> int:
        value = proposal["value"]
        if value not in (4, 8, 12, 16):
            raise t.TraitError(f"preview_count must be 4, 8, 12, or 16; got {value}")
        return value

    @t.validate("tour_direction")
    def _validate_tour_direction(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("forward", "backward"):
            raise t.TraitError(f"tour_direction must be 'forward' or 'backward'; got {value!r}")
        return value

    # ── Init ─────────────────────────────────────────────────────────────
    def __init__(self, *, data: object | None = None, tour: TourResult | None = None, **kwargs):
        super().__init__(**kwargs)
        self._data_buf: bytes | None = None
        self._bases_buf: bytes | None = None
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
        """Set tour bases from a :class:`~dtour.tours.TourResult`."""
        self._bases_buf = tour.bases_raw
        self._n_dims = tour.n_dims
        self.send({"type": "bases", "n_dims": self._n_dims}, buffers=[self._bases_buf])

    def set_metrics(self, metric_result: MetricResult) -> None:
        """Send quality metrics to the JS frontend for radial chart display."""
        self._metrics_buf = metric_result.to_arrow_ipc()
        self.send({"type": "metrics"}, buffers=[self._metrics_buf])

    # ── Custom message handler ───────────────────────────────────────────
    def _handle_custom_msg(self, _widget: object, content: dict, _buffers: list) -> None:
        """Re-send data, bases, and metrics when the JS frontend signals it is ready."""
        if content.get("type") != "ready":
            return
        if self._data_buf is not None:
            self.send({"type": "data"}, buffers=[self._data_buf])
        if self._bases_buf is not None:
            self.send({"type": "bases", "n_dims": self._n_dims}, buffers=[self._bases_buf])
        if self._metrics_buf is not None:
            self.send({"type": "metrics"}, buffers=[self._metrics_buf])
