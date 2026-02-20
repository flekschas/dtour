"""anywidget-based DtourWidget for Jupyter / Marimo."""

from __future__ import annotations

import anywidget
import traitlets as t


class DtourWidget(anywidget.AnyWidget):
    """Interactive dtour scatter widget for Jupyter.

    Parameters
    ----------
    data:
        Arrow IPC bytes containing the dataset (use :func:`dtour.data.from_numpy`
        or :func:`dtour.data.from_pandas` to create these).
    bases:
        Arrow IPC bytes containing the tour basis matrices
        (use :func:`dtour.tours.little_tour` to compute).
    view_count:
        Number of projection views displayed in the gallery. One of 4, 8, 12, 16.

    Example
    -------
    >>> import dtour, pandas as pd
    >>> df = pd.read_parquet("penguins.parquet")
    >>> data_bytes = dtour.data.from_pandas(df)
    >>> tour = dtour.little_tour(df)
    >>> widget = dtour.DtourWidget(data=data_bytes, bases=tour.bases)
    >>> widget
    """

    # JS bundle — set at build time to the bundled viewer JS content.
    # anywidget reads _esm to mount the frontend.
    _esm = t.Unicode("// dtour viewer JS not yet built").tag(sync=True)
    _css = t.Unicode("").tag(sync=True)

    # ── Data (binary traits — no JSON serialization for large buffers) ────────
    data = t.Bytes(b"").tag(sync=True)
    bases = t.Bytes(b"").tag(sync=True)
    quality_metrics = t.Bytes(b"").tag(sync=True)
    selected_points = t.Bytes(b"").tag(sync=True)

    # ── UI state ──────────────────────────────────────────────────────────────
    tour_position = t.Float(0.0, min=0.0, max=1.0).tag(sync=True)
    view_count = t.Int(8).tag(sync=True)
    point_style = t.Dict(
        default_value={
            "pointSize": 0.012,
            "opacity": 0.7,
            "color": [0.25, 0.5, 0.9],
        }
    ).tag(sync=True)

    @t.validate("view_count")
    def _validate_view_count(self, proposal: t.TraitError) -> int:
        value = proposal["value"]
        if value not in (4, 8, 12, 16):
            raise t.TraitError(f"view_count must be 4, 8, 12, or 16; got {value}")
        return value

    def __init__(self, data: bytes = b"", bases: bytes = b"", **kwargs):
        super().__init__(data=data, bases=bases, **kwargs)
