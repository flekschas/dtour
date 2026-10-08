"""anywidget-based Widget for Jupyter / Marimo."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anywidget
import traitlets as t

from .data import _to_ipc_bytes

if TYPE_CHECKING:
    from .metrics import MetricResult
    from .tours import TourResult

_STATIC = Path(__file__).parent / "static"
_BUNDLE = _STATIC / "widget.js"
# Most gallery previews the viewer shows.
_MAX_PREVIEW_COUNT = 32

# Repo root for a checkout (dtour/ → src/ → python/ → packages/ → root). Only a
# checkout has the bundle's inputs, where the locally built bundle can go stale.
_REPO = (Path(__file__).parent / "../../../..").resolve()
_BUNDLE_INPUTS = (
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "packages/scatter/package.json",
    "packages/scatter/vite.config.ts",
    "packages/scatter/src",
    "packages/viewer/package.json",
    "packages/viewer/vite.config.ts",
    "packages/viewer/src",
    "packages/python/package.json",
    "packages/python/vite.config.ts",
    "packages/python/js",
)


def _check_bundle() -> None:
    build_hint = "Build it from the repository root with `pnpm build:widget`."
    if not _BUNDLE.exists():
        raise FileNotFoundError(f"The dtour widget bundle {_BUNDLE} is missing. {build_hint}")
    if not (_REPO / "packages" / "python" / "js").is_dir():
        return
    paths = [_REPO / p for p in _BUNDLE_INPUTS]
    inputs = [f for p in paths for f in (p, *p.rglob("*")) if f.is_file()]
    if max(f.stat().st_mtime for f in inputs) > _BUNDLE.stat().st_mtime:
        warnings.warn(f"The dtour widget bundle is older than its sources. {build_hint}")


_check_bundle()


# Renamed traitlets: old name → (new name, old value → new value, new value → old value)
_RENAMED_TRAITS: dict[str, tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = {
    "show_keyframe_loadings": (
        "preview_label_content",
        lambda show: "auto" if show else "description",
        lambda content: content != "description",
    ),
    "theme": ("theme_mode", lambda value: value, lambda value: value),
}


def _warn_renamed(old: str, stacklevel: int) -> None:
    new = _RENAMED_TRAITS[old][0]
    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def _renamed_trait(old: str) -> property:
    """Deprecated alias of a renamed traitlet."""
    new, to_new, to_old = _RENAMED_TRAITS[old]

    def getter(self: Widget) -> Any:
        _warn_renamed(old, stacklevel=2)
        return to_old(getattr(self, new))

    def setter(self: Widget, value: Any) -> None:
        _warn_renamed(old, stacklevel=2)
        setattr(self, new, to_new(value))

    return property(getter, setter, doc=f"Deprecated alias of :attr:`{new}`.")


class Widget(anywidget.AnyWidget):
    """Interactive dtour scatter widget for Jupyter / Marimo.

    Binary data (Arrow IPC, tour keyframes, metrics) is sent via custom messages
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

    _esm = _BUNDLE
    # CSS is inlined into the JS bundle and injected into the Shadow DOM
    # at runtime — no separate _css file needed.

    # ── DtourSpec fields (flat traitlets, snake_case) ────────────────────
    tour_by = t.Enum(["dimensions", "pca", "parameter"], default_value="dimensions").tag(sync=True)
    tour_position = t.Float(0.0, allow_none=True).tag(sync=True)
    tour_playing = t.Bool(False).tag(sync=True)
    tour_speed = t.Float(1.0).tag(sync=True)
    tour_direction = t.Enum(["forward", "backward"], default_value="forward").tag(sync=True)
    tour_slider_spacing = t.Enum(["equal", "geodesic"], default_value="equal").tag(sync=True)
    tour_slider_visibility = t.Enum(["visible", "subtle", "hidden"], default_value="visible").tag(
        sync=True
    )
    preview_count = t.Int(4).tag(sync=True)
    preview_size = t.Enum(["auto", "small", "medium", "large"], default_value="auto").tag(sync=True)
    preview_padding = t.Float(12.0).tag(sync=True)
    preview_keyframe_numbers = t.Enum(["auto", "visible", "hidden"], default_value="auto").tag(
        sync=True
    )
    preview_label_content = t.Enum(["auto", "description", "loadings"], default_value="auto").tag(
        sync=True
    )
    preview_label_visibility = t.Enum(
        ["auto", "visible", "interactive", "hidden"], default_value="auto"
    ).tag(sync=True)
    point_size = t.Union(
        [t.Float(), t.Unicode()],
        default_value="auto",
    ).tag(sync=True)
    point_opacity = t.Union(
        [t.Float(), t.Unicode()],
        default_value="auto",
    ).tag(sync=True)
    min_point_size = t.Float(2.0).tag(sync=True)
    point_color = t.List(t.Float(), default_value=[0.25, 0.5, 0.9]).tag(sync=True)
    point_color_by = t.Unicode(allow_none=True, default_value=None).tag(sync=True)
    camera_pan_x = t.Float(0.0).tag(sync=True)
    camera_pan_y = t.Float(0.0).tag(sync=True)
    camera_zoom = t.Float(1 / 1.5).tag(sync=True)
    tour_traversal = t.Enum(["guided", "manual", "grand"], default_value="guided").tag(sync=True)
    show_legend = t.Bool(True).tag(sync=True)
    show_axes = t.Bool(False).tag(sync=True)
    show_tour_description = t.Bool(False).tag(sync=True)
    theme_mode = t.Enum(["light", "dark", "system"], default_value="dark").tag(sync=True)
    centering = t.Enum(["midrange", "mean"], default_value="midrange").tag(sync=True)
    metric_bar_width = t.Union(
        [t.Int(), t.Unicode()],
        default_value="full",
    ).tag(sync=True)

    # ── Projected columns ────────────────────────────────────────────────
    tour_dimensions = t.List(t.Unicode(), default_value=[]).tag(sync=True)
    # Synced as state, not with the keyframes message, so the frontend knows
    # the tour family on first render, before the keyframes arrive. Unset until
    # set_tour(), so a family embedded in the data can apply.
    _tour_family = t.Unicode(None, allow_none=True).tag(sync=True)

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
    @t.validate("tour_position")
    def _validate_tour_position(self, proposal: t.Bunch) -> float:
        value = proposal["value"]
        if value is None:
            return 0.0
        return float(value)

    @t.validate("min_point_size")
    def _validate_min_point_size(self, proposal: t.Bunch) -> float:
        value = proposal["value"]
        if not (1 <= value <= 20):
            raise t.TraitError(f"min_point_size must be between 1 and 20; got {value}")
        return value

    @t.validate("preview_count")
    def _validate_preview_count(self, proposal: t.Bunch) -> int:
        value = proposal["value"]
        if not (2 <= value <= _MAX_PREVIEW_COUNT):
            raise t.TraitError(
                f"preview_count must be between 2 and {_MAX_PREVIEW_COUNT}; got {value}"
            )
        return value

    @t.validate("preview_size")
    def _validate_preview_size(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("auto", "small", "medium", "large"):
            raise t.TraitError(
                f"preview_size must be 'auto', 'small', 'medium', or 'large'; got {value!r}"
            )
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
        # Auto-coerce to match the active tour's mode.  This handles stale
        # state round-tripped from the frontend (e.g. Marimo sends back the
        # initial "dimensions" default after set_tour already switched to
        # "parameter").
        tour = getattr(self, "_tour", None)
        if tour is not None:
            if value == "parameter" and tour.tour_family != "sequential":
                return "dimensions"
            if value != "parameter" and tour.tour_family == "sequential":
                return "parameter"
        return value

    @t.validate("tour_traversal")
    def _validate_tour_traversal(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("guided", "manual", "grand"):
            raise t.TraitError(
                f"tour_traversal must be 'guided', 'manual', or 'grand'; got {value!r}"
            )
        return value

    @t.validate("theme_mode")
    def _validate_theme_mode(self, proposal: t.Bunch) -> str:
        value = proposal["value"]
        if value not in ("light", "dark", "system"):
            raise t.TraitError(f"theme_mode must be 'light', 'dark', or 'system'; got {value!r}")
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
        for old in _RENAMED_TRAITS.keys() & kwargs.keys():
            _warn_renamed(old, stacklevel=2)
            new, to_new, _ = _RENAMED_TRAITS[old]
            kwargs.setdefault(new, to_new(kwargs.pop(old)))
        super().__init__(**kwargs)
        self._data_buf: bytes | None = None
        self._keyframes_buf: bytes | None = None
        self._keyframes_msg: dict | None = None
        self._metrics_buf: bytes | None = None
        self._tour: TourResult | None = None
        self.on_msg(self._handle_custom_msg)
        if data is not None:
            self.set_data(data)
        if tour is not None:
            self.set_tour(tour)

    # ── Public properties ────────────────────────────────────────────────
    @property
    def tour_family(self) -> str | None:
        """Family of the current tour (``"hyperdimensional"`` or ``"sequential"``).

        ``None`` until a tour is set. Read-only because it follows from the tour.
        """
        return self._tour_family

    show_keyframe_loadings = _renamed_trait("show_keyframe_loadings")
    theme = _renamed_trait("theme")

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
        """Set tour keyframes from a :class:`~dtour.tours.TourResult`."""
        self._keyframes_buf = tour.keyframes_raw

        msg: dict = {"type": "keyframes", "n_dims": tour.n_dims}

        if tour.description is not None:
            msg["tour_description"] = tour.description
        if tour.keyframe_descriptions is not None:
            msg["keyframe_descriptions"] = tour.keyframe_descriptions

        # Encode keyframe loadings (top-2 per keyframe) if available
        if tour.feature_loadings is not None and tour.feature_names is not None:
            loadings = tour.feature_loadings
            n_eigenvectors = loadings.shape[0]
            keyframe_loadings = []
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
            msg["keyframe_loadings"] = keyframe_loadings

        # Cache the full TourResult for save_spec_to_parquet. Set before
        # tour_by because its validator coerces against the active tour.
        self._tour = tour

        # Sync the family and tour_by together so the frontend never sees a
        # mismatched pair
        with self.hold_sync():
            self._tour_family = tour.tour_family or "hyperdimensional"
            if tour.tour_family == "sequential":
                self.tour_by = "parameter"
            elif self.tour_by == "parameter":
                self.tour_by = "dimensions"

        self._keyframes_msg = msg
        self.send(msg, buffers=[self._keyframes_buf])

        # Auto-set tour_dimensions from the tour's feature names
        if tour.feature_names is not None:
            self.tour_dimensions = tour.feature_names

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

    def select_by_labels(self, labels: list[str]) -> None:
        """Select points by categorical label names.

        Parameters
        ----------
        labels : list of str
            Label values to select. Resolves against the active color column
            on the JS frontend.
        """
        self.selected_labels = list(labels)

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

        from .spec import add_spec_to_parquet

        kwargs: dict = {}

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
        if self.tour_slider_spacing != "equal":
            kwargs["tour_slider_spacing"] = self.tour_slider_spacing
        if self.tour_slider_visibility != "visible":
            kwargs["tour_slider_visibility"] = self.tour_slider_visibility
        if self.preview_count != 4:
            kwargs["preview_count"] = self.preview_count
        if self.preview_size != "auto":
            kwargs["preview_size"] = self.preview_size
        if self.preview_padding != 12.0:
            kwargs["preview_padding"] = self.preview_padding
        if self.preview_keyframe_numbers != "auto":
            kwargs["preview_keyframe_numbers"] = self.preview_keyframe_numbers
        if self.preview_label_content != "auto":
            kwargs["preview_label_content"] = self.preview_label_content
        if self.preview_label_visibility != "auto":
            kwargs["preview_label_visibility"] = self.preview_label_visibility
        if self.point_size != "auto":
            kwargs["point_size"] = self.point_size
        if self.point_opacity != "auto":
            kwargs["point_opacity"] = self.point_opacity
        if self.min_point_size != 2.0:
            kwargs["min_point_size"] = self.min_point_size
        if self.point_color != [0.25, 0.5, 0.9]:
            kwargs["point_color"] = self.point_color
        if self.point_color_by:
            kwargs["point_color_by"] = self.point_color_by
        if self.camera_pan_x != 0.0:
            kwargs["camera_pan_x"] = self.camera_pan_x
        if self.camera_pan_y != 0.0:
            kwargs["camera_pan_y"] = self.camera_pan_y
        if self.camera_zoom != 1 / 1.5:
            kwargs["camera_zoom"] = self.camera_zoom
        if self.tour_traversal != "guided":
            kwargs["tour_traversal"] = self.tour_traversal
        if not self.show_legend:
            kwargs["show_legend"] = self.show_legend
        if self.show_axes:
            kwargs["show_axes"] = self.show_axes
        if self.show_tour_description:
            kwargs["show_tour_description"] = self.show_tour_description
        if self.theme_mode != "dark":
            kwargs["theme_mode"] = self.theme_mode
        if self.centering != "midrange":
            kwargs["centering"] = self.centering

        # Tour dimensions (written inside tour.dimensions by build_dtour_metadata)
        if self.tour_dimensions:
            kwargs["tour_dimensions"] = list(self.tour_dimensions)

        # Color map
        if self.color_map:
            simple_cm: dict[str, str] = {}
            for label, value in self.color_map.items():
                if isinstance(value, str):
                    simple_cm[label] = value
                elif isinstance(value, dict) and "dark" in value:
                    simple_cm[label] = value["dark"]
            if simple_cm:
                kwargs["point_color_map"] = simple_cm

        # Embed tour if available (use cached TourResult to preserve metadata)
        if hasattr(self, "_tour") and self._tour is not None:
            kwargs["tour"] = self._tour

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
        if self._keyframes_buf is not None:
            self.send(self._keyframes_msg, buffers=[self._keyframes_buf])
        if self._metrics_buf is not None:
            self.send({"type": "metrics"}, buffers=[self._metrics_buf])
