"""Tests for spec persistence: build, embed, and round-trip through Parquet."""

import json
import tempfile
from pathlib import Path

import arro3.core as ac
import arro3.io
import numpy as np
import polars as pl
import pytest
from dtour.spec import (
    add_spec_to_parquet,
    build_dtour_metadata,
    read_spec_from_parquet,
)
from dtour.tours import little_tour

# ── build_dtour_metadata ────────────────────────────────────────────────


def test_build_empty():
    result = json.loads(build_dtour_metadata())
    assert result == {}


def test_build_spec_fields():
    result = json.loads(
        build_dtour_metadata(
            point_size=3,
            point_color_by="label",
            camera_zoom=0.5,
            tour_by="pca",
            theme_mode="light",
        )
    )
    assert result["pointSize"] == 3
    assert result["pointColorBy"] == "label"
    assert result["cameraZoom"] == 0.5
    assert result["tourBy"] == "pca"
    assert result["themeMode"] == "light"
    # Only set keys should appear
    assert "previewCount" not in result


def test_build_color_map():
    cm = {"A": "#ff0000", "B": "#00ff00"}
    result = json.loads(build_dtour_metadata(point_color_map=cm))
    assert result["pointColorMap"] == cm


def test_build_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    tour = little_tour(X)
    result = json.loads(build_dtour_metadata(tour=tour, tour_dimensions=["a", "b", "c", "d"]))
    assert "tour" in result
    assert result["tour"]["nDims"] == 4
    assert result["tour"]["nViews"] == tour.n_keyframes
    assert isinstance(result["tour"]["views"], str)  # base64


def test_build_snake_to_camel_all_keys():
    """Every snake_case kwarg maps to the expected camelCase key."""
    kwargs = {
        "tour_by": "dimensions",
        "tour_position": 0.5,
        "tour_playing": True,
        "tour_speed": 2.0,
        "tour_direction": "backward",
        "preview_count": 8,
        "preview_size": "medium",
        "preview_padding": 16.0,
        "point_size": 4,
        "point_opacity": 0.8,
        "min_point_size": 3.0,
        "point_color": "col",
        "point_color_by": "label",
        "point_color_map": {"A": "#ff0000"},
        "camera_pan_x": 0.1,
        "camera_pan_y": -0.1,
        "camera_zoom": 1.5,
        "tour_traversal": "manual",
        "show_legend": False,
        "show_axes": True,
        "preview_keyframe_numbers": "visible",
        "preview_label_content": "description",
        "preview_label_visibility": "interactive",
        "show_tour_description": True,
        "tour_slider_spacing": "equal",
        "tour_slider_visibility": "subtle",
        "theme_mode": "light",
        "centering": "mean",
    }
    result = json.loads(build_dtour_metadata(**kwargs))
    assert result == {
        "tourBy": "dimensions",
        "tourPosition": 0.5,
        "tourPlaying": True,
        "tourSpeed": 2.0,
        "tourDirection": "backward",
        "previewCount": 8,
        "previewSize": "medium",
        "previewPadding": 16.0,
        "pointSize": 4,
        "pointOpacity": 0.8,
        "minPointSize": 3.0,
        "pointColor": "col",
        "pointColorBy": "label",
        "pointColorMap": {"A": "#ff0000"},
        "cameraPanX": 0.1,
        "cameraPanY": -0.1,
        "cameraZoom": 1.5,
        "tourTraversal": "manual",
        "showLegend": False,
        "showAxes": True,
        "previewKeyframeNumbers": "visible",
        "previewLabelContent": "description",
        "previewLabelVisibility": "interactive",
        "showTourDescription": True,
        "tourSliderSpacing": "equal",
        "tourSliderVisibility": "subtle",
        "themeMode": "light",
        "centering": "mean",
    }


def test_build_maps_deprecated_kwargs():
    with pytest.warns(DeprecationWarning) as record:
        result = json.loads(
            build_dtour_metadata(
                preview_scale=0.5,
                show_keyframe_numbers=True,
                show_keyframe_loadings=False,
            )
        )
    assert len(record) == 3
    assert result == {
        "previewSize": "small",
        "previewKeyframeNumbers": "visible",
        "previewLabelContent": "description",
    }


def test_build_prefers_new_kwarg_over_deprecated_one():
    with pytest.warns(DeprecationWarning):
        result = json.loads(build_dtour_metadata(preview_size="large", preview_scale=0.5))
    assert result == {"previewSize": "large"}


def test_build_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="not_a_setting"):
        build_dtour_metadata(not_a_setting=1)


# ── add_spec_to_parquet ─────────────────────────────────────────────────


def _make_table() -> ac.Table:
    return ac.Table.from_pydict(
        {"x": ac.Array.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))}
    )


def test_add_spec_sets_metadata():
    table = _make_table()
    result = add_spec_to_parquet(table, point_size=2, tour_by="pca")
    meta = result.schema.metadata_str
    assert "dtour" in meta
    parsed = json.loads(meta["dtour"])
    assert parsed["pointSize"] == 2
    assert parsed["tourBy"] == "pca"


def test_add_spec_preserves_existing_metadata():
    table = _make_table()
    schema_with_meta = table.schema.with_metadata({"existing_key": "existing_value"})
    table = table.with_schema(schema_with_meta)
    result = add_spec_to_parquet(table, point_color_by="label")
    meta = result.schema.metadata_str
    assert meta["existing_key"] == "existing_value"
    assert "dtour" in meta


def test_add_spec_rejects_non_arrow():
    with pytest.raises(TypeError, match="Arrow-compatible"):
        add_spec_to_parquet("not a table", point_size=2)


def test_add_spec_accepts_polars():
    """Polars DataFrames implement __arrow_c_stream__."""
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"a": [1, 2, 3]})
    result = add_spec_to_parquet(df, point_size=5)
    meta = result.schema.metadata_str
    assert json.loads(meta["dtour"])["pointSize"] == 5


# ── read_spec_from_parquet ──────────────────────────────────────────────


def test_add_spec_maps_deprecated_kwargs():
    with pytest.warns(DeprecationWarning, match="show_keyframe_loadings"):
        result = add_spec_to_parquet(_make_table(), show_keyframe_loadings=False)
    meta = json.loads(result.schema.metadata_str["dtour"])
    assert meta == {"previewLabelContent": "description"}


def test_read_from_table():
    table = add_spec_to_parquet(_make_table(), camera_zoom=0.5, theme_mode="light")
    result = read_spec_from_parquet(table)
    assert result is not None
    assert result["cameraZoom"] == 0.5
    assert result["themeMode"] == "light"


def test_read_from_file():
    table = add_spec_to_parquet(_make_table(), point_color_by="label")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.parquet"
        arro3.io.write_parquet(table, str(path))
        result = read_spec_from_parquet(path)
    assert result is not None
    assert result["pointColorBy"] == "label"


def test_read_returns_none_for_missing():
    table = _make_table()
    assert read_spec_from_parquet(table) is None


def test_read_returns_none_for_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plain.parquet"
        arro3.io.write_parquet(_make_table(), str(path))
        assert read_spec_from_parquet(path) is None


# ── Full round-trip: build → embed → write → read ──────────────────────


def test_round_trip_spec_and_color_map():
    cm = {"A": "#ff0000", "B": "#00ff00"}
    table = add_spec_to_parquet(
        _make_table(),
        point_size=3,
        point_color_by="label",
        point_color_map=cm,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rt.parquet"
        arro3.io.write_parquet(table, str(path))
        result = read_spec_from_parquet(path)
    assert result is not None
    assert result["pointSize"] == 3
    assert result["pointColorBy"] == "label"
    assert result["pointColorMap"] == cm


def test_round_trip_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    tour = little_tour(X)
    table = add_spec_to_parquet(_make_table(), tour=tour, tour_dimensions=["a", "b", "c", "d"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rt_tour.parquet"
        arro3.io.write_parquet(table, str(path))
        result = read_spec_from_parquet(path)
    assert result is not None
    assert result["tour"]["nDims"] == 4
    assert result["tour"]["nViews"] == tour.n_keyframes
    # Verify base64 decodes back to the right number of floats
    import base64

    floats = np.frombuffer(base64.b64decode(result["tour"]["views"]), dtype=np.float32)
    assert len(floats) == tour.n_keyframes * 4 * 2


# ── Widget.save_spec_to_parquet ─────────────────────────────────────────


def test_widget_save_spec_default_zoom():
    """Widget at default zoom should NOT write cameraZoom (matches viewer default)."""
    from dtour.widget import Widget

    w = Widget()
    table = w.save_spec_to_parquet(_make_table())
    meta = json.loads(table.schema.metadata_str["dtour"])
    assert "cameraZoom" not in meta


def test_widget_save_spec_custom_zoom():
    """Non-default zoom should be preserved."""
    from dtour.widget import Widget

    w = Widget(camera_zoom=2.0)
    table = w.save_spec_to_parquet(_make_table())
    meta = json.loads(table.schema.metadata_str["dtour"])
    assert meta["cameraZoom"] == 2.0


def test_widget_save_spec_with_tour():
    """Tour keyframes should be embedded when set."""
    from dtour.widget import Widget

    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    tour = little_tour(X)
    tour.feature_names = ["a", "b", "c", "d"]
    w = Widget(tour=tour)
    table = w.save_spec_to_parquet(pl.DataFrame(X, schema=["a", "b", "c", "d"]))
    meta = json.loads(table.schema.metadata_str["dtour"])
    assert "tour" in meta
    assert meta["tour"]["nDims"] == 4
    assert meta["tour"]["nViews"] == tour.n_keyframes


def _embedding_tour():
    from dtour.tours import TourResult

    rng = np.random.default_rng(42)
    emb = rng.standard_normal((50, 3)).astype(np.float32)
    keyframes = [np.eye(3, 2, k, dtype=np.float32) for k in (0, -1)]
    # Feature names describe the input data, not the embedding columns
    return TourResult(
        keyframes,
        n_dims=3,
        embedding=emb,
        feature_names=["in_a", "in_b", "in_c", "in_d"],
    )


def test_embedding_tour_requires_dimensions():
    with pytest.raises(ValueError, match="tour_dimensions"):
        build_dtour_metadata(tour=_embedding_tour())


def test_widget_save_spec_embedding_tour_dimensions():
    """Embedding tours record the projected columns, not the input features."""
    from dtour.widget import Widget

    tour = _embedding_tour()
    data = pl.DataFrame({"extra": np.arange(50, dtype=np.float32), "label": ["x"] * 50})
    w = Widget(data, tour)
    table = w.save_spec_to_parquet()
    names = ["embedding_0", "embedding_1", "embedding_2"]
    assert table.column_names == [*names, "extra", "label"]
    meta = json.loads(table.schema.metadata_str["dtour"])
    assert meta["tour"]["dimensions"] == names


def test_widget_save_spec_rejects_table_without_tour_columns():
    from dtour.widget import Widget

    tour = _embedding_tour()
    w = Widget(tour=tour)
    with pytest.raises(ValueError, match="lacks the columns"):
        w.save_spec_to_parquet(pl.DataFrame({"label": ["x"] * 50}))
