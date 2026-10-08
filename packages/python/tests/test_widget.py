"""Tests for widget instantiation and trait validation."""

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import dtour
import numpy as np
import pytest
from dtour import widget
from dtour.tours import TourResult, little_tour, sequential_tour
from dtour.widget import Widget


def test_widget_default_traits():
    w = Widget()
    assert w.tour_by == "dimensions"
    assert w.tour_position == 0.0
    assert w.tour_playing is False
    assert w.tour_speed == 1.0
    assert w.tour_direction == "forward"
    assert w.tour_slider_spacing == "equal"
    assert w.tour_slider_visibility == "visible"
    assert w.preview_count == 4
    assert w.preview_size == "auto"
    assert w.preview_padding == 12.0
    assert w.preview_keyframe_numbers == "auto"
    assert w.preview_label_content == "auto"
    assert w.preview_label_visibility == "auto"
    assert w.point_size == "auto"
    assert w.point_opacity == "auto"
    assert w.min_point_size == 2.0
    assert w.point_color == [0.25, 0.5, 0.9]
    assert w.camera_pan_x == 0.0
    assert w.camera_pan_y == 0.0
    assert w.camera_zoom == pytest.approx(1 / 1.5)
    assert w.tour_traversal == "guided"
    assert w.show_legend is True
    assert w.show_axes is False
    assert w.theme_mode == "dark"
    assert w.height == 720


def test_widget_preview_count_validation():
    w = Widget(preview_count=32)
    assert w.preview_count == 32

    with pytest.raises(Exception):
        Widget(preview_count=1)
    with pytest.raises(Exception):
        Widget(preview_count=33)


def test_widget_show_keyframe_loadings_is_deprecated():
    with pytest.warns(DeprecationWarning, match="preview_label_content"):
        w = Widget(show_keyframe_loadings=False)
    assert w.preview_label_content == "description"

    with pytest.warns(DeprecationWarning):
        w.show_keyframe_loadings = True
    assert w.preview_label_content == "auto"
    with pytest.warns(DeprecationWarning):
        assert w.show_keyframe_loadings is True


def test_widget_theme_is_deprecated():
    # Warnings point at the caller so they show up in notebooks
    with pytest.warns(DeprecationWarning, match="theme_mode") as record:
        w = Widget(theme="light")
    assert w.theme_mode == "light"
    assert record[0].filename == __file__

    with pytest.warns(DeprecationWarning) as record:
        w.theme = "system"
    assert w.theme_mode == "system"
    assert record[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        assert w.theme == "system"


def test_widget_min_point_size_validation():
    assert Widget(min_point_size=20).min_point_size == 20

    with pytest.raises(Exception):
        Widget(min_point_size=0.5)


def test_widget_tour_direction_validation():
    w = Widget(tour_direction="backward")
    assert w.tour_direction == "backward"

    with pytest.raises(Exception):
        Widget(tour_direction="sideways")


def test_widget_set_data_bytes():
    w = Widget()
    # Raw bytes passthrough — no conversion needed
    w.set_data(b"fake arrow ipc bytes")
    assert w._data_buf == b"fake arrow ipc bytes"


def test_widget_set_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    tour = little_tour(X)
    w = Widget()
    w.set_tour(tour)
    assert w._keyframes_buf is not None
    assert w._keyframes_msg["n_dims"] == 4
    assert (
        len(w._keyframes_buf) == tour.n_keyframes * 4 * 2 * 4
    )  # n_keyframes * dims * 2 * sizeof(float32)


def test_widget_constructor_with_data_and_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    from dtour.data import from_numpy

    data = from_numpy(X)
    tour = little_tour(X)
    w = Widget(data=data, tour=tour)
    assert w._data_buf is not None
    assert w._keyframes_buf is not None


def _sequential_tour():
    rng = np.random.default_rng(42)
    frames = [rng.standard_normal((50, 2)).astype(np.float32) for _ in range(3)]
    return sequential_tour(
        frames,
        method=lambda embedding, _previous: embedding,
        description="Frames",
        keyframe_descriptions=["A", "B", "C"],
    )


def test_widget_set_tour_syncs_tour_family_and_tour_by():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    assert Widget().tour_family is None

    w = Widget(tour=little_tour(X))
    assert (w._tour_family, w.tour_by) == ("hyperdimensional", "dimensions")
    assert w.tour_family == "hyperdimensional"

    w.set_tour(_sequential_tour())
    assert (w._tour_family, w.tour_by) == ("sequential", "parameter")
    assert w.tour_family == "sequential"
    with pytest.raises(AttributeError):
        w.tour_family = "hyperdimensional"

    w.set_tour(little_tour(X))
    assert (w._tour_family, w.tour_by) == ("hyperdimensional", "dimensions")


def test_widget_set_tour_accepts_more_keyframes_than_previews():
    rng = np.random.default_rng(42)
    frames = [rng.standard_normal((50, 2)).astype(np.float32) for _ in range(33)]
    tour = sequential_tour(frames, method=lambda embedding, _previous: embedding)
    w = Widget()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w.set_tour(tour)
    assert len(w._keyframes_buf) == 33 * tour.n_dims * 2 * 4


def test_widget_ready_resends_full_keyframes_message():
    w = Widget(tour=_sequential_tour())
    sent = []
    w.send = lambda msg, buffers=None: sent.append(msg)

    w._handle_custom_msg({"type": "ready"}, [])

    keyframes = next(msg for msg in sent if msg["type"] == "keyframes")
    assert keyframes["tour_description"] == "Frames"
    assert keyframes["keyframe_descriptions"] == ["A", "B", "C"]
    # The frontend reads the family from synced state, which it has on first render
    assert w.get_state()["_tour_family"] == "sequential"


@pytest.mark.skipif(sys.platform == "win32", reason="needs a shallow POSIX path")
def test_import_from_shallow_install():
    root = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        shutil.copytree(Path(dtour.__file__).parent, root / "dtour")
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", "import dtour; print(dtour.__file__)"],
            cwd="/",
            env={**os.environ, "PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(str(root))
    finally:
        shutil.rmtree(root)


def test_check_bundle_warns_when_older_than_sources(tmp_path, monkeypatch):
    bundle = tmp_path / "widget.js"
    source = tmp_path / "packages" / "python" / "js" / "widget.tsx"
    workspace = tmp_path / "pnpm-workspace.yaml"
    source.parent.mkdir(parents=True)
    for path, mtime in [(source, 100), (workspace, 100), (bundle, 200)]:
        path.touch()
        os.utime(path, (mtime, mtime))
    monkeypatch.setattr(widget, "_REPO", tmp_path)
    monkeypatch.setattr(widget, "_BUNDLE", bundle)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        widget._check_bundle()

    for path in (source, workspace):
        os.utime(path, (300, 300))
        with pytest.warns(UserWarning, match="older than its sources"):
            widget._check_bundle()
        os.utime(path, (100, 100))

    bundle.unlink()
    with pytest.raises(FileNotFoundError, match="pnpm build:widget"):
        widget._check_bundle()


# ── Data + tour ─────────────────────────────────────────────────────────


def _sent_columns(w: Widget) -> list[str]:
    from dtour.data import _read_table

    return _read_table(w._data_buf).column_names


def _embedding_tour(n: int = 30) -> TourResult:
    emb = np.random.default_rng(0).standard_normal((n, 3)).astype(np.float32)
    keyframes = [np.eye(3, 2, k, dtype=np.float32) for k in (0, -1)]
    return TourResult(keyframes, n_dims=3, embedding=emb, embedding_names=["E1", "E2", "E3"])


def _labeled_data(n: int = 30):
    import polars as pl

    rng = np.random.default_rng(1)
    return pl.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n), "label": rng.choice(["a", "b"], n)}
    )


def test_widget_adds_embedding_columns_first():
    w = Widget(_labeled_data(), _embedding_tour())
    assert _sent_columns(w) == ["E1", "E2", "E3", "f1", "f2", "label"]
    assert w.tour_dimensions == ["E1", "E2", "E3"]


def test_widget_tour_without_data_sends_embedding():
    w = Widget(tour=_embedding_tour())
    assert _sent_columns(w) == ["E1", "E2", "E3"]


def test_widget_keeps_data_that_starts_with_embedding():
    import polars as pl

    tour = _embedding_tour()
    data = pl.DataFrame({f"e{i}": tour.embedding[:, i] for i in range(3)}).with_columns(
        _labeled_data()["label"]
    )
    with pytest.warns(FutureWarning, match="already starts with the tour's embedding"):
        w = Widget(data, tour)
    assert _sent_columns(w) == ["e0", "e1", "e2", "label"]
    assert w.tour_dimensions == ["e0", "e1", "e2"]


def test_widget_rejects_data_with_other_row_count():
    with pytest.raises(ValueError, match="rows"):
        Widget(_labeled_data(10), _embedding_tour(30))


def test_widget_drops_embedding_when_tour_changes():
    data = _labeled_data()
    w = Widget(data, _embedding_tour())
    w.set_tour(little_tour(data.select("f1", "f2")))
    assert _sent_columns(w) == ["f1", "f2", "label"]
    assert w.tour_dimensions == ["f1", "f2"]


def test_widget_little_tour_names_its_columns():
    data = _labeled_data()
    w = Widget(data.select("label", "f2", "f1"), little_tour(data.select("f1", "f2")))
    assert w.tour_dimensions == ["f1", "f2"]


def test_widget_unnamed_tour_clears_previous_mapping():
    data = _labeled_data()
    w = Widget(data, little_tour(data.select("f2", "f1")))
    assert w.tour_dimensions == ["f2", "f1"]
    w.set_tour(little_tour(data.select("f1", "f2").to_numpy()))
    assert w.tour_dimensions == []

    w = Widget(data, _embedding_tour())
    w.set_tour(little_tour(data.select("f1", "f2").to_numpy()))
    assert w.tour_dimensions == []
    assert _sent_columns(w) == ["f1", "f2", "label"]


def test_widget_rejects_tour_columns_missing_from_data():
    data = _labeled_data()
    with pytest.raises(ValueError, match="no numeric columns named"):
        Widget(data.select("f1", "label"), little_tour(data.select("f1", "f2")))


def test_widget_keeps_state_when_update_fails():
    data = _labeled_data()
    tour = little_tour(data.select("f1", "f2"))
    w = Widget(data, tour)
    data_buf = w._data_buf
    with pytest.raises(ValueError, match="rows"):
        w.set_tour(_embedding_tour(2))
    assert w._tour is tour
    assert w._data_buf is data_buf
    assert w.tour_dimensions == ["f1", "f2"]

    w = Widget(data, _embedding_tour())
    with pytest.raises(ValueError, match="rows"):
        w.set_data(_labeled_data(10))
    assert w.save_spec_to_parquet().num_rows == 30


def test_widget_snapshots_data():
    import pyarrow as pa

    X = np.random.default_rng(0).standard_normal((30, 3)).astype(np.float32)
    reader = pa.RecordBatchReader.from_batches(
        pa.schema([("a", pa.float32()), ("b", pa.float32()), ("c", pa.float32())]),
        [pa.record_batch([pa.array(X[:, i]) for i in range(3)], names=["a", "b", "c"])],
    )
    assert Widget(reader, little_tour(X)).save_spec_to_parquet().num_rows == 30

    w = Widget(X, little_tour(X))
    first = X[0, 0]
    X[:, 0] += 100
    assert w.save_spec_to_parquet().column("dim_0").to_numpy()[0] == first


def test_widget_saves_unnamed_tour():
    import json

    X = np.random.default_rng(0).standard_normal((30, 3)).astype(np.float32)
    table = Widget(X, little_tour(X)).save_spec_to_parquet()
    meta = json.loads(table.schema.metadata_str["dtour"])
    assert meta["tour"]["dimensions"] == ["dim_0", "dim_1", "dim_2"]


def test_widget_unnamed_embedding_and_numpy_data():
    tour = _embedding_tour()
    tour.embedding_names = None
    X = np.random.default_rng(2).standard_normal((30, 3)).astype(np.float32)
    w = Widget(X, tour)
    assert _sent_columns(w) == [f"embedding_{i}" for i in range(3)] + ["dim_0", "dim_1", "dim_2"]


def test_widget_reads_ipc_file_bytes():
    import io

    import pyarrow as pa
    import pyarrow.ipc

    table = _labeled_data().to_arrow()
    buf = io.BytesIO()
    with pa.ipc.new_file(buf, table.schema) as writer:
        writer.write_table(table)
    w = Widget(buf.getvalue(), _embedding_tour())
    assert _sent_columns(w) == ["E1", "E2", "E3", "f1", "f2", "label"]


def test_widget_rejects_invalid_embedding_names():
    tour = _embedding_tour()
    tour.embedding_names = ["E1", "E1", "E3"]
    with pytest.raises(ValueError, match="unique names"):
        Widget(_labeled_data(), tour)


def test_widget_validates_explicit_mapping():
    data = _labeled_data()
    unnamed = little_tour(data.select("f1", "f2").to_numpy())
    w = Widget(data, unnamed, tour_dimensions=["f1", "f2"])
    assert w.tour_dimensions == ["f1", "f2"]

    with pytest.raises(ValueError, match="no numeric columns named"):
        w.set_data(data.rename({"f1": "other"}))
    assert w.tour_dimensions == ["f1", "f2"]
    assert _sent_columns(w) == ["f1", "f2", "label"]

    for names in (["f1"], ["f1", "f1"]):
        with pytest.raises(ValueError, match="name each once"):
            Widget(data, unnamed, tour_dimensions=names)
    with pytest.raises(ValueError, match="no numeric columns named"):
        Widget(data, unnamed, tour_dimensions=["f1", "missing"])


def test_widget_keeps_state_when_keyframes_fail_to_serialize():
    data = _labeled_data()
    tour = little_tour(data.select("f1", "f2"))
    w = Widget(data, tour)
    keyframes_buf = w._keyframes_buf
    bad = TourResult([np.eye(2, 2, dtype=np.float32), np.eye(3, 2, dtype=np.float32)], n_dims=2)
    with pytest.raises(ValueError):
        w.set_tour(bad)
    assert w._tour is tour
    assert w._keyframes_buf is keyframes_buf
    assert w.tour_dimensions == ["f1", "f2"]
