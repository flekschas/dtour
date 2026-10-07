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
from dtour.tours import little_tour, sequential_tour
from dtour.widget import Widget


def test_widget_default_traits():
    w = Widget()
    assert w.tour_by == "dimensions"
    assert w.tour_position == 0.0
    assert w.tour_playing is False
    assert w.tour_speed == 1.0
    assert w.tour_direction == "forward"
    assert w.preview_count == 4
    assert w.preview_padding == 12.0
    assert w.point_size == "auto"
    assert w.point_opacity == "auto"
    assert w.point_color == [0.25, 0.5, 0.9]
    assert w.camera_pan_x == 0.0
    assert w.camera_pan_y == 0.0
    assert w.camera_zoom == pytest.approx(1 / 1.5)
    assert w.tour_traversal == "guided"
    assert w.show_legend is True
    assert w.theme == "dark"
    assert w.height == 720


def test_widget_preview_count_validation():
    w = Widget(preview_count=8)
    assert w.preview_count == 8

    with pytest.raises(Exception):
        Widget(preview_count=1)


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
    assert w._views_buf is not None
    assert w._views_msg["n_dims"] == 4
    assert len(w._views_buf) == tour.n_views * 4 * 2 * 4  # n_views * dims * 2 * sizeof(float32)


def test_widget_constructor_with_data_and_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    from dtour.data import from_numpy

    data = from_numpy(X)
    tour = little_tour(X)
    w = Widget(data=data, tour=tour)
    assert w._data_buf is not None
    assert w._views_buf is not None


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


def test_widget_ready_resends_full_views_message():
    w = Widget(tour=_sequential_tour())
    sent = []
    w.send = lambda msg, buffers=None: sent.append(msg)

    w._handle_custom_msg({"type": "ready"}, [])

    views = next(msg for msg in sent if msg["type"] == "views")
    assert views["tour_description"] == "Frames"
    assert views["keyframe_descriptions"] == ["A", "B", "C"]
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
