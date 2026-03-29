"""Tests for widget instantiation and trait validation."""

import numpy as np
import pytest
from dtour.tours import little_tour
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
    assert w.view_mode == "guided"
    assert w.show_legend is True
    assert w.theme == "dark"
    assert w.height == 720


def test_widget_preview_count_validation():
    w = Widget(preview_count=8)
    assert w.preview_count == 8

    with pytest.raises(Exception):
        Widget(preview_count=5)


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
    assert w._n_dims == 4
    assert len(w._views_buf) == tour.n_views * 4 * 2 * 4  # n_views * dims * 2 * sizeof(float32)


def test_widget_constructor_with_data_and_tour():
    X = np.random.default_rng(42).standard_normal((50, 4)).astype(np.float32)
    from dtour.data import from_numpy

    data = from_numpy(X)
    tour = little_tour(X)
    w = Widget(data=data, tour=tour)
    assert w._data_buf is not None
    assert w._views_buf is not None
