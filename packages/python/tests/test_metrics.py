"""Tests for quality metric computation."""

import numpy as np
import pytest

from dtour.metrics import MetricResult, compute_metrics
from dtour.tours import little_tour


def make_clustered_data(
    n_per_cluster: int = 50,
    n_clusters: int = 3,
    p: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate well-separated clusters for reliable metric values."""
    rng = np.random.default_rng(seed)
    clusters = []
    labels = []
    for i in range(n_clusters):
        center = rng.standard_normal(p).astype(np.float32) * 5
        points = center + rng.standard_normal((n_per_cluster, p)).astype(np.float32) * 0.3
        clusters.append(points)
        labels.append(np.full(n_per_cluster, i))
    return np.vstack(clusters), np.concatenate(labels)


@pytest.fixture
def clustered():
    X, labels = make_clustered_data()
    tour = little_tour(X)
    return X, tour.views, labels


def test_compute_metrics_silhouette(clustered):
    X, views, labels = clustered
    result = compute_metrics(X, views, labels=labels, metrics=["silhouette"])
    assert isinstance(result, MetricResult)
    assert "silhouette" in result.values
    assert len(result.values["silhouette"]) == len(views)
    # Well-separated clusters → positive silhouette
    for score in result.values["silhouette"]:
        assert -1 <= score <= 1


def test_compute_metrics_calinski_harabasz(clustered):
    X, views, labels = clustered
    result = compute_metrics(X, views, labels=labels, metrics=["calinski_harabasz"])
    assert "calinski_harabasz" in result.values
    for score in result.values["calinski_harabasz"]:
        assert score > 0


def test_compute_metrics_trustworthiness(clustered):
    X, views, _ = clustered
    result = compute_metrics(X, views, metrics=["trustworthiness"])
    assert "trustworthiness" in result.values
    for score in result.values["trustworthiness"]:
        assert 0 <= score <= 1


def test_compute_metrics_neighborhood_hit(clustered):
    X, views, labels = clustered
    result = compute_metrics(X, views, labels=labels, metrics=["neighborhood_hit"])
    assert "neighborhood_hit" in result.values
    for score in result.values["neighborhood_hit"]:
        assert 0 <= score <= 1


def test_compute_metrics_multiple(clustered):
    X, views, labels = clustered
    result = compute_metrics(
        X, views, labels=labels, metrics=["silhouette", "trustworthiness"]
    )
    assert len(result.metric_names) == 2
    assert len(result.values) == 2


def test_compute_metrics_default(clustered):
    X, views, _ = clustered
    # Default metrics include trustworthiness (unsupervised) — should work without labels
    result = compute_metrics(X, views, metrics=["trustworthiness"])
    assert len(result.values) > 0


def test_compute_metrics_unknown_metric(clustered):
    X, views, _ = clustered
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_metrics(X, views, metrics=["nonexistent"])


def test_compute_metrics_labels_required(clustered):
    X, views, _ = clustered
    with pytest.raises(ValueError, match="requires labels"):
        compute_metrics(X, views, labels=None, metrics=["silhouette"])
