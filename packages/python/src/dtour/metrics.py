"""Quality metric computation for tour projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class MetricResult:
    """Per-view quality metric values.

    Attributes:
        values: Mapping from metric name to a list of per-view scores.
        metric_names: Ordered list of metric names computed.
    """

    values: dict[str, list[float]]
    metric_names: list[str] = field(default_factory=list)

    def to_arrow_ipc(self) -> bytes:
        """Serialize metric values as Arrow IPC stream.

        Each column is a metric, rows are per-view values.
        """
        import arro3.core as ac
        import arro3.io

        arrays = {
            name: ac.Array.from_numpy(np.array(vals, dtype=np.float32))
            for name, vals in self.values.items()
        }
        table = ac.Table.from_pydict(arrays)
        buf = BytesIO()
        arro3.io.write_ipc_stream(table, buf)
        return buf.getvalue()


# All supported metrics and their requirements
_REQUIRES_LABELS = {"silhouette", "calinski_harabasz", "neighborhood_hit"}
_UNSUPERVISED = {"trustworthiness"}
_ALL_METRICS = _REQUIRES_LABELS | _UNSUPERVISED

_DEFAULT_METRICS = ["silhouette", "trustworthiness"]


def compute_metrics(
    X: np.ndarray,
    bases: list[np.ndarray],
    labels: np.ndarray | None = None,
    metrics: list[str] | None = None,
    k: int = 7,
) -> MetricResult:
    """Project data through each basis and compute quality metrics.

    Args:
        X: Data matrix, shape ``(n_samples, n_features)``, float32.
        bases: List of basis matrices, each shape ``(p, 2)`` where
            ``p = n_features``.
        labels: Cluster / class labels for supervised metrics (silhouette,
            calinski_harabasz, neighborhood_hit). Shape ``(n_samples,)``.
        metrics: Which metrics to compute. Defaults to
            ``["silhouette", "trustworthiness"]``.
        k: Number of neighbors for neighborhood-based metrics.

    Returns:
        A :class:`MetricResult` with per-view scores for each metric.

    Raises:
        ValueError: If a requested metric requires labels but none are given.
    """
    X = np.asarray(X, dtype=np.float32)
    requested = metrics or _DEFAULT_METRICS

    for m in requested:
        if m not in _ALL_METRICS:
            raise ValueError(f"Unknown metric {m!r}. Supported: {sorted(_ALL_METRICS)}")
        if m in _REQUIRES_LABELS and labels is None:
            raise ValueError(f"Metric {m!r} requires labels but none were provided.")

    result_values: dict[str, list[float]] = {m: [] for m in requested}

    for basis in bases:
        # Project: (n, p) @ (p, 2) → (n, 2)
        proj = X @ basis

        for m in requested:
            score = _compute_single(m, X, proj, labels, k)
            result_values[m].append(float(score))

    return MetricResult(values=result_values, metric_names=list(requested))


def _compute_single(
    metric: str,
    X: np.ndarray,
    proj: np.ndarray,
    labels: np.ndarray | None,
    k: int,
) -> float:
    if metric == "silhouette":
        from sklearn.metrics import silhouette_score

        return silhouette_score(proj, labels)

    if metric == "calinski_harabasz":
        from sklearn.metrics import calinski_harabasz_score

        return calinski_harabasz_score(proj, labels)

    if metric == "trustworthiness":
        from sklearn.manifold import trustworthiness

        return trustworthiness(X, proj, n_neighbors=k)

    if metric == "neighborhood_hit":
        return _neighborhood_hit(proj, labels, k)

    raise ValueError(f"Unknown metric: {metric!r}")


def _neighborhood_hit(
    proj: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Fraction of k-nearest neighbors in 2D that share the same label."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(proj)
    indices = nn.kneighbors(proj, return_distance=False)
    # indices[:, 0] is the point itself; skip it
    neighbor_labels = labels[indices[:, 1:]]
    hits = (neighbor_labels == labels[:, None]).mean()
    return float(hits)
