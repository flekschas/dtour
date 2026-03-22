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
        arro3.io.write_ipc_stream(table, buf, compression=None)
        return buf.getvalue()


# All supported metrics and their requirements
_REQUIRES_LABELS = {"silhouette", "calinski_harabasz", "neighborhood_hit", "confusion"}
_UNSUPERVISED = {"trustworthiness", "hdbscan_score"}
_ALL_METRICS = _REQUIRES_LABELS | _UNSUPERVISED

_DEFAULT_METRICS = ["silhouette", "trustworthiness"]

# Per-metric default subsample sizes (None = use all points)
_SUBSAMPLE_DEFAULTS: dict[str, int | None] = {
    "silhouette": 10_000,
    "trustworthiness": 10_000,
    "calinski_harabasz": 10_000,
    "hdbscan_score": 50_000,
    "neighborhood_hit": None,
    "confusion": None,
}


def compute_metrics(
    X: np.ndarray,
    views: list[np.ndarray],
    labels: np.ndarray | None = None,
    metrics: list[str] | None = None,
    k: int = 7,
    subsample: int | dict[str, int | None] | None = None,
    exclude_labels: list[str] | None = None,
) -> MetricResult:
    """Project data through each basis and compute quality metrics.

    Args:
        X: Data matrix, shape ``(n_samples, n_features)``, float32.
        views: List of projection (basis) matrices, each shape ``(p, 2)``
            where ``p = n_features``.
        labels: Cluster / class labels for supervised metrics (silhouette,
            calinski_harabasz, neighborhood_hit, confusion).
            Shape ``(n_samples,)``.
        metrics: Which metrics to compute. Defaults to
            ``["silhouette", "trustworthiness"]``.
        k: Number of neighbors for neighborhood-based metrics.
        subsample: Controls per-metric subsampling to speed up expensive
            metrics on large datasets.

            - ``None`` (default): use built-in per-metric defaults (e.g.
              10K for silhouette/trustworthiness, 50K for hdbscan_score,
              no subsampling for neighborhood_hit/confusion).
            - ``int``: override all metrics to subsample to this many points.
            - ``dict``: per-metric overrides merged with the defaults, e.g.
              ``{"silhouette": 20_000, "hdbscan_score": None}``.
        exclude_labels: Label values to exclude from all label-based metrics.
            Points with these labels are removed before computing any metric
            that uses labels. Unsupervised metrics are unaffected.

    Returns:
        A :class:`MetricResult` with per-view scores for each metric.

    Raises:
        ValueError: If a requested metric requires labels but none are given.
    """
    X = np.asarray(X, dtype=np.float32)

    # Filter out excluded labels (affects both X and labels)
    if exclude_labels and labels is not None:
        exclude_set = set(exclude_labels)
        mask = np.array([l not in exclude_set for l in labels])
        X = X[mask]
        labels = labels[mask]

    # Normalize each dimension to [-0.5, 0.5] to match the GPU shader which
    # applies per-dim (raw - min) / range - 0.5 before projecting.
    mins = X.min(axis=0)
    ranges = X.max(axis=0) - mins
    ranges[ranges == 0] = 1e-6
    X_norm = (X - mins) / ranges - 0.5

    n = X.shape[0]
    requested = metrics or _DEFAULT_METRICS

    for m in requested:
        if m not in _ALL_METRICS:
            raise ValueError(f"Unknown metric {m!r}. Supported: {sorted(_ALL_METRICS)}")
        if m in _REQUIRES_LABELS and labels is None:
            raise ValueError(f"Metric {m!r} requires labels but none were provided.")

    # Resolve per-metric subsample sizes
    if isinstance(subsample, int):
        resolved_subsample = {m: subsample for m in requested}
    elif isinstance(subsample, dict):
        resolved_subsample = {m: subsample.get(m, _SUBSAMPLE_DEFAULTS.get(m)) for m in requested}
    else:
        resolved_subsample = {m: _SUBSAMPLE_DEFAULTS.get(m) for m in requested}

    result_values: dict[str, list[float]] = {m: [] for m in requested}
    rng = np.random.default_rng(seed=0)

    for basis in views:
        # Project normalized data: (n, p) @ (p, 2) → (n, 2)
        proj = X_norm @ basis

        # Pre-compute subsampled index arrays (one per unique subsample size)
        idx_cache: dict[int, np.ndarray] = {}

        for m in requested:
            sub_n = resolved_subsample[m]
            if sub_n is not None and sub_n < n:
                if sub_n not in idx_cache:
                    idx_cache[sub_n] = rng.choice(n, size=sub_n, replace=False)
                idx = idx_cache[sub_n]
                sub_proj = proj[idx]
                sub_X = X[idx]
                sub_labels = labels[idx] if labels is not None else None
            else:
                sub_proj, sub_X, sub_labels = proj, X, labels

            score = _compute_single(m, sub_X, sub_proj, sub_labels, k)
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

    if metric == "hdbscan_score":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=50).fit(proj)
        # Number of clusters found (excluding noise label -1)
        return float(clusterer.labels_.max() + 1)

    if metric == "confusion":
        return _confusion(proj, labels)

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


def _confusion(proj: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-label confusion (0 = perfect separation, 1 = fully mixed).

    Uses cev-metrics Rust KNN to build a label confusion matrix, normalizes
    each row by its row sum, then averages ``1 - diag`` across labels.
    Labels with 2 or fewer points are excluded (assigned 0 confusion).
    """
    import cev_metrics
    import pandas as pd

    cat = pd.Categorical(labels)
    df = pd.DataFrame({
        "x": proj[:, 0],
        "y": proj[:, 1],
        "label": cat,
    })
    cm = np.asarray(cev_metrics.confusion(df), dtype=np.float64)
    row_sums = cm.sum(axis=1)
    # Avoid division by zero for empty rows
    row_sums[row_sums == 0] = 1.0
    normed = cm / row_sums[:, None]
    per_label = 1.0 - np.diag(normed)
    # Zero out labels with 2 or fewer points
    counts = pd.Series(cat).value_counts()
    for i, category in enumerate(cat.categories):
        if counts.get(category, 0) <= 2:
            per_label[i] = 0.0
    return float(np.mean(per_label))
