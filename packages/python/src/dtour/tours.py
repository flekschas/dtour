"""Tour computation helpers: little_tour, little_umap_tour."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import pyarrow as pa


@dataclass
class TourResult:
    """Output of a tour computation.

    Attributes:
        views: List of projection (basis) matrices, each shape ``(p, 2)`` float32.
        n_views: Number of projection views.
        n_dims: Number of retained dimensions (p).
        explained_variance_ratio: Fraction of variance explained by each PCA
            component (when applicable).
    """

    views: list[np.ndarray]
    n_views: int
    n_dims: int
    explained_variance_ratio: list[float] = field(default_factory=list)
    embedding: np.ndarray | None = None

    @property
    def views_raw(self) -> bytes:
        """Raw float32 bytes of all view matrices (for widget transfer).

        Layout: ``n_views`` contiguous blocks of ``p * 2`` float32 values,
        each in column-major order ``[x0 .. xp-1, y0 .. yp-1]``.
        """
        return np.stack([b.flatten() for b in self.views]).astype(np.float32).tobytes()


def _to_float32(X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table) -> np.ndarray:
    """Convert tabular input to a float32 numpy array, selecting numeric columns."""
    if isinstance(X, np.ndarray):
        return np.asarray(X, dtype=np.float32)

    type_name = type(X).__qualname__
    module = type(X).__module__

    # pandas DataFrame
    if type_name == "DataFrame" and module.startswith("pandas"):
        return X.select_dtypes(include="number").to_numpy(dtype=np.float32)

    # polars DataFrame
    if type_name == "DataFrame" and module.startswith("polars"):
        import polars.selectors as cs

        return X.select(cs.numeric()).to_numpy().astype(np.float32)

    # pyarrow Table
    if type_name == "Table" and module.startswith("pyarrow"):
        import pyarrow as pa

        numeric_names = [
            f.name
            for f in X.schema
            if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
        ]
        arrays = [X.column(c).to_numpy(zero_copy_only=False) for c in numeric_names]
        if not arrays:
            return np.empty((X.num_rows, 0), dtype=np.float32)
        return np.column_stack(arrays).astype(np.float32)

    # fallback
    return np.asarray(X, dtype=np.float32)


def little_tour(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table,
    n_components: int | None = None,
) -> TourResult:
    """Compute a PCA-based little tour over a dataset.

    Creates a cyclic sequence of 2D projections using consecutive pairs of
    principal components: [PC1, PC2] -> [PC2, PC3] -> ... -> [PCN, PC1].

    Args:
        X: Data matrix, shape (n_samples, n_features). DataFrame columns must
            be numeric.
        n_components: Number of PCA components to retain. Defaults to
            min(n_features, 10) to keep the tour manageable.

    Returns:
        A :class:`TourResult` with basis matrices as numpy arrays.
    """
    arr = _to_float32(X)

    n_features = arr.shape[1]
    k = min(n_components or 10, n_features)

    pca = PCA(n_components=k)
    pca.fit(arr)

    # Components: shape (k, n_features). Each row is a principal component direction.
    components = pca.components_  # (k, p)

    # Build cyclic pairs: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0)
    views: list[np.ndarray] = []
    for i in range(k):
        a = components[i]  # (p,)
        b = components[(i + 1) % k]  # (p,)
        # Column-major px2: [a0, a1, ..., ap-1, b0, b1, ..., bp-1]
        basis = np.stack([a, b], axis=1).astype(np.float32)  # (p, 2)
        views.append(basis)

    return TourResult(
        views=views,
        n_views=k,
        n_dims=n_features,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )


def little_umap_tour(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table,
    n_components: int = 10,
    umap_kwargs: dict | None = None,
) -> TourResult:
    """Reduce to n_components with UMAP, then compute a little tour.

    Args:
        X: Input data.
        n_components: UMAP output dimensions (= tour dimensionality).
        umap_kwargs: Extra keyword arguments passed to ``umap.UMAP``.
    """
    import umap

    arr = _to_float32(X)

    kwargs = {"n_components": n_components, **(umap_kwargs or {})}
    embedding = umap.UMAP(**kwargs).fit_transform(arr)

    result = little_tour(embedding, n_components=n_components)
    result.embedding = embedding
    return result
