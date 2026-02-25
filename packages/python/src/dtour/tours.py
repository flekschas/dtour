"""Tour computation helpers: little_tour, little_umap_tour."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class TourResult:
    """Output of a tour computation.

    Attributes:
        bases: List of basis matrices, each shape ``(p, 2)`` float32.
        n_views: Number of projection views (bases).
        n_dims: Number of retained dimensions (p).
        explained_variance_ratio: Fraction of variance explained by each PCA
            component (when applicable).
    """

    bases: list[np.ndarray]
    n_views: int
    n_dims: int
    explained_variance_ratio: list[float] = field(default_factory=list)

    @property
    def bases_raw(self) -> bytes:
        """Raw float32 bytes of all basis matrices (for widget transfer).

        Layout: ``n_views`` contiguous blocks of ``p * 2`` float32 values,
        each in column-major order ``[x0 .. xp-1, y0 .. yp-1]``.
        """
        return np.stack([b.flatten() for b in self.bases]).astype(np.float32).tobytes()


def little_tour(
    X: np.ndarray | pd.DataFrame,
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
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include="number").to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    n_features = X.shape[1]
    k = min(n_components or 10, n_features)

    pca = PCA(n_components=k)
    pca.fit(X)

    # Components: shape (k, n_features). Each row is a principal component direction.
    components = pca.components_  # (k, p)

    # Build cyclic pairs: (0,1), (1,2), ..., (k-2, k-1), (k-1, 0)
    bases: list[np.ndarray] = []
    for i in range(k):
        a = components[i]               # (p,)
        b = components[(i + 1) % k]     # (p,)
        # Column-major px2: [a0, a1, ..., ap-1, b0, b1, ..., bp-1]
        basis = np.stack([a, b], axis=1).astype(np.float32)  # (p, 2)
        bases.append(basis)

    return TourResult(
        bases=bases,
        n_views=k,
        n_dims=n_features,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )


def little_umap_tour(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 10,
    umap_kwargs: dict | None = None,
) -> TourResult:
    """Reduce to n_components with UMAP, then compute a little tour.

    Requires ``umap-learn`` (install with ``pip install dtour[umap]``).

    Args:
        X: Input data.
        n_components: UMAP output dimensions (= tour dimensionality).
        umap_kwargs: Extra keyword arguments passed to ``umap.UMAP``.
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            "umap-learn is required for little_umap_tour. "
            "Install it with: pip install dtour[umap]"
        ) from e

    import pandas as pd

    if isinstance(X, pd.DataFrame):
        X = X.select_dtypes(include="number").to_numpy(dtype=np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    kwargs = {"n_components": n_components, "random_state": 42, **(umap_kwargs or {})}
    embedding = umap.UMAP(**kwargs).fit_transform(X)

    return little_tour(embedding, n_components=n_components)
