"""Tour computation helpers: little_tour, little_umap_tour."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
from sklearn.decomposition import PCA

from .data import _table_to_ipc_bytes

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class TourResult:
    """Output of a tour computation.

    Attributes:
        bases: Arrow IPC bytes of a table with columns ``basis_<i>`` where
            each column is a flattened p×2 float32 array (one row per view).
        n_views: Number of projection views (bases).
        n_dims: Number of retained dimensions (p).
        explained_variance_ratio: Fraction of variance explained by each PCA
            component (when applicable).
    """

    bases: bytes
    n_views: int
    n_dims: int
    explained_variance_ratio: list[float]


def little_tour(
    X: "np.ndarray | pd.DataFrame",
    n_components: int | None = None,
) -> TourResult:
    """Compute a PCA-based little tour over a dataset.

    Creates a cyclic sequence of 2D projections using consecutive pairs of
    principal components: [PC1, PC2] → [PC2, PC3] → ... → [PCN, PC1].

    Args:
        X: Data matrix, shape (n_samples, n_features). DataFrame columns must
            be numeric.
        n_components: Number of PCA components to retain. Defaults to
            min(n_features, 10) to keep the tour manageable.

    Returns:
        A :class:`TourResult` with basis matrices encoded as Arrow IPC bytes.
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
    n_views = k
    bases: list[np.ndarray] = []
    for i in range(n_views):
        a = components[i]           # (p,)
        b = components[(i + 1) % k] # (p,)
        # Column-major p×2: [a0, a1, ..., ap, b0, b1, ..., bp]
        basis = np.stack([a, b], axis=1).astype(np.float32)  # (p, 2)
        bases.append(basis.flatten())

    bases_arr = np.stack(bases)  # (n_views, p*2)

    # Encode as Arrow table: one row per view, one column per flattened element
    col_names = [f"b{i}" for i in range(bases_arr.shape[1])]
    arrays = [pa.array(bases_arr[:, j]) for j in range(bases_arr.shape[1])]
    table = pa.table(dict(zip(col_names, arrays)))
    bases_ipc = _table_to_ipc_bytes(table)

    return TourResult(
        bases=bases_ipc,
        n_views=n_views,
        n_dims=n_features,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )


def little_umap_tour(
    X: "np.ndarray | pd.DataFrame",
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
