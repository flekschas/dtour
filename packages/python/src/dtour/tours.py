"""Tour computation helpers: little_tour, umap_little_tour, le_tour."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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
        feature_loadings: OLS regression coefficients mapping each embedding
            dimension back onto original features, shape ``(n_components, n_features)``.
        feature_names: Original feature column names for labeling loadings.
        feature_r2: Per-dimension R-squared from the OLS regression.
    """

    views: list[np.ndarray]
    n_views: int
    n_dims: int
    explained_variance_ratio: list[float] = field(default_factory=list)
    embedding: np.ndarray | None = None
    feature_loadings: np.ndarray | None = None
    feature_names: list[str] | None = None
    feature_r2: list[float] | None = None

    @property
    def views_raw(self) -> bytes:
        """Raw float32 bytes of all view matrices (for widget transfer).

        Layout: ``n_views`` contiguous blocks of ``p * 2`` float32 values,
        each in column-major order ``[x0 .. xp-1, y0 .. yp-1]``.
        """
        return np.stack([b.flatten("F") for b in self.views]).astype(np.float32).tobytes()

    def save(self, path: str | Path) -> None:
        """Save the tour to a ``.npz`` file for later reuse.

        The embedding (if present) and all basis matrices are stored so
        the tour can be restored without recomputation.
        """
        path = Path(path)
        arrays: dict[str, np.ndarray] = {
            "n_views": np.array([self.n_views]),
            "n_dims": np.array([self.n_dims]),
            "explained_variance_ratio": np.asarray(self.explained_variance_ratio, dtype=np.float64),
        }
        for i, v in enumerate(self.views):
            arrays[f"view_{i}"] = v
        if self.embedding is not None:
            arrays["embedding"] = self.embedding
        if self.feature_loadings is not None:
            arrays["feature_loadings"] = self.feature_loadings
        if self.feature_names is not None:
            arrays["feature_names_json"] = np.array([json.dumps(self.feature_names)])
        if self.feature_r2 is not None:
            arrays["feature_r2"] = np.asarray(self.feature_r2, dtype=np.float64)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> TourResult:
        """Load a tour previously saved with :meth:`save`."""
        data = np.load(path)
        n_views = int(data["n_views"][0])
        n_dims = int(data["n_dims"][0])
        views = [data[f"view_{i}"].astype(np.float32) for i in range(n_views)]
        evr_key = "explained_variance_ratio"
        evr = data[evr_key].tolist() if evr_key in data else []
        embedding = data["embedding"] if "embedding" in data else None
        feature_loadings = data["feature_loadings"] if "feature_loadings" in data else None
        feature_names = (
            json.loads(str(data["feature_names_json"][0])) if "feature_names_json" in data else None
        )
        feature_r2 = data["feature_r2"].tolist() if "feature_r2" in data else None
        return cls(
            views=views,
            n_views=n_views,
            n_dims=n_dims,
            explained_variance_ratio=evr,
            embedding=embedding,
            feature_loadings=feature_loadings,
            feature_names=feature_names,
            feature_r2=feature_r2,
        )


def _extract_feature_names(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table,
) -> list[str] | None:
    """Extract column names from tabular inputs, or ``None`` for plain arrays."""
    type_name = type(X).__qualname__
    module = type(X).__module__ or ""

    if type_name == "DataFrame" and module.startswith("pandas"):
        return X.select_dtypes(include="number").columns.tolist()

    if type_name == "DataFrame" and module.startswith("polars"):
        import polars.selectors as cs

        return X.select(cs.numeric()).columns

    if type_name == "Table" and module.startswith("pyarrow"):
        import pyarrow as _pa

        return [
            f.name
            for f in X.schema
            if _pa.types.is_integer(f.type) or _pa.types.is_floating(f.type)
        ]

    return None


def _compute_feature_loadings(
    X: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    """Regress each embedding dimension onto original features via OLS.

    Args:
        X: Original feature matrix, shape ``(n_samples, n_features)``.
        embedding: Low-dimensional embedding, shape ``(n_samples, n_components)``.

    Returns:
        loadings: Coefficients, shape ``(n_components, n_features)``.
        r2: Per-component R-squared values.
    """
    n_components = embedding.shape[1]
    loadings = np.empty((n_components, X.shape[1]), dtype=np.float32)
    r2: list[float] = []
    for k in range(n_components):
        y = embedding[:, k]
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        loadings[k] = w
        ss_res = np.sum((y - X @ w) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0)
    return loadings, r2


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
            f.name for f in X.schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
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

    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(
            f"little_tour requires a 2-D array with at least 2 columns, got shape {arr.shape}"
        )
    if n_components is not None and n_components < 2:
        raise ValueError(f"n_components must be >= 2, got {n_components}")

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


def umap_little_tour(
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


def _nystroem_extend(
    arr_train: np.ndarray,
    embedding_train: np.ndarray,
    arr_oos: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Project out-of-sample points into an existing spectral embedding.

    Uses kNN-weighted interpolation: for each OOS point, find its nearest
    neighbors in the training set and compute a distance-weighted average
    of their embedding coordinates.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(arr_train)
    distances, indices = nn.kneighbors(arr_oos)

    # Heat-kernel weights: exp(-d^2 / 2*sigma^2), sigma = median distance
    sigma = np.median(distances) + 1e-10
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average of neighbor embeddings
    neighbor_embeddings = embedding_train[indices]  # (m, k, n_components)
    return np.einsum("ij,ijk->ik", weights, neighbor_embeddings).astype(np.float32)


def _circular_basis(
    n_components: int,
    active: range | list[int],
) -> np.ndarray:
    """Build a (n_components, 2) basis with equal-weight circular layout.

    Active eigenvectors are placed at uniform angles ``2π·j/k`` around
    the full circle where *k* = len(active).  Columns are exactly
    orthonormal and all active rows have identical norm ``√(2/k)``.
    """
    k = len(active)
    basis = np.zeros((n_components, 2), dtype=np.float64)
    scale = np.sqrt(2.0 / k)
    for j, idx in enumerate(active):
        angle = 2.0 * np.pi * j / k
        basis[idx, 0] = scale * np.cos(angle)
        basis[idx, 1] = scale * np.sin(angle)
    return basis.astype(np.float32)


def _cumulative_views(
    n_dims: int,
    embedding: np.ndarray,
    n_remove: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build cumulative views via uniform circular projection.

    **Build-up phase** (n_dims - 1 views): start with the identity pair
    (LE0 vs LE1), then add one eigenvector at a time using full-circle
    ``2π·i/k`` angles until all *n_dims* are active.

    **Removal phase** (n_remove views, optional): progressively zero out
    the lowest-frequency eigenvectors, isolating high-frequency local
    structure — a spectral high-pass filter.

    Returns ``(views, normalized_embedding)``.
    """
    # Normalize to unit variance so all eigenvectors contribute equally
    stds = embedding.std(axis=0)
    stds[stds == 0] = 1
    emb_norm = (embedding / stds).astype(np.float32)

    views: list[np.ndarray] = []

    # Build-up phase: k=2 (identity pair), k=3..n_dims (circular)
    for k in range(2, n_dims + 1):
        if k == 2:
            basis = np.zeros((n_dims, 2), dtype=np.float32)
            basis[0, 0] = 1.0
            basis[1, 1] = 1.0
            views.append(basis)
        else:
            views.append(_circular_basis(n_dims, active=range(k)))

    # Removal phase: strip low-frequency eigenvectors one at a time
    for j in range(1, n_remove + 1):
        remaining = range(j, n_dims)
        if len(remaining) < 2:
            break
        views.append(_circular_basis(n_dims, active=remaining))

    return views, emb_norm


def le_tour(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table,
    n_components: int = 8,
    n_neighbors: int = 15,
    feature_names: list[str] | None = None,
    random_state: int | None = None,
    subsample: int | None = None,
    cumulative: bool = False,
    n_frames: int | None = None,
    n_remove: int = 0,
    se_kwargs: dict | None = None,
) -> TourResult:
    """Compute a Laplacian Eigenmaps tour.

    Builds a kNN affinity graph from *X*, computes the graph Laplacian's
    smallest non-trivial eigenvectors (spectral embedding / Laplacian
    eigenmaps), then builds a tour through the eigenvector space.

    Each eigenvector is regressed (OLS) back onto the original features so
    that loadings and R-squared values are available on the returned
    :class:`TourResult`.

    Args:
        X: Input data, shape ``(n_samples, n_features)``.
        n_components: Number of Laplacian eigenvectors to compute.  In
            cumulative mode with *n_frames*, this is derived automatically
            (``n_frames + 1``) and should not be set.
        n_neighbors: Number of nearest neighbors for the kNN affinity graph.
        feature_names: Original feature column names.  If ``None`` and *X*
            is a DataFrame, names are extracted automatically.
        random_state: Random seed for reproducibility.
        subsample: If set, randomly subsample this many rows for the spectral
            embedding computation, then project remaining rows via kNN
            interpolation.  Useful for large datasets (>100k rows).
        cumulative: If ``True``, each view progressively incorporates one more
            eigenvector through a fixed projection (global→local accumulation).
            Eigenvectors are variance-normalized so each contributes equally.
            If ``False`` (default), views are cyclic pairs in eigenvalue order.
        n_frames: Number of build-up frames in cumulative mode.  Since the
            first frame shows LE0 vs LE1 (2 eigenvectors) and each subsequent
            frame adds one, ``n_frames + 1`` eigenvectors are computed.
            Only used when ``cumulative=True``; ignored otherwise.
        n_remove: Number of low-frequency eigenvectors to progressively
            remove after the build-up phase (only used when ``cumulative=True``).
            Creates a spectral high-pass effect, isolating local structure.
        se_kwargs: Extra keyword arguments passed to
            ``sklearn.manifold.SpectralEmbedding``.

    Returns:
        A :class:`TourResult` with basis matrices, embedding,
        ``feature_loadings``, ``feature_names``, and ``feature_r2``.
    """
    from sklearn.manifold import SpectralEmbedding

    if feature_names is None:
        feature_names = _extract_feature_names(X)

    # In cumulative mode, n_frames drives the eigenvector count
    if cumulative and n_frames is not None:
        n_components = n_frames + 1

    arr = _to_float32(X)
    n_samples = arr.shape[0]

    rng = np.random.default_rng(random_state)

    kwargs: dict = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "affinity": "nearest_neighbors",
        "eigen_solver": "amg",
        **({"random_state": random_state} if random_state is not None else {}),
        **(se_kwargs or {}),
    }

    if subsample is not None and n_samples > subsample:
        # Subsample for the expensive eigendecomposition
        idx_train = rng.choice(n_samples, size=subsample, replace=False)
        idx_train.sort()
        arr_train = arr[idx_train]

        embedding_train = SpectralEmbedding(**kwargs).fit_transform(arr_train)

        # Project remaining points via kNN interpolation
        mask_oos = np.ones(n_samples, dtype=bool)
        mask_oos[idx_train] = False
        arr_oos = arr[mask_oos]

        embedding_oos = _nystroem_extend(arr_train, embedding_train, arr_oos, n_neighbors)

        # Reassemble full embedding in original row order
        embedding = np.empty((n_samples, n_components), dtype=np.float32)
        embedding[idx_train] = embedding_train
        embedding[mask_oos] = embedding_oos
    else:
        embedding = SpectralEmbedding(**kwargs).fit_transform(arr)

    if cumulative:
        views, emb_for_tour = _cumulative_views(n_components, embedding, n_remove)
    else:
        # Cyclic pairs in native eigenvalue order (global→local)
        views = []
        for i in range(n_components):
            basis = np.zeros((n_components, 2), dtype=np.float32)
            basis[i, 0] = 1.0
            basis[(i + 1) % n_components, 1] = 1.0
            views.append(basis)
        emb_for_tour = embedding

    result = TourResult(
        views=views,
        n_views=len(views),
        n_dims=n_components,
    )
    result.embedding = emb_for_tour

    loadings, r2 = _compute_feature_loadings(arr, embedding)
    result.feature_loadings = loadings
    result.feature_names = feature_names
    result.feature_r2 = r2

    return result
