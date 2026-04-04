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
        feature_loadings: Pearson correlations between each embedding
            dimension and each original feature, shape ``(n_components, n_features)``.
        feature_names: Original feature column names for labeling loadings.
        feature_r2: Per-dimension R-squared from the OLS regression.
        frame_summaries: Per-frame text describing the top projection-driving
            features, e.g. ``"Structure: CD3, CD4"``.
        tour_mode: The embedding mode: ``None`` (vanilla LE),
            ``"signed"`` (true signed Laplacian), or
            ``"discriminative"`` (spectral Fisher discriminant).
        tour_description: Human-readable description of the tour
            (shown in the description sub-bar).
        tour_frame_description: Template string for per-frame tooltips.
            Supports ``{dim1}``, ``{dim2}``, and ``{relation}`` placeholders.
    """

    views: list[np.ndarray]
    n_views: int
    n_dims: int
    explained_variance_ratio: list[float] = field(default_factory=list)
    embedding: np.ndarray | None = None
    feature_loadings: np.ndarray | None = None
    feature_names: list[str] | None = None
    feature_r2: list[float] | None = None
    frame_summaries: list[str] | None = None
    tour_mode: str | None = None
    tour_description: str | None = None
    tour_frame_description: str | None = None

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
        if self.frame_summaries is not None:
            arrays["frame_summaries_json"] = np.array([json.dumps(self.frame_summaries)])
        if self.tour_mode is not None:
            arrays["tour_mode"] = np.array([self.tour_mode])
        if self.tour_description is not None:
            arrays["tour_description"] = np.array([self.tour_description])
        if self.tour_frame_description is not None:
            arrays["tour_frame_description"] = np.array([self.tour_frame_description])
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
        frame_summaries = (
            json.loads(str(data["frame_summaries_json"][0]))
            if "frame_summaries_json" in data
            else None
        )
        # Read tour_mode; fall back to old n_attract for backward compat
        if "tour_mode" in data:
            tour_mode = str(data["tour_mode"][0])
        elif "n_attract" in data:
            tour_mode = "signed"
        else:
            tour_mode = None
        tour_description = str(data["tour_description"][0]) if "tour_description" in data else None
        tour_frame_description = (
            str(data["tour_frame_description"][0]) if "tour_frame_description" in data else None
        )
        return cls(
            views=views,
            n_views=n_views,
            n_dims=n_dims,
            explained_variance_ratio=evr,
            embedding=embedding,
            feature_loadings=feature_loadings,
            feature_names=feature_names,
            feature_r2=feature_r2,
            frame_summaries=frame_summaries,
            tour_mode=tour_mode,
            tour_description=tour_description,
            tour_frame_description=tour_frame_description,
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
    """Pearson correlation between each feature and each embedding dimension.

    Uses marginal Pearson *r* rather than OLS regression coefficients.
    Correlations are bounded [-1, 1], scale-invariant, and unaffected by
    multicollinearity — making them more suitable for ranking feature
    importance.

    The per-dimension R² is still computed via OLS to indicate how well
    *all* features together linearly predict each embedding dimension.

    Args:
        X: Original feature matrix, shape ``(n_samples, n_features)``.
        embedding: Low-dimensional embedding, shape ``(n_samples, n_components)``.

    Returns:
        correlations: Pearson *r*, shape ``(n_components, n_features)``.
        r2: Per-component multivariate R-squared values (OLS).
    """
    n_components = embedding.shape[1]

    # Pearson correlation: r(emb_k, X_j) for each (k, j) pair
    X_c = X - X.mean(axis=0, keepdims=True)
    E_c = embedding - embedding.mean(axis=0, keepdims=True)

    X_norm = np.sqrt((X_c**2).sum(axis=0, keepdims=True))  # (1, n_features)
    E_norm = np.sqrt((E_c**2).sum(axis=0, keepdims=True))  # (1, n_components)

    X_norm = np.where(X_norm == 0, 1, X_norm)
    E_norm = np.where(E_norm == 0, 1, E_norm)

    # (n_components, n_features)
    correlations = (E_c / E_norm).T @ (X_c / X_norm)

    # Per-dimension multivariate R² via OLS
    r2: list[float] = []
    for k in range(n_components):
        y = embedding[:, k]
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ss_res = np.sum((y - X @ w) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0)

    return correlations.astype(np.float32), r2


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

    # Heat-kernel weights: exp(-d^2 / sigma^2), sigma = median distance
    # (matches the kernel used by _build_knn_affinity)
    sigma = np.median(distances) + 1e-10
    weights = np.exp(-(distances**2) / (sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)

    # Weighted average of neighbor embeddings
    neighbor_embeddings = embedding_train[indices]  # (m, k, n_components)
    return np.einsum("ij,ijk->ik", weights, neighbor_embeddings).astype(np.float32)


def _stratified_subsample(
    labels: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return stratified subsample indices that preserve all classes.

    Each class gets at least ``max(1, n_neighbors_min)`` representatives,
    with remaining budget distributed proportionally to class frequency.

    Raises ``ValueError`` if the requested subsample size is too small
    to include at least one sample from every class.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)

    if n_total < n_classes:
        msg = (
            f"subsample={n_total} is too small to represent all "
            f"{n_classes} classes. Need at least {n_classes}."
        )
        raise ValueError(msg)

    # Guarantee at least 1 sample per class, then distribute remainder
    per_class = np.ones(n_classes, dtype=int)
    remainder = n_total - n_classes
    if remainder > 0:
        # Proportional allocation of the remainder
        fracs = counts / counts.sum()
        extra = np.floor(fracs * remainder).astype(int)
        # Distribute any leftover from rounding to the largest classes
        shortfall = remainder - extra.sum()
        if shortfall > 0:
            top = np.argsort(-fracs)[:shortfall]
            extra[top] += 1
        per_class += extra

    # Cap at actual class size
    per_class = np.minimum(per_class, counts)

    indices = []
    for lb, n_pick in zip(unique, per_class):
        class_idx = np.where(labels == lb)[0]
        chosen = rng.choice(class_idx, size=n_pick, replace=False)
        indices.append(chosen)

    idx = np.concatenate(indices)
    idx.sort()
    return idx


def _build_knn_affinity(
    arr: np.ndarray,
    n_neighbors: int,
):
    """Build a symmetric kNN affinity matrix with heat-kernel weights.

    Uses ``sklearn.neighbors.kneighbors_graph`` to find k-nearest neighbors,
    symmetrises the graph, and applies a heat kernel with bandwidth set to
    the median non-zero distance.

    Returns a sparse CSR affinity matrix.
    """
    from sklearn.neighbors import kneighbors_graph

    G = kneighbors_graph(arr, n_neighbors=n_neighbors, mode="distance", include_self=False)
    # Symmetrise: take the union of kNN relationships
    G = 0.5 * (G + G.T)
    # Heat kernel: W_ij = exp(-d^2 / sigma^2), sigma = median nonzero distance
    sigma = np.median(G.data) + 1e-10
    G.data = np.exp(-(G.data**2) / (sigma**2))
    return G


def _split_affinity_by_labels(
    W,
    labels: np.ndarray,
):
    """Split an affinity matrix into same-label and cross-label edges.

    Returns:
        W_same: Sparse CSR matrix with only same-label edges.
        W_cross: Sparse CSR matrix with only cross-label edges.
    """
    from scipy import sparse

    rows, cols = W.nonzero()
    same_mask = labels[rows] == labels[cols]
    cross_mask = ~same_mask

    W_same = W.copy().tolil()
    W_same[rows[cross_mask], cols[cross_mask]] = 0
    W_same = sparse.csr_matrix(W_same)
    W_same.eliminate_zeros()

    W_cross = W.copy().tolil()
    W_cross[rows[same_mask], cols[same_mask]] = 0
    W_cross = sparse.csr_matrix(W_cross)
    W_cross.eliminate_zeros()

    return W_same, W_cross


def _lobpcg_eigenvectors(
    A,
    n_components: int,
    B=None,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute smallest eigenvectors using LOBPCG with AMG preconditioner.

    Uses pyamg's ``smoothed_aggregation_solver`` as preconditioner for fast
    convergence on large sparse systems, matching the solver used by
    ``sklearn.manifold.SpectralEmbedding`` with ``eigen_solver="amg"``.

    Args:
        A: Sparse symmetric matrix (the operator).
        B: Optional sparse SPD matrix for the generalised problem A x = λ B x.
        n_components: Number of eigenvectors to compute.
        random_state: Seed for the initial random guess.

    Returns:
        Eigenvectors corresponding to the smallest eigenvalues,
        shape ``(n, n_components)``, as float32.
    """
    from pyamg import smoothed_aggregation_solver
    from scipy.sparse.linalg import lobpcg

    rng = np.random.default_rng(random_state)
    n = A.shape[0]
    X0 = rng.standard_normal((n, n_components)).astype(np.float64)

    # AMG preconditioner on A (or A itself for the standard problem)
    ml = smoothed_aggregation_solver(A.tocsr())
    M = ml.aspreconditioner()

    eigenvalues, eigenvectors = lobpcg(A, X0, B=B, M=M, largest=False, tol=1e-8, maxiter=500)

    # Sort by eigenvalue (ascending)
    order = np.argsort(eigenvalues)
    return eigenvectors[:, order].astype(np.float32)


def _signed_laplacian_embed(
    arr: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    n_neighbors: int,
    alpha: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute eigenvectors of the true signed graph Laplacian.

    Builds ``W_signed = W_same - alpha * W_cross`` from the kNN affinity
    graph, then solves for the smallest eigenvectors of the **normalised**
    signed Laplacian ``L_sym = D^{-1/2} L D^{-1/2}``.

    The degree-normalisation (matching what ``sklearn.SpectralEmbedding``
    uses for standard LE) bounds the eigenvector entries and prevents
    extreme outliers from nodes with unusual degree.

    Unlike standard LE, the constant vector is NOT a trivial eigenvector
    of the signed Laplacian, so all returned eigenvectors are informative.

    Args:
        arr: Feature matrix, shape ``(n_samples, n_features)``.
        labels: Per-sample class labels.
        n_components: Number of eigenvectors to compute.
        n_neighbors: kNN graph parameter.
        alpha: Repulsion strength.  ``0`` recovers same-label-only LE,
            ``1`` gives balanced attraction/repulsion.
        random_state: Seed for the eigensolver.

    Returns:
        Embedding matrix, shape ``(n_samples, n_components)``, float32.
    """
    from scipy import sparse

    W = _build_knn_affinity(arr, n_neighbors)
    W_same, W_cross = _split_affinity_by_labels(W, labels)

    # Signed affinity: positive for same-label, negative for cross-label
    W_signed = W_same - alpha * W_cross

    # Degree matrix uses absolute values of signed affinity
    W_abs = W_same + alpha * W_cross  # |W_signed|
    d = np.asarray(W_abs.sum(axis=1)).ravel()

    # Symmetric normalisation: L_sym = D^{-1/2} L D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-10))
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.diags(d) - W_signed
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) / 2  # numerical symmetry

    evecs_norm = _lobpcg_eigenvectors(
        L_sym,
        n_components,
        random_state=random_state,
    )

    # Transform back to original space: v = D^{-1/2} * u
    return (D_inv_sqrt @ evecs_norm).astype(np.float32)


def _spectral_fisher_embed(
    arr: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    n_neighbors: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Compute Fisher-discriminant embedding by LDA on LE eigenvectors.

    First computes a standard Laplacian Eigenmaps embedding (which is
    well-scaled thanks to sklearn's degree normalisation), then solves
    a small Fisher LDA problem in the embedding space::

        S_b · v = λ · S_w · v

    where ``S_b`` and ``S_w`` are the between-class and within-class
    scatter matrices of the spectral embedding.  The largest eigenvalues
    correspond to the most discriminative projections.

    This avoids the numerical issues of graph-Laplacian generalised
    eigenvalue problems (unbounded eigenvector entries for poorly-connected
    minority-class nodes) because the LE embedding is already bounded
    and the LDA reduces to a tiny dense eigenvalue problem.

    Args:
        arr: Feature matrix, shape ``(n_samples, n_features)``.
        labels: Per-sample class labels.
        n_components: Number of eigenvectors to compute.
        n_neighbors: kNN graph parameter.
        random_state: Seed for the eigensolver.

    Returns:
        Embedding matrix, shape ``(n_samples, n_components)``, float32.
    """
    from scipy.linalg import eigh
    from sklearn.manifold import SpectralEmbedding

    # Step 1: Standard LE embedding (well-normalised by sklearn)
    se = SpectralEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        affinity="nearest_neighbors",
        eigen_solver="amg",
        **({"random_state": random_state} if random_state is not None else {}),
    )
    emb = se.fit_transform(arr)  # (n, n_components)

    # Step 2: Fisher LDA in the embedding space
    unique_labels = np.unique(labels)
    d = n_components
    global_mean = emb.mean(axis=0)  # (d,)

    S_b = np.zeros((d, d), dtype=np.float64)
    S_w = np.zeros((d, d), dtype=np.float64)

    for lb in unique_labels:
        mask = labels == lb
        n_c = mask.sum()
        class_emb = emb[mask]  # (n_c, d)
        mean_c = class_emb.mean(axis=0)

        # Between-class scatter
        diff = (mean_c - global_mean).reshape(-1, 1)
        S_b += n_c * (diff @ diff.T)

        # Within-class scatter
        centered = class_emb - mean_c
        S_w += centered.T @ centered

    # Regularise S_w for numerical stability
    S_w += 1e-6 * np.eye(d)

    # Solve S_b v = lambda S_w v  (largest eigenvalues)
    _, eigenvectors = eigh(S_b, S_w)

    # eigh returns ascending order; reverse for descending (most discriminative first)
    eigenvectors = eigenvectors[:, ::-1]

    # Step 3: Project LE embedding through Fisher directions
    return (emb @ eigenvectors).astype(np.float32)


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


def _compute_frame_summaries(
    views: list[np.ndarray],
    loadings: np.ndarray,
    feature_names: list[str],
    tour_mode: str | None = None,
) -> list[str]:
    """Compute per-frame text summaries from per-eigenvector loadings.

    For each cumulative frame, reports the top-2 features (by absolute
    loading) of the **newly added eigenvector** — i.e. the eigenvector
    that distinguishes this frame from the previous one.  Frame 0 uses
    eigenvector 0, frame 1 uses eigenvector 1, etc.  This matches the
    per-row information shown in the loading heatmap.
    """
    if tour_mode == "discriminative":
        prefix = "Discriminant"
    else:
        prefix = "Structure"

    summaries: list[str] = []
    n_eigenvectors = loadings.shape[0]
    for i, _basis in enumerate(views):
        # In the build-up phase (frames 0..n_eigenvectors-2), each frame
        # adds eigenvector i+1.  Frame 0 is the identity pair (ev0, ev1),
        # so we report eigenvector 1 for it.  Clamp to valid range.
        ev_idx = min(i + 1, n_eigenvectors - 1)
        importance = np.abs(loadings[ev_idx])
        top_k = np.argsort(importance)[::-1][:2]
        top_names = [feature_names[j].rstrip("_") for j in top_k]
        summaries.append(f"{prefix}: {top_names[0]}, {top_names[1]}")
    return summaries


_TOUR_DESCRIPTIONS: dict[str | None, str] = {
    None: "Neighborhood structure from broad to fine. Each frame adds increasingly local variation.",  # noqa: E501
    "signed": "Label-aware structure from coarse to fine. Each frame adds finer label-aware patterns.",  # noqa: E501
    "discriminative": (
        "Label contrasts from strongest to subtlest."
        " Each frame adds a new between-label difference."
    ),
}

_TOUR_FRAME_DESCRIPTIONS: dict[str | None, str] = {
    None: "Top new correlates for finer neighborhoods: {relation} {dim1} and {dim2} (vs. previous frame)",  # noqa: E501
    "signed": "Top new correlates for finer label patterns: {relation} {dim1} and {dim2} (vs. previous frame)",  # noqa: E501
    "discriminative": "Top new correlates for subtler label contrasts: {relation} {dim1} and {dim2} (vs. previous frame)",  # noqa: E501
}


def le_tour(
    X: np.ndarray | pd.DataFrame | pl.DataFrame | pa.Table,
    n_components: int = 8,
    n_neighbors: int = 15,
    feature_names: list[str] | None = None,
    random_state: int | None = None,
    subsample: int | None = None,
    n_frames: int | None = None,
    n_remove: int = 0,
    se_kwargs: dict | None = None,
    labels: np.ndarray | None = None,
    discriminative: bool = False,
    alpha: float = 1.0,
) -> TourResult:
    """Compute a Laplacian Eigenmaps tour.

    Builds a kNN affinity graph from *X*, computes the graph Laplacian's
    smallest non-trivial eigenvectors (spectral embedding / Laplacian
    eigenmaps), then builds a cumulative tour through the eigenvector space.

    Each view progressively incorporates one more eigenvector through a
    fixed circular projection (global → local accumulation).  Eigenvectors
    are variance-normalised so each contributes equally.

    Each eigenvector is correlated (Pearson *r*) with the original features
    so that loadings and R-squared values are available on the returned
    :class:`TourResult`.

    When *labels* are provided, the kNN graph becomes class-aware via a
    true signed Laplacian: same-label edges attract and cross-label edges
    repel.  This produces a single ordered set of eigenvectors that
    capture class-aware structure from coarse to fine.

    When *discriminative* is ``True`` (requires *labels*), a spectral
    Fisher discriminant is used instead: eigenvectors are ordered by
    discriminative power so the tour builds up from the most to least
    class-separating directions.

    Args:
        X: Input data, shape ``(n_samples, n_features)``.
        n_components: Number of Laplacian eigenvectors to compute.  When
            *n_frames* is given, this is derived automatically
            (``n_frames + 1``) and should not be set.
        n_neighbors: Number of nearest neighbors for the kNN affinity graph.
        feature_names: Original feature column names.  If ``None`` and *X*
            is a DataFrame, names are extracted automatically.
        random_state: Random seed for reproducibility.
        subsample: If set, randomly subsample this many rows for the spectral
            embedding computation, then project remaining rows via kNN
            interpolation.  Useful for large datasets (>100k rows).
        n_frames: Total number of tour frames.  The first frame shows
            eigenvectors 0 and 1, each subsequent frame adds one more,
            so ``n_frames + 1`` eigenvectors are computed.
        n_remove: Number of low-frequency eigenvectors to progressively
            remove after the build-up phase.  Creates a spectral high-pass
            effect, isolating local structure.
        se_kwargs: Extra keyword arguments passed to
            ``sklearn.manifold.SpectralEmbedding`` (vanilla LE path only).
        labels: Per-sample class labels.  When provided (without
            *discriminative*), enables the **signed Laplacian** path
            where same-label edges attract and cross-label edges repel.
        discriminative: If ``True``, use spectral Fisher discriminant
            instead of the signed Laplacian.  Requires *labels*.
            Eigenvectors are ordered by discriminative power.
        alpha: Repulsion strength for the signed Laplacian path.
            ``0`` recovers same-label-only LE, ``1`` (default) gives
            balanced attraction/repulsion.  Ignored when *discriminative*
            is ``True``.

    Returns:
        A :class:`TourResult` with basis matrices, embedding,
        ``feature_loadings``, ``feature_names``, ``feature_r2``, and
        ``frame_summaries``.
    """
    from sklearn.manifold import SpectralEmbedding

    if feature_names is None:
        feature_names = _extract_feature_names(X)

    if discriminative and labels is None:
        msg = "discriminative=True requires labels to be provided."
        raise ValueError(msg)

    # Derive n_components from n_frames (or vice versa)
    if n_frames is not None:
        n_components = n_frames + 1

    arr = _to_float32(X)
    n_samples = arr.shape[0]

    if labels is not None and len(labels) != n_samples:
        msg = f"labels length ({len(labels)}) must match number of samples ({n_samples})."
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)

    # Determine tour mode
    if discriminative:
        tour_mode = "discriminative"
    elif labels is not None:
        tour_mode = "signed"
    else:
        tour_mode = None

    # ── Label-aware paths (signed Laplacian or spectral Fisher) ───────
    if labels is not None:
        labels_arr = np.asarray(labels)

        if subsample is not None and n_samples > subsample:
            idx_train = _stratified_subsample(labels_arr, subsample, rng)
            arr_train = arr[idx_train]
            labels_train = labels_arr[idx_train]

            if discriminative:
                embedding_train = _spectral_fisher_embed(
                    arr_train,
                    labels_train,
                    n_components,
                    n_neighbors,
                    random_state=random_state,
                )
            else:
                embedding_train = _signed_laplacian_embed(
                    arr_train,
                    labels_train,
                    n_components,
                    n_neighbors,
                    alpha=alpha,
                    random_state=random_state,
                )

            mask_oos = np.ones(n_samples, dtype=bool)
            mask_oos[idx_train] = False
            arr_oos = arr[mask_oos]

            embedding_oos = _nystroem_extend(
                arr_train,
                embedding_train,
                arr_oos,
                n_neighbors,
            )

            embedding = np.empty((n_samples, n_components), dtype=np.float32)
            embedding[idx_train] = embedding_train
            embedding[mask_oos] = embedding_oos
        else:
            if discriminative:
                embedding = _spectral_fisher_embed(
                    arr,
                    labels_arr,
                    n_components,
                    n_neighbors,
                    random_state=random_state,
                )
            else:
                embedding = _signed_laplacian_embed(
                    arr,
                    labels_arr,
                    n_components,
                    n_neighbors,
                    alpha=alpha,
                    random_state=random_state,
                )

        views, emb_for_tour = _cumulative_views(n_components, embedding, n_remove)
        loadings, r2 = _compute_feature_loadings(arr, embedding)

        frame_summaries = None
        if feature_names is not None:
            frame_summaries = _compute_frame_summaries(
                views,
                loadings,
                feature_names,
                tour_mode=tour_mode,
            )

        result = TourResult(
            views=views,
            n_views=len(views),
            n_dims=n_components,
        )
        result.embedding = emb_for_tour
        result.feature_loadings = loadings
        result.feature_names = feature_names
        result.feature_r2 = r2
        result.frame_summaries = frame_summaries
        result.tour_mode = tour_mode
        result.tour_description = _TOUR_DESCRIPTIONS.get(tour_mode)
        result.tour_frame_description = _TOUR_FRAME_DESCRIPTIONS.get(tour_mode)
        return result

    # ── Standard path (vanilla LE, no labels) ─────────────────────────
    kwargs: dict = {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "affinity": "nearest_neighbors",
        "eigen_solver": "amg",
        **({"random_state": random_state} if random_state is not None else {}),
        **(se_kwargs or {}),
    }

    if subsample is not None and n_samples > subsample:
        idx_train = rng.choice(n_samples, size=subsample, replace=False)
        idx_train.sort()
        arr_train = arr[idx_train]

        embedding_train = SpectralEmbedding(**kwargs).fit_transform(arr_train)

        mask_oos = np.ones(n_samples, dtype=bool)
        mask_oos[idx_train] = False
        arr_oos = arr[mask_oos]

        embedding_oos = _nystroem_extend(arr_train, embedding_train, arr_oos, n_neighbors)

        embedding = np.empty((n_samples, n_components), dtype=np.float32)
        embedding[idx_train] = embedding_train
        embedding[mask_oos] = embedding_oos
    else:
        embedding = SpectralEmbedding(**kwargs).fit_transform(arr)

    views, emb_for_tour = _cumulative_views(n_components, embedding, n_remove)
    loadings, r2 = _compute_feature_loadings(arr, embedding)

    frame_summaries = None
    if feature_names is not None:
        frame_summaries = _compute_frame_summaries(
            views,
            loadings,
            feature_names,
            tour_mode=tour_mode,
        )

    result = TourResult(
        views=views,
        n_views=len(views),
        n_dims=n_components,
    )
    result.embedding = emb_for_tour
    result.feature_loadings = loadings
    result.feature_names = feature_names
    result.feature_r2 = r2
    result.frame_summaries = frame_summaries
    result.tour_description = _TOUR_DESCRIPTIONS[None]
    result.tour_frame_description = _TOUR_FRAME_DESCRIPTIONS[None]

    return result
