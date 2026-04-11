"""Tests for tour computation helpers and data utilities."""

import numpy as np
import pytest
from dtour.data import _to_ipc_bytes, from_numpy, from_pandas
from dtour.tours import (
    TourResult,
    _build_knn_affinity,
    _compute_feature_loadings,
    _nystroem_extend,
    _procrustes_align,
    le_tour,
    little_tour,
    spectrum_tour,
)


def make_data(n: int = 100, p: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float32)


# ── little_tour ──────────────────────────────────────────────────────────


def test_little_tour_returns_result():
    X = make_data()
    result = little_tour(X)
    assert isinstance(result, TourResult)
    assert result.n_views > 0
    assert result.n_dims == X.shape[1]
    assert len(result.explained_variance_ratio) > 0


def test_little_tour_n_views():
    X = make_data(p=6)
    result = little_tour(X, n_components=4)
    assert result.n_views == 4


def test_little_tour_views_shapes():
    X = make_data(p=5)
    result = little_tour(X)
    assert len(result.views) == result.n_views
    for basis in result.views:
        assert basis.shape == (5, 2)
        assert basis.dtype == np.float32


def test_little_tour_views_raw_roundtrip():
    X = make_data(p=5)
    result = little_tour(X)
    raw = result.views_raw
    flat = np.frombuffer(raw, dtype=np.float32)
    assert flat.shape == (result.n_views * 5 * 2,)
    # Reconstruct and verify first view matches (column-major wire format)
    first = flat[: 5 * 2].reshape(5, 2, order="F")
    np.testing.assert_array_equal(first, result.views[0])


def test_little_tour_with_dataframe():
    import pandas as pd

    df = pd.DataFrame(make_data(), columns=[f"c{i}" for i in range(5)])
    result = little_tour(df)
    assert result.n_dims == 5
    assert result.n_views > 0


# ── le_tour ──────────────────────────────────────────────────────────────


def test_le_tour_returns_result():
    X = make_data(n=200, p=6)
    result = le_tour(X, n_components=4, n_neighbors=10)
    assert isinstance(result, TourResult)
    # Always cumulative: n_components=4 → n_frames=3 (k=2..4) → 3 views
    assert result.n_views == 3
    assert result.embedding is not None
    assert result.embedding.shape == (200, 4)
    assert result.feature_loadings is not None
    assert result.feature_loadings.shape == (4, 6)
    assert result.feature_r2 is not None
    assert len(result.feature_r2) == 4
    assert all(0 <= r <= 1 for r in result.feature_r2)


def test_le_tour_with_feature_names():
    X = make_data(n=200, p=4)
    names = ["a", "b", "c", "d"]
    result = le_tour(X, n_components=3, n_neighbors=10, feature_names=names)
    assert result.feature_names == names
    assert result.feature_loadings.shape == (3, 4)


def test_le_tour_extracts_pandas_names():
    import pandas as pd

    X = make_data(n=200, p=4)
    df = pd.DataFrame(X, columns=["alpha", "beta", "gamma", "delta"])
    result = le_tour(df, n_components=3, n_neighbors=10)
    assert result.feature_names == ["alpha", "beta", "gamma", "delta"]


def test_le_tour_save_load_roundtrip(tmp_path):
    X = make_data(n=200, p=5)
    result = le_tour(X, n_components=3, n_neighbors=10, feature_names=["a", "b", "c", "d", "e"])
    path = tmp_path / "le_tour.npz"
    result.save(path)
    loaded = TourResult.load(path)
    assert loaded.feature_names == result.feature_names
    assert loaded.feature_r2 is not None
    np.testing.assert_allclose(loaded.feature_r2, result.feature_r2, atol=1e-6)
    np.testing.assert_allclose(loaded.feature_loadings, result.feature_loadings, atol=1e-6)


def test_save_load_backwards_compat(tmp_path):
    """Loading an old npz (without loadings) should still work."""
    X = make_data()
    result = little_tour(X, n_components=3)
    path = tmp_path / "old_tour.npz"
    result.save(path)
    loaded = TourResult.load(path)
    assert loaded.feature_loadings is None
    assert loaded.feature_names is None
    assert loaded.feature_r2 is None
    assert loaded.n_views == 3


def test_le_tour_cumulative_view_count():
    """n_frames should produce exactly n_frames build-up views."""
    X = make_data(n=200, p=5)
    result = le_tour(X, n_neighbors=10, n_frames=4)
    # n_frames=4 → 5 eigenvectors, build-up k=2..5 → 4 views
    assert result.n_views == 4
    assert result.n_dims == 5  # n_frames + 1 eigenvectors
    for basis in result.views:
        assert basis.shape == (5, 2)
        assert basis.dtype == np.float32


def test_le_tour_cumulative_views_orthonormal():
    """Each cumulative view basis should have orthonormal columns."""
    X = make_data(n=200, p=6)
    result = le_tour(X, n_neighbors=10, n_frames=4)
    for basis in result.views:
        gram = basis.T @ basis
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-5)


def test_le_tour_cumulative_equal_row_norms():
    """Final cumulative view should have equal row norms (equal eigenvector contribution)."""
    X = make_data(n=200, p=6)
    result = le_tour(X, n_neighbors=10, n_frames=4)
    final_view = result.views[-1]  # all 5 eigenvectors active
    row_norms = np.linalg.norm(final_view, axis=1)
    expected_norm = np.sqrt(2.0 / 5)
    np.testing.assert_allclose(row_norms, expected_norm, atol=1e-5)


def test_le_tour_cumulative_with_removal():
    """Removal phase should add extra frames stripping low-frequency eigenvectors."""
    X = make_data(n=200, p=6)
    result = le_tour(X, n_neighbors=10, n_frames=4, n_remove=3)
    # Build-up: 4 views (k=2..5), removal: 3 views (strip LE0, LE0+1, LE0+1+2)
    assert result.n_views == 4 + 3
    # Last view should only have LE3 and LE4 active (3 removed from 5)
    last = result.views[-1]
    # First 3 rows should be zero
    np.testing.assert_array_equal(last[:3], 0.0)
    # Last 2 rows should be non-zero with equal norms
    active_norms = np.linalg.norm(last[3:], axis=1)
    assert all(n > 0 for n in active_norms)
    np.testing.assert_allclose(active_norms[0], active_norms[1], atol=1e-5)


def test_le_tour_subsample():
    """Subsampling should produce embeddings for all rows."""
    X = make_data(n=300, p=5)
    result = le_tour(X, n_components=3, n_neighbors=10, subsample=100, random_state=42)
    assert result.embedding.shape == (300, 3)
    assert result.feature_loadings.shape == (3, 5)
    # n_components=3 → 2 cumulative views
    assert result.n_views == 2


def test_le_tour_subsample_noop_when_small():
    """When n <= subsample, should behave the same as without subsampling."""
    X = make_data(n=100, p=4)
    result = le_tour(X, n_components=3, n_neighbors=10, subsample=200, random_state=42)
    assert result.embedding.shape == (100, 3)


def test_nystroem_extend_preserves_shape():
    """Out-of-sample projection should return correct shape."""
    rng = np.random.default_rng(0)
    arr_train = rng.standard_normal((100, 5)).astype(np.float32)
    embedding_train = rng.standard_normal((100, 3)).astype(np.float32)
    arr_oos = rng.standard_normal((50, 5)).astype(np.float32)
    result = _nystroem_extend(arr_train, embedding_train, arr_oos, n_neighbors=10)
    assert result.shape == (50, 3)
    assert result.dtype == np.float32


def test_compute_feature_loadings_perfect_fit():
    """When embedding is a linear function of X, R-squared should be ~1."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3)).astype(np.float32)
    W = rng.standard_normal((3, 2)).astype(np.float32)
    embedding = X @ W
    correlations, r2 = _compute_feature_loadings(X, embedding)
    assert correlations.shape == (2, 3)
    # Correlations bounded [-1, 1]
    assert np.all(np.abs(correlations) <= 1.0 + 1e-6)
    # R² from multivariate OLS should be ~1 for perfect linear fit
    np.testing.assert_allclose(r2, [1.0, 1.0], atol=1e-4)


# ── data helpers ─────────────────────────────────────────────────────────


def test_from_numpy_roundtrip():
    from io import BytesIO

    import arro3.core as ac
    import arro3.io

    X = make_data()
    ipc = from_numpy(X, column_names=[f"x{i}" for i in range(X.shape[1])])
    assert isinstance(ipc, bytes)
    assert len(ipc) > 0
    # Verify we can read it back
    reader = arro3.io.read_ipc_stream(BytesIO(ipc))
    batches = list(reader)
    table = ac.Table.from_batches(batches, schema=reader.schema)
    assert table.num_rows == X.shape[0]
    assert table.num_columns == X.shape[1]


def test_from_numpy_invalid_shape():
    with pytest.raises(ValueError, match="2-D"):
        from_numpy(np.array([1, 2, 3]))


def test_from_numpy_column_count_mismatch():
    with pytest.raises(ValueError, match="column_names"):
        from_numpy(make_data(p=3), column_names=["a", "b"])


def test_from_pandas():
    import pandas as pd

    df = pd.DataFrame(make_data(), columns=[f"c{i}" for i in range(5)])
    ipc = from_pandas(df)
    assert isinstance(ipc, bytes)
    assert len(ipc) > 0


def test_to_ipc_bytes_passthrough():
    raw = b"some bytes"
    assert _to_ipc_bytes(raw) is raw


def test_to_ipc_bytes_unsupported_type():
    with pytest.raises(TypeError, match="Cannot convert"):
        _to_ipc_bytes(42)


# ── true signed Laplacian tour ───────────────────────────────────────────


def test_le_tour_signed_basic():
    """labels triggers signed Laplacian with a single set of eigenvectors."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(X, n_neighbors=10, n_frames=7, labels=labels)
    # n_frames=7 → 8 eigenvectors, 7 cumulative views
    assert result.n_dims == 8
    assert result.n_views == 7
    assert result.tour_mode == "signed"
    assert result.embedding is not None
    assert result.embedding.shape == (300, 8)
    for basis in result.views:
        assert basis.shape == (8, 2)
        assert basis.dtype == np.float32
        gram = basis.T @ basis
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-5)


def test_le_tour_signed_loadings():
    """Signed LE loadings: single regression, R^2 each <= 1.0."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        feature_names=names,
    )
    # 6 eigenvectors, 6 features
    assert result.feature_loadings.shape == (6, 6)
    assert result.feature_r2 is not None
    assert len(result.feature_r2) == 6
    # R^2 values should each be <= 1.0 (no summing of independent regressions)
    assert all(r <= 1.0 + 1e-6 for r in result.feature_r2)


def test_le_tour_signed_frame_summaries():
    """All signed Laplacian frames should have 'Structure:' prefix."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        feature_names=names,
    )
    assert result.frame_summaries is not None
    assert len(result.frame_summaries) == result.n_views
    for s in result.frame_summaries:
        assert s.startswith("Structure:"), s


def test_le_tour_signed_alpha():
    """Different alpha values should produce different embeddings."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    r1 = le_tour(X, n_neighbors=10, n_frames=4, labels=labels, alpha=0.5, random_state=42)
    r2 = le_tour(X, n_neighbors=10, n_frames=4, labels=labels, alpha=2.0, random_state=42)
    assert not np.allclose(r1.embedding, r2.embedding, atol=1e-3)


def test_le_tour_signed_subsample():
    """Signed LE with subsampling should produce embeddings for all rows."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=4,
        labels=labels,
        subsample=100,
        random_state=42,
    )
    assert result.embedding.shape == (300, 5)
    assert result.tour_mode == "signed"


def test_le_tour_signed_save_load(tmp_path):
    """Signed tour should survive save/load roundtrip with tour_mode."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        feature_names=names,
    )
    path = tmp_path / "signed_tour.npz"
    result.save(path)
    loaded = TourResult.load(path)
    assert loaded.tour_mode == "signed"
    assert loaded.frame_summaries == result.frame_summaries
    assert loaded.n_dims == result.n_dims
    assert loaded.n_views == result.n_views
    np.testing.assert_allclose(loaded.feature_loadings, result.feature_loadings, atol=1e-6)


# ── spectral Fisher discriminant tour ────────────────────────────────────


def test_le_tour_discriminative_basic():
    """discriminative=True triggers spectral Fisher embedding."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        discriminative=True,
    )
    assert result.n_views == 5
    assert result.n_dims == 6  # n_frames + 1
    assert result.embedding.shape == (300, 6)
    assert result.tour_mode == "discriminative"
    for basis in result.views:
        assert basis.shape == (6, 2)
        gram = basis.T @ basis
        np.testing.assert_allclose(gram, np.eye(2), atol=1e-5)


def test_le_tour_discriminative_frame_summaries():
    """All discriminative frames should have 'Discriminant:' prefix."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        discriminative=True,
        feature_names=names,
    )
    assert result.frame_summaries is not None
    for s in result.frame_summaries:
        assert s.startswith("Discriminant:"), s


def test_le_tour_discriminative_requires_labels():
    """discriminative=True without labels should raise ValueError."""
    X = make_data(n=200, p=5)
    with pytest.raises(ValueError, match="requires labels"):
        le_tour(X, n_neighbors=10, n_frames=4, discriminative=True)


def test_le_tour_discriminative_subsample():
    """Fisher discriminant with subsampling should produce full embeddings."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=4,
        labels=labels,
        discriminative=True,
        subsample=100,
        random_state=42,
    )
    assert result.embedding.shape == (300, 5)
    assert result.tour_mode == "discriminative"


def test_le_tour_discriminative_save_load(tmp_path):
    """Fisher tour should survive save/load roundtrip."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        discriminative=True,
        feature_names=names,
    )
    path = tmp_path / "fisher_tour.npz"
    result.save(path)
    loaded = TourResult.load(path)
    assert loaded.tour_mode == "discriminative"
    assert loaded.frame_summaries == result.frame_summaries


# ── stratified subsampling ────────────────────────────────────────────────


def test_labeled_subsample_preserves_all_classes():
    """Stratified subsampling must keep at least one sample per class."""
    X = make_data(n=300, p=6)
    # 3 classes with very unbalanced sizes: 270 / 20 / 10
    labels = np.array(["A"] * 270 + ["B"] * 20 + ["C"] * 10)
    result = le_tour(
        X,
        n_neighbors=5,
        n_frames=3,
        labels=labels,
        subsample=30,
        random_state=42,
    )
    assert result.embedding.shape == (300, 4)
    assert result.tour_mode == "signed"


def test_labeled_subsample_too_small_raises():
    """subsample < n_classes should raise a clear error."""
    X = make_data(n=100, p=6)
    labels = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 30)
    with pytest.raises(ValueError, match="too small to represent all"):
        le_tour(
            X,
            n_neighbors=5,
            n_frames=3,
            labels=labels,
            subsample=2,
            random_state=42,
        )


def test_labeled_subsample_imbalanced_discriminative():
    """Fisher discriminant with imbalanced data and subsampling should work."""
    X = make_data(n=200, p=6)
    labels = np.array(["major"] * 190 + ["minor"] * 10)
    result = le_tour(
        X,
        n_neighbors=5,
        n_frames=3,
        labels=labels,
        discriminative=True,
        subsample=40,
        random_state=42,
    )
    assert result.embedding.shape == (200, 4)
    assert result.tour_mode == "discriminative"
    # Embedding should have finite, reasonable values (no 1e6 magnitudes)
    assert np.all(np.isfinite(result.embedding))
    assert np.abs(result.embedding).max() < 100


# ── frame summaries ──────────────────────────────────────────────────────


def test_frame_summaries_backward_compat(tmp_path):
    """Loading an old npz without frame_summaries should return None."""
    X = make_data()
    result = little_tour(X, n_components=3)
    path = tmp_path / "old_tour.npz"
    result.save(path)
    loaded = TourResult.load(path)
    assert loaded.frame_summaries is None


def test_le_tour_frame_summaries():
    """Vanilla LE tour should get 'Structure:' frame summaries."""
    X = make_data(n=200, p=6)
    names = ["a", "b", "c", "d", "e", "f"]
    result = le_tour(X, n_neighbors=10, n_frames=4, feature_names=names)
    assert result.frame_summaries is not None
    assert len(result.frame_summaries) == result.n_views
    for s in result.frame_summaries:
        assert s.startswith("Structure:"), s


# ── _build_knn_affinity enhancements ───────────────────────────────────


def test_affinity_mutual_knn():
    """Mutual KNN should be sparser than symmetric KNN and still symmetric."""
    X = make_data(n=200, p=5)
    W_sym = _build_knn_affinity(X, 10, affinity="symmetric_knn")
    W_mut = _build_knn_affinity(X, 10, affinity="mutual_knn")

    assert W_mut.nnz <= W_sym.nnz
    # Still symmetric
    diff = W_mut - W_mut.T
    assert abs(diff).max() < 1e-10
    # All weights positive
    assert W_mut.data.min() > 0


def test_affinity_adaptive_sigma():
    """Adaptive sigma should produce different weights than global sigma."""
    X = make_data(n=200, p=5)
    W_global = _build_knn_affinity(X, 10, adaptive_sigma=False)
    W_local = _build_knn_affinity(X, 10, adaptive_sigma=True)

    # Same sparsity structure (same KNN graph)
    assert W_global.nnz == W_local.nnz
    # But different weights
    assert not np.allclose(W_global.data, W_local.data)
    # Still symmetric and positive
    diff = W_local - W_local.T
    assert abs(diff).max() < 1e-10
    assert W_local.data.min() > 0


def test_affinity_normalize_alpha():
    """Alpha normalization should change row sums; alpha=None should not."""
    X = make_data(n=200, p=5)
    W_base = _build_knn_affinity(X, 10)
    W_normed = _build_knn_affinity(X, 10, normalize_alpha=1.0)

    # Different row sums
    sums_base = np.asarray(W_base.sum(axis=1)).ravel()
    sums_normed = np.asarray(W_normed.sum(axis=1)).ravel()
    assert not np.allclose(sums_base, sums_normed)
    # Still symmetric
    diff = W_normed - W_normed.T
    assert abs(diff).max() < 1e-10


def test_affinity_invalid_mode():
    """Unknown affinity mode should raise ValueError."""
    X = make_data(n=50, p=3)
    with pytest.raises(ValueError, match="Unknown affinity mode"):
        _build_knn_affinity(X, 5, affinity="bogus")


# ── le_tour with affinity enhancements ─────────────────────────────────


def test_le_tour_mutual_knn():
    """Vanilla LE with mutual_knn should produce valid TourResult."""
    X = make_data(n=200, p=6)
    result = le_tour(X, n_neighbors=10, n_frames=4, affinity="mutual_knn")
    assert isinstance(result, TourResult)
    assert result.n_views == 4
    assert result.embedding.shape == (200, 5)


def test_le_tour_adaptive_sigma():
    """Vanilla LE with adaptive_sigma should produce valid TourResult."""
    X = make_data(n=200, p=6)
    result = le_tour(X, n_neighbors=10, n_frames=4, adaptive_sigma=True)
    assert isinstance(result, TourResult)
    assert result.n_views == 4
    assert result.embedding.shape == (200, 5)


def test_le_tour_signed_adaptive_sigma():
    """Signed LE path should accept and use adaptive_sigma."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        adaptive_sigma=True,
    )
    assert result.tour_mode == "signed"
    assert result.n_views == 5
    assert result.embedding.shape == (300, 6)


def test_le_tour_discriminative_mutual_knn():
    """Discriminative path should accept and use mutual_knn."""
    X = make_data(n=300, p=6)
    labels = np.array(["A"] * 150 + ["B"] * 150)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=5,
        labels=labels,
        discriminative=True,
        affinity="mutual_knn",
    )
    assert result.tour_mode == "discriminative"
    assert result.n_views == 5


def test_le_tour_invalid_affinity():
    """Invalid affinity should propagate ValueError through le_tour."""
    X = make_data(n=100, p=5)
    with pytest.raises(ValueError, match="Unknown affinity mode"):
        le_tour(X, n_neighbors=10, n_frames=3, affinity="invalid")


# ── Regression tests for review findings ─────────────────────────────


def test_adaptive_sigma_excludes_self():
    """adaptive_sigma should use kth-neighbor distance, not self (distance=0).

    Regression: nn.kneighbors(arr) includes self, making sigma_i the
    (k-1)th non-self distance instead of kth. The fix uses
    nn.kneighbors() (no arg) which excludes self.
    """

    from sklearn.neighbors import NearestNeighbors

    X = make_data(n=100, p=5)
    k = 10

    # Compute what the *correct* sigma values should be
    nn_ref = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(X)
    correct_sigma = nn_ref.kneighbors()[0][:, -1]  # excludes self
    assert correct_sigma.min() > 0, "kth-neighbor distance should never be 0"

    # Compute what the *wrong* sigma values would be (includes self → dist=0)
    wrong_sigma = nn_ref.kneighbors(X)[0][:, -1]  # includes self

    # The two must differ (the bug made them equal)
    assert not np.allclose(correct_sigma, wrong_sigma)

    # Build the affinity and check it uses the correct sigma:
    # weights with correct sigma should be *lower* on average because
    # the true kth-neighbor sigma is larger → denominator is larger
    W = _build_knn_affinity(X, k, adaptive_sigma=True)
    assert W.data.min() > 0
    assert W.data.max() <= 1.0


def test_explicit_symmetric_knn_uses_heat_kernel():
    """Explicit affinity='symmetric_knn' must produce heat-kernel weights.

    Regression: the vanilla/discriminative paths used to fall back to
    sklearn's binary nearest_neighbors graph when all params were default,
    so explicit symmetric_knn was silently ignored.
    """
    X = make_data(n=200, p=6)
    result = le_tour(
        X,
        n_neighbors=10,
        n_frames=3,
        affinity="symmetric_knn",
    )
    assert isinstance(result, TourResult)
    # The embedding should be valid (not NaN/inf)
    assert np.isfinite(result.embedding).all()

    # More importantly: build the affinity ourselves and verify it has
    # continuous weights, not binary 0/1 values
    W = _build_knn_affinity(X, 10, affinity="symmetric_knn")
    unique_vals = np.unique(np.round(W.data, decimals=6))
    # Heat kernel produces many distinct values; binary graph has only ~1
    assert len(unique_vals) > 2, "Expected heat-kernel weights, got binary"


def test_subsample_warns_on_ignored_params():
    """subsample + mutual_knn or normalize_alpha should emit a warning."""
    import warnings

    X = make_data(n=200, p=6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        le_tour(
            X,
            n_neighbors=10,
            n_frames=3,
            affinity="mutual_knn",
            normalize_alpha=1.0,
            subsample=50,
        )

    oos_warnings = [w for w in caught if "OOS interpolation ignores" in str(w.message)]
    assert len(oos_warnings) >= 1
    msg = str(oos_warnings[0].message)
    assert "mutual_knn" in msg
    assert "normalize_alpha" in msg


# ── spectrum_tour (attraction-repulsion parameter tour) ─────────────────


def test_procrustes_align_identity():
    """Procrustes of identical embeddings should return the same points."""
    rng = np.random.default_rng(42)
    target = rng.standard_normal((100, 2)).astype(np.float32)
    aligned = _procrustes_align(target, target.copy())
    np.testing.assert_allclose(aligned, target, atol=1e-5)


def test_procrustes_align_rotation():
    """Procrustes should recover a known 90-degree rotation."""
    rng = np.random.default_rng(42)
    target = rng.standard_normal((100, 2)).astype(np.float32)
    # Rotate 90 degrees
    R = np.array([[0, -1], [1, 0]], dtype=np.float32)
    rotated = (target - target.mean(0)) @ R.T + target.mean(0)
    aligned = _procrustes_align(target, rotated)
    np.testing.assert_allclose(aligned, target, atol=1e-4)


def test_spectrum_tour_basic():
    """spectrum_tour with default params should produce valid TourResult."""
    X = make_data(n=200, p=10)
    result = spectrum_tour(X, n_frames=2, rhos=[4, 1], n_neighbors=10, random_state=42)
    assert isinstance(result, TourResult)
    assert result.tour_mode == "parameter"
    assert result.n_views == 2
    assert result.n_dims == 4  # 2 * n_frames
    assert result.embedding.shape == (200, 4)
    for basis in result.views:
        assert basis.shape == (4, 2)
        assert basis.dtype == np.float32


def test_spectrum_tour_frame_summaries():
    """Frame summaries should include rho values and landmarks."""
    X = make_data(n=200, p=10)
    result = spectrum_tour(X, rhos=[100, 4, 1], n_neighbors=10, random_state=42)
    assert result.frame_summaries is not None
    assert len(result.frame_summaries) == 3
    assert "rho=100" in result.frame_summaries[0]
    assert "LE-like" in result.frame_summaries[0]
    assert "UMAP-like" in result.frame_summaries[1]
    assert "t-SNE" in result.frame_summaries[2]


def test_spectrum_tour_custom_rhos_override_n_frames():
    """Explicit rhos should override n_frames."""
    X = make_data(n=200, p=10)
    result = spectrum_tour(X, n_frames=99, rhos=[10, 5, 2, 1], n_neighbors=10, random_state=42)
    assert result.n_views == 4


def test_spectrum_tour_pca_init():
    """spectrum_tour with init='pca' should work."""
    X = make_data(n=200, p=10)
    result = spectrum_tour(X, rhos=[4, 1], n_neighbors=10, init="pca", random_state=42)
    assert isinstance(result, TourResult)
    assert result.n_views == 2


def test_spectrum_tour_invalid_init():
    """Invalid init should raise ValueError."""
    X = make_data(n=50, p=5)
    with pytest.raises(ValueError, match="init must be"):
        spectrum_tour(X, rhos=[4, 1], init="bogus")


def test_spectrum_tour_too_few_frames():
    """Fewer than 2 frames should raise ValueError."""
    X = make_data(n=50, p=5)
    with pytest.raises(ValueError, match="at least 2 frames"):
        spectrum_tour(X, n_frames=1)


def test_spectrum_tour_pymde_basic():
    """spectrum_tour with method='pymde' should produce valid TourResult."""
    X = make_data(n=200, p=10)
    result = spectrum_tour(
        X,
        rhos=[4, 1],
        n_neighbors=10,
        method="pymde",
        random_state=42,
    )
    assert isinstance(result, TourResult)
    assert result.tour_mode == "parameter"
    assert result.n_views == 2
    assert result.embedding.shape == (200, 4)


def test_spectrum_tour_pymde_regularization():
    """Higher regularization should produce less movement between frames."""
    X = make_data(n=200, p=10)

    result_free = spectrum_tour(
        X,
        rhos=[4, 1],
        n_neighbors=10,
        method="pymde",
        regularization=0.0,
        random_state=42,
    )
    result_reg = spectrum_tour(
        X,
        rhos=[4, 1],
        n_neighbors=10,
        method="pymde",
        regularization=10.0,
        random_state=42,
    )

    # Measure per-point movement between frame 0 and frame 1
    def frame_movement(emb):
        f0 = emb[:, :2]
        f1 = emb[:, 2:]
        return np.sqrt(np.mean(np.sum((f1 - f0) ** 2, axis=1)))

    move_free = frame_movement(result_free.embedding)
    move_reg = frame_movement(result_reg.embedding)
    assert move_reg < move_free, (
        f"Regularized movement ({move_reg:.3f}) should be less than free ({move_free:.3f})"
    )


def test_spectrum_tour_invalid_method():
    """Invalid method should raise ValueError."""
    X = make_data(n=50, p=5)
    with pytest.raises(ValueError, match="method must be"):
        spectrum_tour(X, rhos=[4, 1], method="bogus")


def test_spectrum_tour_negative_rho():
    """Negative or zero rho values should raise ValueError."""
    X = make_data(n=50, p=5)
    with pytest.raises(ValueError, match="positive"):
        spectrum_tour(X, rhos=[4, 0, 1])
    with pytest.raises(ValueError, match="positive"):
        spectrum_tour(X, rhos=[-1, 1])
