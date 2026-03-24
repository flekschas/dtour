"""Tests for tour computation helpers and data utilities."""

import numpy as np
import pytest

from dtour.data import _to_ipc_bytes, from_numpy, from_pandas
from dtour.tours import TourResult, little_tour


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
