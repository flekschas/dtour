"""Basic tests for tour computation helpers."""

import numpy as np
import pytest
import pyarrow as pa

from dtour.tours import little_tour, TourResult
from dtour.data import from_numpy, from_pandas


def make_data(n: int = 100, p: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float32)


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


def test_little_tour_bases_ipc_roundtrip():
    X = make_data()
    result = little_tour(X)
    table = pa.ipc.open_stream(result.bases).read_all()
    assert table.num_rows == result.n_views


def test_from_numpy_roundtrip():
    X = make_data()
    ipc = from_numpy(X, column_names=[f"x{i}" for i in range(X.shape[1])])
    table = pa.ipc.open_stream(ipc).read_all()
    assert table.num_rows == X.shape[0]
    assert table.num_columns == X.shape[1]


def test_from_pandas():
    import pandas as pd

    df = pd.DataFrame(make_data(), columns=[f"c{i}" for i in range(5)])
    ipc = from_pandas(df)
    table = pa.ipc.open_stream(ipc).read_all()
    assert table.num_rows == len(df)
