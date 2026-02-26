"""Helpers for converting data to Arrow IPC bytes for the widget."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def from_numpy(X: np.ndarray, column_names: list[str] | None = None) -> bytes:
    """Convert a 2-D float numpy array to Arrow IPC bytes.

    Args:
        X: Shape (n_samples, n_dims). Will be cast to float32.
        column_names: Optional list of column names. Defaults to "dim_0", "dim_1", ...
    """
    import arro3.core as ac
    import arro3.io

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    _, n_dims = X.shape
    names = column_names or [f"dim_{i}" for i in range(n_dims)]
    if len(names) != n_dims:
        raise ValueError(f"column_names has {len(names)} items, expected {n_dims}")

    arrays = {name: ac.Array.from_numpy(np.ascontiguousarray(X[:, i], dtype=np.float32)) for i, name in enumerate(names)}
    table = ac.Table.from_pydict(arrays)

    buf = BytesIO()
    arro3.io.write_ipc_stream(table, buf, compression=None)
    return buf.getvalue()


def from_pandas(df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
    """Convert a pandas DataFrame to Arrow IPC bytes.

    Only float-compatible columns are included. Pass ``columns`` to select a subset.
    Converts via numpy to avoid requiring pyarrow.
    """
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    subset = df[cols]
    return from_numpy(subset.to_numpy(dtype=np.float32), column_names=list(cols))


def from_arrow(table: object) -> bytes:
    """Serialize any Arrow-compatible object to Arrow IPC bytes.

    Accepts anything with ``__arrow_c_stream__`` (pyarrow Table, polars
    DataFrame, arro3 Table, DuckDB relation, etc.).
    """
    return _to_ipc_bytes(table)


def _to_ipc_bytes(data: object) -> bytes:
    """Convert *data* to Arrow IPC stream bytes.

    Accepts:
    - ``bytes`` — assumed to be Arrow IPC already, returned as-is
    - ``str`` or ``Path`` — read file contents
    - ``np.ndarray`` — 2-D array, converted via :func:`from_numpy`
    - Anything with ``__arrow_c_stream__`` — serialized via arro3
    """
    import arro3.core as ac
    import arro3.io

    if isinstance(data, bytes):
        return data

    if isinstance(data, (str, Path)):
        return Path(data).read_bytes()

    if isinstance(data, np.ndarray):
        return from_numpy(data)

    if hasattr(data, "__arrow_c_stream__"):
        table = ac.Table.from_arrow(data)
        buf = BytesIO()
        arro3.io.write_ipc_stream(table, buf, compression=None)
        return buf.getvalue()

    raise TypeError(
        f"Cannot convert {type(data).__name__} to Arrow IPC bytes. "
        "Pass bytes, a file path, a numpy ndarray, or an object with "
        "__arrow_c_stream__ (pandas DataFrame, polars DataFrame, pyarrow Table, etc.)."
    )
