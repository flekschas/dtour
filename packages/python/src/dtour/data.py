"""Helpers for converting data to Arrow IPC bytes for the widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd


def from_numpy(X: np.ndarray, column_names: list[str] | None = None) -> bytes:
    """Convert a 2-D float numpy array to Arrow IPC bytes.

    Args:
        X: Shape (n_samples, n_dims). Will be cast to float32.
        column_names: Optional list of column names. Defaults to "dim_0", "dim_1", ...
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    n_rows, n_dims = X.shape
    names = column_names or [f"dim_{i}" for i in range(n_dims)]
    if len(names) != n_dims:
        raise ValueError(f"column_names has {len(names)} items, expected {n_dims}")

    arrays = [pa.array(X[:, i].astype(np.float32)) for i in range(n_dims)]
    table = pa.table(dict(zip(names, arrays)))
    return _table_to_ipc_bytes(table)


def from_pandas(df: "pd.DataFrame", columns: list[str] | None = None) -> bytes:
    """Convert a pandas DataFrame to Arrow IPC bytes.

    Only float-compatible columns are included. Pass `columns` to select a subset.
    """
    import pandas as pd

    cols = columns or df.select_dtypes(include="number").columns.tolist()
    subset = df[cols].astype(np.float32)
    table = pa.Table.from_pandas(subset, preserve_index=False)
    return _table_to_ipc_bytes(table)


def from_arrow(table: pa.Table) -> bytes:
    """Serialize an existing PyArrow Table to Arrow IPC bytes."""
    return _table_to_ipc_bytes(table)


def _table_to_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return sink.getvalue().to_pybytes()
