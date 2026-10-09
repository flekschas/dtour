"""Helpers for converting data to Arrow IPC bytes for the widget."""

from __future__ import annotations

import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arro3.core as ac
    import pandas as pd


def from_numpy(X: np.ndarray, column_names: list[str] | None = None) -> bytes:
    """Convert a 2-D float numpy array to Arrow IPC bytes.

    Args:
        X: Shape (n_samples, n_dims). Will be cast to float32.
        column_names: Optional list of column names. Defaults to "dim_0", "dim_1", ...
    """
    import arro3.core as ac

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")

    _, n_dims = X.shape
    names = column_names or [f"dim_{i}" for i in range(n_dims)]
    if len(names) != n_dims:
        raise ValueError(f"column_names has {len(names)} items, expected {n_dims}")

    arrays = {
        name: ac.Array.from_numpy(np.ascontiguousarray(X[:, i], dtype=np.float32))
        for i, name in enumerate(names)
    }
    return _write_ipc(ac.Table.from_pydict(arrays))


def from_pandas(df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
    """Convert a pandas DataFrame to Arrow IPC bytes.

    Numeric columns (``select_dtypes(include="number")``, which includes timedeltas)
    become float32 columns, matching the dimensions the tour functions use. All other
    columns become string columns, which the widget treats as categories (e.g., for
    ``point_color_by``). By default, only numeric, categorical, string, object, and
    boolean columns are included; ``columns`` selects the columns to include instead.
    Converts via numpy and Python lists to avoid requiring pyarrow.
    """
    import arro3.core as ac

    numeric = set(df.select_dtypes(include="number").columns)
    if columns is None:
        include = ["number", "category", "object", "string", "bool"]
        columns = df.select_dtypes(include=include).columns.tolist()

    names = [str(name) for name in columns]
    if len(set(names)) != len(names):
        raise ValueError(f"Column names must be unique as strings; got {names}")

    arrays = {}
    for name, key in zip(columns, names):
        col = df[name]
        if name in numeric:
            values = col.to_numpy(dtype=np.float32, na_value=np.nan)
            arrays[key] = ac.Array.from_numpy(np.ascontiguousarray(values))
        else:
            labels = col.astype(str).tolist()
            for i in np.flatnonzero(col.isna().to_numpy()):
                labels[i] = None
            arrays[key] = ac.Array(labels, type=ac.DataType.string())
    return _write_ipc(ac.Table.from_pydict(arrays))


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
    - ``str`` or ``Path`` — read file contents (Arrow or Parquet, kept as-is)
    - ``np.ndarray`` — 2-D array, converted via :func:`from_numpy`
    - Anything with ``__arrow_c_stream__`` — serialized via arro3
    """
    import arro3.core as ac

    if isinstance(data, bytes):
        return data

    if isinstance(data, (str, Path)):
        return Path(data).read_bytes()

    if isinstance(data, np.ndarray):
        return from_numpy(data)

    # Route pandas DataFrames through from_pandas() so pyarrow isn't needed
    type_name = type(data).__qualname__
    module = type(data).__module__ or ""
    if type_name == "DataFrame" and module.startswith("pandas"):
        return from_pandas(data)

    if hasattr(data, "__arrow_c_stream__"):
        return _write_ipc(ac.Table.from_arrow(data))

    raise TypeError(
        f"Cannot convert {type(data).__name__} to Arrow IPC bytes. "
        "Pass bytes, a file path, a numpy ndarray, or an object with "
        "__arrow_c_stream__ (pandas DataFrame, polars DataFrame, pyarrow Table, etc.)."
    )


def _write_ipc(table: object) -> bytes:
    import arro3.io

    buf = BytesIO()
    arro3.io.write_ipc_stream(table, buf, compression=None)
    return buf.getvalue()


def _reader(buf: bytes) -> Any:
    """Open Parquet, Arrow IPC file, or Arrow IPC stream bytes."""
    import arro3.io

    if buf[:4] == b"PAR1":
        return arro3.io.read_parquet(BytesIO(buf))
    if buf[:6] == b"ARROW1":
        return arro3.io.read_ipc(BytesIO(buf))
    return arro3.io.read_ipc_stream(BytesIO(buf))


def _read_table(buf: bytes) -> ac.Table:
    import arro3.core as ac

    return ac.Table.from_arrow(_reader(buf))


def _numeric_columns(buf: bytes) -> list[str]:
    """Names of the columns the viewer reads as numeric dimensions, in order."""
    return [f.name for f in _reader(buf).schema if _is_numeric_field(f)]


def _is_numeric_field(field: Any) -> bool:
    """Whether the viewer reads this column as a numeric dimension."""
    import arro3.core as ac

    if re.fullmatch(r"__index_level_\d+__", field.name):
        return False
    t = field.type
    return ac.DataType.is_floating(t) or ac.DataType.is_integer(t) or ac.DataType.is_boolean(t)


def _add_embedding(
    table: ac.Table | None, embedding: np.ndarray, names: list[str] | None
) -> tuple[ac.Table, list[str]]:
    """Put the embedding columns first, ahead of the columns of *table*.

    Rows are matched by position. Returns the table and the names of its
    embedding columns. If the first numeric columns of *table* equal the
    embedding, reuses those columns and warns. Without *names*, generates
    names that don't clash with the columns of *table*.
    """
    import arro3.core as ac

    n, p = embedding.shape
    columns = [] if table is None else table.column_names
    if names is None:
        names = []
        for i in range(p):
            name = f"embedding_{i}"
            while name in columns:
                name = f"_{name}"
            names.append(name)
    elif len(names) != p or len(set(names)) != p:
        raise ValueError(
            f"tour.embedding_names must be {p} unique names, one per embedding column; got {names}"
        )

    if table is not None:
        if table.num_rows != n:
            raise ValueError(
                f"The data has {table.num_rows} rows but the tour's embedding has {n}. "
                "Pass the data the tour was computed from, with one row per point in "
                "the same order."
            )

        leading = [f.name for f in table.schema if _is_numeric_field(f)][:p]
        if len(leading) == p and all(
            _equals(table.column(name), embedding[:, i]) for i, name in enumerate(leading)
        ):
            warnings.warn(
                "The data already starts with the tour's embedding. Pass the data the tour "
                "was computed from instead; the widget adds the embedding columns itself.",
                FutureWarning,
                stacklevel=2,
            )
            return table, leading

        clashes = [name for name in names if name in columns]
        if clashes:
            raise ValueError(
                f"The data already has columns named {clashes}, which the tour's embedding "
                "uses. Rename them or set tour.embedding_names."
            )

    arrays = {
        name: ac.Array.from_numpy(np.ascontiguousarray(embedding[:, i], dtype=np.float32))
        for i, name in enumerate(names)
    }
    for name in columns:
        arrays[name] = table.column(name)
    return ac.Table.from_pydict(arrays), names


def _equals(column: Any, values: np.ndarray) -> bool:
    try:
        return np.array_equal(column.to_numpy(), values)
    except Exception:  # e.g., nulls or non-numeric types
        return False
