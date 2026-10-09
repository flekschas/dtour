"""MCP server that shows dtour tours inline in chat apps like Claude Desktop.

The model calls ``visualize`` with a data file. The server computes the tour and
stores the data and tour as one Parquet file. The viewer, an MCP App in the
host's iframe, fetches that file from a localhost HTTP server, or, if the host
blocks localhost, in chunks through the app-only ``read_data`` tool. Hosts
without MCP Apps get a link to the same viewer in the browser.

Run with ``dtour-mcp`` (stdio). Logs go to stderr because stdout carries the protocol.
"""

from __future__ import annotations

import base64
import secrets
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from mcp.server.apps import Apps, ResourceCsp, client_supports_apps
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent

from .data import _is_numeric_field, _read_table, _write_ipc
from .spec import add_spec_to_parquet
from .tours import TourResult, le_tour, little_tour, umap_little_tour
from .widget import _viewer_data

if TYPE_CHECKING:
    import polars as pl

VIEWER_URI = "ui://dtour/viewer.html"

# Bytes per `read_data` call. Claude Desktop drops the connection on messages
# over 32 MiB, and base64 adds a third.
CHUNK_SIZE = 8 * 1024 * 1024

# Embedding tours subsample larger data for the spectral step
LE_SUBSAMPLE = 50_000

Tour = Literal["pca", "le", "umap"]

# Markers of missing values in CSV files, e.g., `NA` from R
NULL_VALUES = ["", "NA", "N/A", "NaN", "nan", "null", "NULL"]


class _View(NamedTuple):
    """A visualization: its data with the tour embedded, and the tour's input columns."""

    parquet: bytes
    columns: list[str]


class _Views:
    """The most recent views, by token. Older views are forgotten."""

    def __init__(self, capacity: int = 16) -> None:
        self._views: OrderedDict[str, _View] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, view: _View) -> str:
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._views[token] = view
            while len(self._views) > self._capacity:
                self._views.popitem(last=False)
        return token

    def get(self, token: str) -> _View:
        with self._lock:
            view = self._views.get(token)
        if view is None:
            raise ToolError("This view's data is gone. Ask for the visualization again.")
        return view


def _viewer_script() -> str:
    script = (Path(__file__).parent / "static" / "mcp-app.js").read_text()
    # An inline script ends at the first `</script`, even inside a string
    return script.replace("</script", "<\\/script")


def _viewer_html(script: str) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>dtour</title></head>"
        f'<body><script type="module">{script}</script></body></html>'
    )


def _start_http_server(views: _Views, html: str) -> int:
    """Serve the data files and a standalone viewer on a free localhost port.

    ``/data/<token>`` is a Parquet file and ``/view/<token>`` the viewer showing it.
    Returns the port.
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes = b"", content_type: str = "") -> None:
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            # Chromium asks before a page on a public origin may reach localhost
            self.send_header("Access-Control-Allow-Private-Network", "true")
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self) -> None:
            kind, _, token = self.path.lstrip("/").partition("/")
            try:
                view = views.get(token)
            except ToolError:
                view = None
            if view is None or kind not in ("data", "view"):
                self._send(404)
            elif kind == "data":
                self._send(200, view.parquet, "application/vnd.apache.parquet")
            else:
                self._send(200, html.encode(), "text/html; charset=utf-8")

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def _read(path: str) -> pl.DataFrame:
    """Read a CSV, Parquet, or Arrow file from a local path or URL."""
    import polars as pl

    is_url = "://" in path
    source = path if is_url else str(Path(path).expanduser())
    suffix = Path(path.split("?")[0]).suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        separator = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(
            source, separator=separator, infer_schema_length=10_000, null_values=NULL_VALUES
        )
    if suffix in (".arrow", ".feather", ".ipc"):
        # `_read_table` reads both the Arrow IPC file and stream formats
        data = urllib.request.urlopen(path).read() if is_url else Path(source).read_bytes()
        return pl.DataFrame(_read_table(data))
    return pl.read_parquet(source)


def _prepare(
    df: pl.DataFrame, columns: list[str] | None, sample: int | None
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """The data to show, the columns to tour, and notes about what changed.

    Numeric columns with missing values break the projection, so rows missing a
    tour column are dropped, and other numeric columns with missing values too.
    """
    import arro3.core as ac
    import polars as pl

    # The viewer reads neither Decimal nor boolean values as numbers
    df = df.with_columns(pl.col(pl.Decimal).cast(pl.Float64), pl.col(pl.Boolean).cast(pl.UInt8))
    numeric = [field.name for field in ac.Table.from_arrow(df).schema if _is_numeric_field(field)]
    columns = columns or numeric
    unknown = [name for name in columns if name not in df.columns]
    if unknown:
        raise ValueError(f"No columns named {unknown}. The columns are {df.columns}.")
    not_numeric = [name for name in columns if name not in numeric]
    if not_numeric:
        raise ValueError(f"Tour columns must be numeric; {not_numeric} are not.")
    if len(columns) < 2:
        raise ValueError(f"A tour needs at least 2 numeric columns; got {columns}.")

    notes = []
    n = df.height
    df = df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)).drop_nulls(columns)
    if df.height < n:
        notes.append(f"Dropped {n - df.height} of {n} rows with missing values.")
    incomplete = [name for name in numeric if name not in columns and df[name].null_count()]
    if incomplete:
        df = df.drop(incomplete)
        notes.append(f"Dropped columns with missing values: {incomplete}.")
    if sample is not None and df.height > sample:
        notes.append(f"Sampled {sample} of {df.height} rows.")
        df = df.sample(sample, seed=0)
    return df, columns, notes


def _compute_tour(X: np.ndarray, tour: Tour, names: list[str]) -> TourResult:
    # The viewer scales each column by its range, so the tour does too
    span = X.max(axis=0) - X.min(axis=0)
    X = X / np.where(span > 0, span, 1)
    if tour == "le":
        subsample = LE_SUBSAMPLE if len(X) > 2 * LE_SUBSAMPLE else None
        return le_tour(X, feature_names=names, random_state=0, subsample=subsample)
    if tour == "umap":
        return umap_little_tour(X, n_components=min(8, X.shape[1]))
    return little_tour(X)


def _tour_parquet(
    df: pl.DataFrame, columns: list[str], tour: Tour, settings: dict[str, Any]
) -> bytes:
    """The data with the tour and settings embedded, as Parquet."""
    import arro3.io
    import polars as pl

    X = df.select(pl.col(columns).cast(pl.Float32)).to_numpy()
    result = _compute_tour(X, tour, columns)
    ipc, dims = _viewer_data(_write_ipc(df), result, columns)
    table = add_spec_to_parquet(_read_table(ipc), tour=result, tour_dimensions=dims, **settings)
    out = BytesIO()
    arro3.io.write_parquet(table, out)
    return out.getvalue()


def _describe_selection(df: pl.DataFrame, selected: np.ndarray, columns: list[str]) -> str:
    """How the selected rows differ from the others, for the model to read."""
    import polars as pl

    n, k = len(df), int(selected.sum())
    lines = [f"{k} of {n} rows selected ({k / n:.1%})."]
    if k in (0, n):
        return lines[0]

    X = df.select(pl.col(columns).cast(pl.Float64)).to_numpy()
    inside, outside = X[selected].mean(axis=0), X[~selected].mean(axis=0)
    sd = X.std(axis=0)
    effect = (inside - outside) / np.where(sd > 0, sd, 1)
    lines.append("Largest differences from the other rows (mean; difference in SDs):")
    for i in np.argsort(-np.abs(effect))[:5]:
        lines.append(f"- {columns[i]}: {inside[i]:.4g} vs {outside[i]:.4g} ({effect[i]:+.1f} SD)")

    categories = [
        name
        for name, dtype in df.schema.items()
        if dtype in (pl.String, pl.Categorical) and df[name].n_unique() <= 50
    ]
    if categories:
        lines.append("Most common categories in the selection (share of selection; overall):")
    mask = pl.Series(selected)
    for name in categories:
        shares = df[name].filter(mask).value_counts(normalize=True, sort=True).head(3)
        overall = dict(df[name].value_counts(normalize=True).iter_rows())
        parts = [
            f"{label} {share:.0%} ({overall[label]:.0%})" for label, share in shares.iter_rows()
        ]
        lines.append(f"- {name}: {', '.join(parts)}")
    return "\n".join(lines)


VISUALIZE_DESCRIPTION = """\
Show a dataset as an interactive dtour: a tour through 2D projections of its \
numeric columns, with smooth transitions between keyframes. Use it for data with \
4+ numeric columns, to see clusters, outliers, and how groups relate.

path: a local CSV, TSV, Parquet, or Arrow file, or an http(s) URL to one.

tour:
- "pca" (default): consecutive pairs of principal components. Fast. Start here.
- "le": Laplacian Eigenmaps, coarse to fine. Finds nonlinear and cluster structure.
- "umap": PCA tour over an 8-D UMAP embedding. For many columns.
"le" and "umap" can take minutes beyond ~100k rows; then pass `sample` (e.g. \
50000) or warn the user first.

columns: the numeric columns to tour (default: all numeric columns). Exclude IDs \
and integer-coded labels; they aren't measurements. Columns are scaled to the \
same range. Rows missing a value in these columns are dropped.

settings: viewer settings (snake_case), e.g. {"point_color_by": "species"} to \
color by a column (categorical or numeric), {"point_color_by": ["x", "y"]} for \
a 2D colormap, "point_color_map" ({label: "#hex"}), "point_size" and \
"point_opacity" (number or "auto"), "tour_traversal" ("guided", "manual", \
"grand"), "tour_playing" (bool), "tour_speed" (0.1-5), "show_axes" (bool), \
"preview_label_content" ("auto", "description", "loadings"), "theme_mode" \
("light", "dark", "system"). Name only columns that exist.

Inspect the file's columns first if you don't know them.\
"""


def create_server() -> MCPServer:
    """The dtour MCP server. Starts its localhost HTTP server."""
    views = _Views()
    script = _viewer_script()
    port = _start_http_server(views, _viewer_html(script))
    origin = f"http://127.0.0.1:{port}"
    print(f"[dtour-mcp] serving data on {origin}", file=sys.stderr)

    apps = Apps()
    apps.add_html_resource(
        VIEWER_URI,
        _viewer_html(script),
        title="dtour viewer",
        csp=ResourceCsp(connect_domains=[origin]),
    )

    @apps.tool(resource_uri=VIEWER_URI, name="visualize", description=VISUALIZE_DESCRIPTION)
    def visualize(
        path: str,
        ctx: Context,
        tour: Tour = "pca",
        columns: list[str] | None = None,
        sample: int | None = None,
        settings: dict[str, Any] | None = None,
    ) -> CallToolResult:
        t0 = time.perf_counter()
        try:
            df, columns, notes = _prepare(_read(path), columns, sample)
            parquet = _tour_parquet(df, columns, tour, settings or {})
        except Exception as error:
            # The SDK hides other exceptions' messages, but the model needs them
            # to fix its call
            raise ToolError(f"{type(error).__name__}: {error}") from error
        token = views.add(_View(parquet, columns))
        seconds = time.perf_counter() - t0

        lines = [
            f"Showing a {tour} tour of {df.height} rows over {len(columns)} columns "
            f"({', '.join(columns)}), computed in {seconds:.1f}s.",
            *notes,
        ]
        if not client_supports_apps(ctx):
            lines.append(f"Open the viewer in a browser: {origin}/view/{token}")
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
            structured_content={"url": f"{origin}/data/{token}", "token": token},
        )

    @apps.tool(resource_uri=VIEWER_URI, visibility=["app"], name="read_data")
    def read_data(token: str, offset: int = 0) -> CallToolResult:
        """Read a chunk of the Parquet file for *token*, base64-encoded."""
        data = views.get(token).parquet
        chunk = data[offset : offset + CHUNK_SIZE]
        return CallToolResult(
            content=[TextContent(type="text", text=f"{len(chunk)} bytes")],
            structured_content={
                "base64": base64.b64encode(chunk).decode("ascii"),
                "size": len(data),
            },
        )

    @apps.tool(resource_uri=VIEWER_URI, visibility=["app"], name="describe_selection")
    def describe_selection(token: str, mask: str) -> str:
        """Compare the selected rows to the others.

        *mask* is base64-encoded little-endian uint32 words with one bit per row.
        """
        import polars as pl

        view = views.get(token)
        df = pl.read_parquet(BytesIO(view.parquet))
        bits = np.unpackbits(np.frombuffer(base64.b64decode(mask), np.uint8), bitorder="little")
        return _describe_selection(df, bits[: len(df)].astype(bool), view.columns)

    return MCPServer("dtour", extensions=[apps])


def main() -> None:
    """Run over stdio, or over HTTP with ``--http PORT`` (e.g., for ext-apps' basic-host)."""
    server = create_server()
    if "--http" not in sys.argv:
        server.run()
        return

    import uvicorn
    from starlette.middleware.cors import CORSMiddleware

    app = CORSMiddleware(
        server.streamable_http_app(),
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[sys.argv.index("--http") + 1]))


if __name__ == "__main__":
    main()
