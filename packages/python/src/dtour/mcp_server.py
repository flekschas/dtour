"""MCP server that shows dtour tours inline in chat apps like Claude Desktop.

The model calls ``visualize`` with a data file. The server computes one or more
views of it, each a tour or a static 2D scatter, and stores each view as a
Parquet file with the data, tour, and settings. The viewer, an MCP App in the
host's iframe, shows the views side by side with linked selections. It fetches
the files from a localhost HTTP server, or, if the host blocks localhost, in
chunks through the app-only ``read_data`` tool. Hosts without MCP Apps get a
link to the same viewer in the browser.

Run with ``dtour-mcp`` (stdio). Logs go to stderr because stdout carries the protocol.
"""

from __future__ import annotations

import base64
import hashlib
import json
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
from pydantic import ConfigDict, with_config
from typing_extensions import TypedDict

from .data import _add_embedding, _is_numeric_field, _read_table, _write_ipc
from .spec import add_spec_to_parquet
from .tours import TourResult, _import_umap, le_tour, little_tour, umap_little_tour
from .widget import _viewer_data

if TYPE_CHECKING:
    import polars as pl

# Bytes per `read_data` call. Claude Desktop drops the connection on messages
# over 32 MiB, and base64 adds a third.
CHUNK_SIZE = 8 * 1024 * 1024

# Embedding tours subsample larger data for the spectral step
LE_SUBSAMPLE = 50_000

# More views don't fit side by side
MAX_VIEWS = 4

Tour = Literal["pca", "le", "umap"]

# Markers of missing values in CSV files, e.g., `NA` from R
NULL_VALUES = ["", "NA", "N/A", "NaN", "nan", "null", "NULL"]


# A misspelled field fails instead of being ignored
@with_config(ConfigDict(extra="forbid"))
class View(TypedDict, total=False):
    """One view of a ``visualize`` call. Omitted fields take the call's value."""

    tour: Tour
    columns: list[str]
    dimensions: int
    settings: dict[str, Any]


class _Visualization(NamedTuple):
    """The views of a ``visualize`` call, each its data with the tour and settings
    embedded as Parquet, with a title for each, and the columns the views use."""

    views: list[bytes]
    titles: list[str]
    columns: list[str]


class _Visualizations:
    """The most recent visualizations, by token. Older ones are forgotten."""

    def __init__(self, capacity: int = 16) -> None:
        self._items: OrderedDict[str, _Visualization] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, visualization: _Visualization) -> str:
        token = secrets.token_urlsafe(16)
        with self._lock:
            self._items[token] = visualization
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return token

    def get(self, token: str) -> _Visualization:
        with self._lock:
            visualization = self._items.get(token)
        if visualization is None:
            raise ToolError("This visualization's data is gone. Ask for it again.")
        return visualization


def _manifest(origin: str, token: str, visualization: _Visualization) -> dict[str, Any]:
    """What the viewer needs to fetch and describe the views of *token*."""
    return {
        "token": token,
        "views": [
            {"url": f"{origin}/data/{token}/{i}", "title": title}
            for i, title in enumerate(visualization.titles)
        ],
    }


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


def _start_http_server(visualizations: _Visualizations, html: str) -> int:
    """Serve the data files and a standalone viewer on a free localhost port.

    ``/data/<token>/<i>`` is the Parquet file of view *i*, ``/data/<token>`` the
    manifest listing them, and ``/view/<token>`` the viewer showing them.
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
            kind, _, rest = self.path.lstrip("/").partition("/")
            token, _, view = rest.partition("/")
            try:
                visualization = visualizations.get(token)
            except ToolError:
                visualization = None
            if visualization is None:
                self._send(404)
            elif kind == "view" and not view:
                self._send(200, html.encode(), "text/html; charset=utf-8")
            elif kind == "data" and not view:
                origin = f"http://127.0.0.1:{self.server.server_address[1]}"
                manifest = _manifest(origin, token, visualization)
                self._send(200, json.dumps(manifest).encode(), "application/json")
            elif kind == "data" and view.isdigit() and int(view) < len(visualization.views):
                self._send(200, visualization.views[int(view)], "application/vnd.apache.parquet")
            else:
                self._send(404)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def _read(path: str) -> pl.DataFrame:
    """Read a CSV, Parquet, or Arrow file from a local path or URL."""
    import polars as pl

    source: bytes | Path
    if "://" in path:
        # Some hosts reject the default Python user agent
        request = urllib.request.Request(path, headers={"User-Agent": "dtour"})
        with urllib.request.urlopen(request) as response:
            source = response.read()
    else:
        source = Path(path).expanduser()
    suffix = Path(path.split("?")[0]).suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        separator = "\t" if suffix == ".tsv" else ","
        return pl.read_csv(
            source, separator=separator, infer_schema_length=10_000, null_values=NULL_VALUES
        )
    if suffix in (".arrow", ".feather", ".ipc"):
        # `_read_table` reads both the Arrow IPC file and stream formats
        data = source.read_bytes() if isinstance(source, Path) else source
        return pl.DataFrame(_read_table(data))
    return pl.read_parquet(source)


def _prepare(
    df: pl.DataFrame, views: list[list[str] | None], sample: int | None
) -> tuple[pl.DataFrame, list[list[str]], list[str]]:
    """The data to show, the columns of each view, and notes about what changed.

    A view without columns uses all numeric columns. Numeric columns with missing
    values break the projection, so rows missing a value in any view's columns
    are dropped, and other numeric columns with missing values too. All views
    share the resulting rows, so selections link by row.
    """
    import arro3.core as ac
    import polars as pl

    # The viewer reads neither Decimal nor boolean values as numbers
    df = df.with_columns(pl.col(pl.Decimal).cast(pl.Float64), pl.col(pl.Boolean).cast(pl.UInt8))
    numeric = [field.name for field in ac.Table.from_arrow(df).schema if _is_numeric_field(field)]
    views = [columns or numeric for columns in views]
    for columns in views:
        unknown = [name for name in columns if name not in df.columns]
        if unknown:
            raise ValueError(f"No columns named {unknown}. The columns are {df.columns}.")
        not_numeric = [name for name in columns if name not in numeric]
        if not_numeric:
            raise ValueError(f"Tour columns must be numeric; {not_numeric} are not.")
        if len(columns) < 2:
            raise ValueError(f"A view needs at least 2 numeric columns; got {columns}.")
    used = list(dict.fromkeys(name for columns in views for name in columns))

    notes = []
    n = df.height
    df = df.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)).drop_nulls(used)
    if df.height < n:
        notes.append(f"Dropped {n - df.height} of {n} rows with missing values.")
    incomplete = [name for name in numeric if name not in used and df[name].null_count()]
    if incomplete:
        df = df.drop(incomplete)
        notes.append(f"Dropped columns with missing values: {incomplete}.")
    if sample is not None and df.height > sample:
        notes.append(f"Sampled {sample} of {df.height} rows.")
        df = df.sample(sample, seed=0)
    return df, views, notes


def _scaled(df: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """The *columns* of *df*, each divided by its range as the viewer does."""
    import polars as pl

    X = df.select(pl.col(columns).cast(pl.Float32)).to_numpy()
    span = X.max(axis=0) - X.min(axis=0)
    return X / np.where(span > 0, span, 1)


def _compute_tour(
    X: np.ndarray, tour: Tour, names: list[str], dimensions: int | None
) -> TourResult:
    if tour == "le":
        subsample = LE_SUBSAMPLE if len(X) > 2 * LE_SUBSAMPLE else None
        kwargs = {} if dimensions is None else {"n_components": dimensions}
        return le_tour(X, feature_names=names, random_state=0, subsample=subsample, **kwargs)
    if tour == "umap":
        return umap_little_tour(X, n_components=dimensions or min(8, X.shape[1]))
    return little_tour(X, n_components=dimensions)


def _embed_2d(X: np.ndarray, tour: Tour) -> tuple[np.ndarray, list[str]]:
    """A 2D embedding of *X* and names for its columns."""
    if tour == "umap":
        return _import_umap().UMAP(n_components=2).fit_transform(X), ["UMAP1", "UMAP2"]
    if tour == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2).fit_transform(X), ["PC1", "PC2"]
    raise ValueError(f'A 2D embedding needs tour "pca" or "umap", not "{tour}".')


def _view_parquet(
    df: pl.DataFrame,
    columns: list[str],
    tour: Tour,
    dimensions: int | None,
    settings: dict[str, Any],
) -> tuple[bytes, str]:
    """The view's data with its tour and settings embedded, as Parquet, and its title."""
    import arro3.core as ac
    import arro3.io

    shown = columns if len(columns) <= 8 else [*columns[:6], "…"]
    listed = f"{len(columns)} columns ({', '.join(shown)})"
    if dimensions is not None and dimensions < 2:
        raise ValueError(f"dimensions must be at least 2; got {dimensions}.")
    if len(columns) != 2 and dimensions != 2:
        result = _compute_tour(_scaled(df, columns), tour, columns, dimensions)
        ipc, dims = _viewer_data(_write_ipc(df), result, columns)
        title = f"a {tour} tour over {listed}"
        if result.embedding is not None:
            title = f"a {tour} tour over a {result.n_dims}D embedding of {listed}"
        table = add_spec_to_parquet(_read_table(ipc), tour=result, tour_dimensions=dims, **settings)
    else:
        # The viewer shows exactly two tour dimensions as a static scatter
        table = ac.Table.from_arrow(df)
        if len(columns) == 2:
            names = columns
            title = f"a static scatter of {columns[0]} (x) and {columns[1]} (y)"
        else:
            embedding, names = _embed_2d(_scaled(df, columns), tour)
            # Unique names, as `_add_embedding` generates them
            for i, name in enumerate(names):
                while name in table.column_names:
                    name = f"_{name}"
                names[i] = name
            table, names = _add_embedding(table, embedding.astype(np.float32), names)
            title = f"a static 2D {tour} embedding of {listed}"
        # A PCA tour ignores the tour dimensions, which would turn the scatter into a tour
        settings = {**settings, "tour_by": "dimensions"}
        table = add_spec_to_parquet(table, tour_dimensions=names, **settings)
    out = BytesIO()
    arro3.io.write_parquet(table, out)
    return out.getvalue(), title


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
4+ numeric columns, to see clusters, outliers, and how groups relate. One call can \
also show up to 4 views side by side with linked selections, e.g. a tour next to \
a static 2D UMAP (see views); prefer that over separate calls or a custom chart.

path: a local CSV, TSV, Parquet, or Arrow file, or an http(s) URL to one.

tour:
- "pca" (default): consecutive pairs of principal components. Fast. Start here.
- "le": Laplacian Eigenmaps, coarse to fine. Finds nonlinear and cluster structure.
- "umap": PCA tour over an 8-D UMAP embedding. For many columns. With \
`dimensions: 2`, a plain static 2D UMAP.
"le" and "umap" can take minutes beyond ~100k rows; then pass `sample` (e.g. \
50000) or warn the user first.

columns: the numeric columns to tour (default: all numeric columns). Exclude IDs \
and integer-coded labels; they aren't measurements. Columns are scaled to the \
same range. Rows missing a value in these columns are dropped. Exactly two \
columns show as a static scatter, the first on x, e.g. precomputed UMAP columns.

dimensions: how many dimensions the tour moves through: principal components \
for "pca", eigenvectors for "le", embedding dimensions for "umap". 2 shows a \
static 2D embedding instead of a tour ("pca" or "umap" only).

views: up to 4 views of the same rows side by side, e.g. to compare a tour \
with a 2D embedding, two tours, or embeddings from two models stored as \
separate columns. Each view is an object with any of tour, columns, \
dimensions, and settings; omitted fields take the call's values, and a view's \
settings add to the call's. Selections link across views. Two views fit best. \
Their legends start hidden unless settings set "show_legend".

settings: viewer settings (snake_case), e.g. {"point_color_by": "species"} to \
color by a column (categorical or numeric), {"point_color_by": ["x", "y"]} for \
a 2D colormap, "point_color_map" ({label: "#hex"}), "point_size" and \
"point_opacity" (number or "auto"), "tour_traversal" ("guided", "manual", \
"grand"), "tour_playing" (bool), "tour_speed" (0.1-5), "show_axes" (bool), \
"preview_label_content" ("auto", "description", "loadings"), "theme_mode" \
("light", "dark", "system"), "tooltip_columns" (the columns the hover tooltip \
lists, in order; [] hides it; columns of image URLs or data URIs show as \
images), "legend_order" (labels the legend lists first). Name only columns \
that exist.

Inspect the file's columns first if you don't know them.\
"""


def create_server() -> MCPServer:
    """The dtour MCP server. Starts its localhost HTTP server."""
    visualizations = _Visualizations()
    html = _viewer_html(_viewer_script())
    port = _start_http_server(visualizations, html)
    origin = f"http://127.0.0.1:{port}"
    print(f"[dtour-mcp] serving data on {origin}", file=sys.stderr)

    # Hosts cache the viewer by URI, so a new viewer or port, which its CSP
    # names, gets a new URI
    digest = hashlib.sha256(f"{origin}\n{html}".encode()).hexdigest()[:16]
    viewer_uri = f"ui://dtour/viewer-{digest}.html"
    apps = Apps()
    apps.add_html_resource(
        viewer_uri,
        html,
        title="dtour viewer",
        csp=ResourceCsp(connect_domains=[origin]),
    )

    @apps.tool(resource_uri=viewer_uri, name="visualize", description=VISUALIZE_DESCRIPTION)
    def visualize(
        path: str,
        ctx: Context,
        tour: Tour = "pca",
        columns: list[str] | None = None,
        dimensions: int | None = None,
        sample: int | None = None,
        settings: dict[str, Any] | None = None,
        views: list[View] | None = None,
    ) -> CallToolResult:
        t0 = time.perf_counter()
        views = views or [View()]
        try:
            if len(views) > MAX_VIEWS:
                raise ValueError(f"Show at most {MAX_VIEWS} views; got {len(views)}.")
            df, view_columns, notes = _prepare(
                _read(path), [view.get("columns", columns) for view in views], sample
            )
            built = [
                _view_parquet(
                    df,
                    view_columns[i],
                    view.get("tour", tour),
                    view.get("dimensions", dimensions),
                    # Views side by side are narrow enough that the legend covers
                    # their points, so it starts hidden
                    {
                        **({"show_legend": False} if len(views) > 1 else {}),
                        **(settings or {}),
                        **view.get("settings", {}),
                    },
                )
                for i, view in enumerate(views)
            ]
        except Exception as error:
            # The SDK hides other exceptions' messages, but the model needs them
            # to fix its call
            raise ToolError(f"{type(error).__name__}: {error}") from error
        titles = [title for _, title in built]
        used = list(dict.fromkeys(name for names in view_columns for name in names))
        visualization = _Visualization([parquet for parquet, _ in built], titles, used)
        token = visualizations.add(visualization)
        seconds = time.perf_counter() - t0

        if len(titles) == 1:
            lines = [f"Showing {titles[0]} for {df.height} rows, computed in {seconds:.1f}s."]
        else:
            lines = [
                f"Showing {len(titles)} views of the same {df.height} rows side by side, "
                f"computed in {seconds:.1f}s. Selections in one view show in the others.",
                *(f"- View {i + 1}: {title}" for i, title in enumerate(titles)),
            ]
        lines.extend(notes)
        if not client_supports_apps(ctx):
            lines.append(f"Open the viewer in a browser: {origin}/view/{token}")
        return CallToolResult(
            content=[TextContent(type="text", text="\n".join(lines))],
            structured_content=_manifest(origin, token, visualization),
        )

    @apps.tool(resource_uri=viewer_uri, visibility=["app"], name="read_data")
    def read_data(token: str, view: int = 0, offset: int = 0) -> CallToolResult:
        """Read a chunk of the Parquet file of *view* for *token*, base64-encoded."""
        views = visualizations.get(token).views
        if not 0 <= view < len(views):
            raise ToolError(f"There is no view {view}; there are {len(views)}.")
        data = views[view]
        chunk = data[offset : offset + CHUNK_SIZE]
        return CallToolResult(
            content=[TextContent(type="text", text=f"{len(chunk)} bytes")],
            structured_content={
                "base64": base64.b64encode(chunk).decode("ascii"),
                "size": len(data),
            },
        )

    @apps.tool(resource_uri=viewer_uri, visibility=["app"], name="describe_selection")
    def describe_selection(token: str, mask: str) -> str:
        """Compare the selected rows to the others.

        *mask* is base64-encoded little-endian uint32 words with one bit per row.
        """
        import polars as pl

        visualization = visualizations.get(token)
        # All views have the same rows
        df = pl.read_parquet(BytesIO(visualization.views[0]))
        bits = np.unpackbits(np.frombuffer(base64.b64decode(mask), np.uint8), bitorder="little")
        return _describe_selection(df, bits[: len(df)].astype(bool), visualization.columns)

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
