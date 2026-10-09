# dtour: Python

This is the dtour Python package: a data-generic [anywidget](https://github.com/manzt/anywidget) that drops into [Jupyter](https://jupyter.org) and [marimo](https://marimo.io) notebooks.

## Install

```sh
pip install dtour
```

Optional extras enable additional tour generators and metrics:

```sh
pip install "dtour[umap]"   # umap_little_tour() + UMAP-based tours (umap-learn, numba)
pip install "dtour[tsne]"   # attraction–repulsion tours (openTSNE)
pip install "dtour[pymde]"  # PyMDE-based tours
pip install "dtour[cev]"    # confusion metric (cev-metrics)
pip install "dtour[mcp]"    # the MCP server (see below)
```

## Quick start

> [!TIP]
> Take a look at our [example notebooks](notebooks) for complete, runnable examples
> on real text, image, and single-cell datasets.

Load a dataset and instantiate the widget:

```py
import dtour
import polars as pl

df = pl.read_parquet("https://github.com/uwdata/mosaic/raw/main/data/athletes.parquet")

dtour.Widget(data=df)
```

## Widget API

```py
dtour.Widget(
    data=...,  # the data the tour was computed from: DataFrame, Arrow table, numpy array, or file path
    tour=...,  # TourResult from any tour generator
    # display
    height=720,  # canvas height in pixels
    preview_count=4,  # keyframe previews: 2–32
    preview_size="auto",  # "auto" | "small" | "medium" | "large"
    preview_padding=12.0,  # gap between previews
    preview_keyframe_numbers="auto",  # "auto" | "visible" | "hidden"
    preview_label_content="auto",  # "auto" | "description" | "loadings"
    preview_label_visibility="auto",  # "auto" | "visible" | "interactive" | "hidden"
    # point style
    point_size="auto",  # point radius or "auto"
    point_opacity="auto",  # point alpha or "auto"
    min_point_size=2.0,  # smallest automatic point size in px: 1–20
    point_color=[0.25, 0.5, 0.9],  # default RGB color
    point_color_by=None,  # column name, or [x, y] numeric columns for a 2D colormap
    point_color_map_2d="schumann",  # 2D colormap for a column pair
    color_map={},  # label → color mapping (see build_color_map())
    # tour playback
    tour_by="dimensions",  # "dimensions" | "pca" | "parameter"
    tour_position=0.0,  # 0–1 position along the tour
    tour_playing=False,  # auto-play on load
    tour_speed=1.0,  # playback speed multiplier
    tour_direction="forward",  # "forward" | "backward"
    tour_slider_spacing="equal",  # "equal" | "geodesic"
    tour_slider_visibility="visible",  # "visible" | "subtle" | "hidden"
    tour_dimensions=[],  # columns in the tour (set by the tour; without one, the columns checked in the toolbar)
    # camera
    camera_pan_x=0.0,
    camera_pan_y=0.0,
    camera_zoom=1 / 1.5,
    centering="midrange",  # "midrange" | "mean"
    # mode & appearance
    tour_traversal="guided",  # "guided" | "manual" | "grand"
    show_legend=True,  # show/hide color legend
    show_axes=False,  # show/hide the axis biplot in guided mode
    theme_mode="dark",  # "light" | "dark" | "system"
)
```

All settings are exposed as [traitlets](https://traitlets.readthedocs.io/), so they
can be read, set, and observed live from the notebook.

## Widget methods

```py
w = dtour.Widget(data=X, tour=tour)
w.set_data(df)  # load new data
w.set_data(df, tour)  # load new data with its tour
w.set_tour(tour)  # set tour keyframes
w.set_metrics(metrics)  # display radial quality charts
w.select([0, 1, 2])  # select points by index
w.clear_selection()  # clear selection
```

## Tour computation

dtour ships with two families of tour generators. **Hyperdimensional** tours show one high-dimensional space from different angles:

```py
# PCA: cycles through consecutive pairs of principal components
tour = dtour.little_tour(
    X,  # (n_samples, n_features) array or DataFrame
    n_components=None,  # defaults to min(n_features, 10)
)

# UMAP to n_components, then a little tour over the embedding (pip install dtour[umap])
tour = dtour.umap_little_tour(X, n_components=10, umap_kwargs=None)

# Laplacian Eigenmaps: each keyframe adds one more eigenvector (global → local)
tour = dtour.le_tour(
    X,
    n_frames=8,  # computes n_frames + 1 eigenvectors
    n_neighbors=15,
    labels=None,  # class labels: same-label edges attract, cross-label edges repel
    discriminative=False,  # with labels: order eigenvectors by class separation (spectral Fisher)
    subsample=None,  # fit on a subsample and extend to the rest (large data)
)
```

**Sequential** tours morph between aligned 2D embeddings of the same points, e.g. across time points, hyperparameters, or models:

```py
# One embedding per dataset, each warm-started from the previous one
tour = dtour.sequential_tour([X_1, X_2, X_3], method="umap")  # "umap" | "tsne" | "pymde" | callable

# Sweep from attraction (LE-like) to repulsion (t-SNE) (pip install dtour[tsne])
tour = dtour.attraction_repulsion_tour(X, n_frames=4)

# Optimize all embeddings jointly with UMAP's AlignedUMAP (pip install dtour[umap])
tour = dtour.aligned_umap_tour([X_1, X_2, X_3])
```

All generators return a `TourResult` with `.keyframes` (list of p×2 float32 arrays), `.n_keyframes`, `.n_dims`, `.tour_family`, and `.save(path)` / `TourResult.load(path)` for persistence. `little_tour` projects the input columns. All other tours project their own `.embedding`, which the widget adds to the data. Either way, pass the data you computed the tour from, plus any columns to color by:

```py
tour = dtour.le_tour(df.select(features), n_frames=8)
dtour.Widget(df, tour, point_color_by="cell_type")
```

## Quality metrics

Compute per-keyframe quality scores and display them as radial bar charts on the circular slider:

```py
metrics = dtour.compute_metrics(
    X,  # (n_samples, n_features) float32
    keyframes=tour.keyframes,  # from TourResult
    labels=None,  # cluster/class labels for supervised metrics
    metrics=None,  # list of metric names; defaults to ["silhouette", "trustworthiness"]
    k=7,  # neighbors for neighborhood-based metrics
    subsample=None,  # int, per-metric dict, or None for built-in defaults
    exclude_labels=None,  # label values to exclude from label-based metrics
)

w = dtour.Widget(data=X, tour=tour)
w.set_metrics(metrics)
```

Supported metrics: `silhouette`, `trustworthiness`, `calinski_harabasz`, `neighborhood_hit`, `confusion` (require `labels`), `hdbscan_score` (unsupervised). `confusion` needs the optional `cev` extra (`pip install dtour[cev]`).

## Color maps

Build a label → color mapping that matches the engine's auto-assignment:

```py
cmap = dtour.build_color_map(
    labels=sorted_unique_labels,  # same order the engine sees
    theme=None,  # "light" | "dark" | None (theme-aware dicts)
    overrides=None,  # per-label color overrides
)
dtour.Widget(data=df, point_color_by="cluster", color_map=cmap)
```

## MCP server

`dtour-mcp` is an [MCP](https://modelcontextprotocol.io) server that lets an AI assistant
show tours right in the chat. Ask, e.g., _"Show me a UMAP tour of ~/data/cells.csv
colored by cell_type"_. The server reads the file, computes the tour, and the viewer
appears inline as an [MCP App](https://github.com/modelcontextprotocol/ext-apps).

It has one tool, `visualize`, which takes:

- `path`: a CSV, TSV, Parquet, or Arrow file, or an http(s) URL to one
- `tour`: `"pca"` (default), `"le"` (Laplacian Eigenmaps), or `"umap"`
- `columns`: the numeric columns to tour (default: all)
- `sample`: a number of rows to sample, for large data
- `settings`: viewer settings, as in `build_dtour_metadata`, e.g. `{"point_color_by": "cell_type"}`

Rows with missing values in the tour columns are dropped, and the columns are scaled to
the same range. In apps without MCP Apps support, like Claude Code, the tool returns a
link that opens the viewer in the browser. The viewer follows the app's light or dark
theme unless `settings` sets `theme_mode`.

As you explore, the viewer tells the assistant what you see: the traversal mode, the
color column, selected legend labels, and for selected points, how their column means
and categories differ from the other rows. So you can ask, e.g., _"What's special about
the points I selected?"_

What leaves your computer: the server reads files and computes tours locally, and the
viewer loads the data from a server on `127.0.0.1`. If the app blocks that, the viewer
receives the file through the app instead. The assistant sees the tool's arguments and
results, like file paths and column names, and the summaries above. A summary of a few
selected points shows their values; for a single point, it shows that row's values.

In Claude Desktop, download
[`dtour.mcpb`](https://github.com/flekschas/dtour/releases/latest/download/dtour.mcpb)
and open it to install the extension. Claude Desktop sets up Python and the dependencies
itself.

In Claude Code, install the plugin, which also includes the dtour skill:
`/plugin install dtour --marketplace flekschas/dtour`. It needs
[uv](https://docs.astral.sh/uv/).

Other MCP clients can run the server with uv, e.g.:

```json
{
  "mcpServers": {
    "dtour": {
      "command": "uvx",
      "args": ["--from", "dtour[mcp,umap]", "dtour-mcp"]
    }
  }
}
```

## Example notebooks

The [`notebooks/`](notebooks) directory has self-contained [marimo](https://marimo.io)
demos on real datasets (Fashion-MNIST, a developing-brain scRNA-seq atlas, a
ShareGPT4V × COCO image embedding, and immune-cell CyTOF markers). Each notebook
declares its own dependencies and downloads its data on first run, so you can open
one straight from a checkout:

```sh
uvx marimo edit --sandbox notebooks/demo_spectral.py
```

See the [notebooks README](notebooks/README.md) for a description of each and more
ways to run them.

## Development

`dtour` loads the widget's JavaScript bundle from `src/dtour/static/widget.js`, which
is not checked in. Build it, together with the `@dtour/scatter` and `@dtour/viewer`
packages it bundles, from the repo root:

```sh
pnpm install
pnpm build:widget
```

Rebuild after changing any frontend code. Importing `dtour` warns when the bundle is
older than the frontend sources.

Edit a notebook against the local source with all dev extras:

```sh
uv run --extra dev marimo edit notebooks/demo_immune_cell_markers.py
```

Run the MCP server from the local source, e.g., in Claude Desktop's config with
`"command": "uv"` and `"args": ["run", "--directory", "<repo>/packages/python",
"--extra", "mcp", "--extra", "umap", "dtour-mcp"]`. To try it in a browser, run
`uv run --extra mcp dtour-mcp --http 3001` and connect ext-apps'
[basic-host](https://github.com/modelcontextprotocol/ext-apps/tree/main/examples/basic-host)
to `http://localhost:3001/mcp`.

Build the Claude Desktop extension with `pnpm build:mcpb` from the repo root. It installs
the dtour release named in `mcpb/pyproject.toml` from PyPI; with `pnpm build:mcpb --local`,
it includes a wheel of your checkout instead. Open
`mcpb/dist/dtour.mcpb` to install it.

Run the tests:

```sh
uv run --extra dev --extra cev pytest
```

Smoke-test the demo notebooks against your local working copy (from the repo root):

```sh
pnpm test:notebooks                 # all notebooks
pnpm test:notebooks demo_spectral   # a single notebook
```
