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
    data=...,             # DataFrame, pyarrow Table, Arrow IPC bytes, or file path
    tour=...,             # TourResult from little_tour() / umap_little_tour()
    # display
    height=720,           # canvas height in pixels
    preview_count=4,      # keyframe previews: 2–16
    preview_size="large", # "small" | "medium" | "large"
    preview_padding=12.0, # gap between previews
    # point style
    point_size="auto",    # point radius or "auto"
    point_opacity="auto", # point alpha or "auto"
    point_color=[0.25, 0.5, 0.9],  # default RGB color
    point_color_by=None,  # column name for categorical coloring
    color_map={},         # label → color mapping (see build_color_map())
    # tour playback
    tour_by="dimensions", # "dimensions" | "pca" | "parameter"
    tour_position=0.0,    # 0–1 position along the tour
    tour_playing=False,   # auto-play on load
    tour_speed=1.0,       # playback speed multiplier
    tour_direction="forward",  # "forward" | "backward"
    tour_dimensions=[],   # explicit column names for the tour
    # camera
    camera_pan_x=0.0,
    camera_pan_y=0.0,
    camera_zoom=1/1.5,
    centering="midrange", # "midrange" | "mean"
    # mode & appearance
    tour_traversal="guided",   # "guided" | "manual" | "grand"
    show_legend=True,     # show/hide color legend
    show_keyframe_loadings=True,  # show/hide feature loadings
    theme="dark",         # "light" | "dark" | "system"
)
```

All settings are exposed as [traitlets](https://traitlets.readthedocs.io/), so they
can be read, set, and observed live from the notebook.

## Widget methods

```py
w = dtour.Widget(data=X, tour=tour)
w.set_data(df)                    # load new data
w.set_tour(tour)                  # set tour views
w.set_metrics(metrics)            # display radial quality charts
w.select([0, 1, 2])              # select points by index
w.clear_selection()               # clear selection
```

## Tour computation

dtour ships with two tour generators:

```py
# PCA-based: cycles through consecutive pairs of principal components
tour = dtour.little_tour(
    X,                    # (n_samples, n_features) array or DataFrame
    n_components=None,    # defaults to min(n_features, 10)
)

# UMAP + PCA: reduce to n_components with UMAP first (pip install dtour[umap])
tour = dtour.umap_little_tour(
    X,
    n_components=10,
    umap_kwargs=None,     # extra kwargs passed to umap.UMAP
)
```

Both return a `TourResult` with `.views` (list of p×2 float32 arrays), `.n_views`, `.n_dims`, `.explained_variance_ratio`, and `.save(path)` / `TourResult.load(path)` for persistence.

## Quality metrics

Compute per-view quality scores and display them as radial bar charts on the circular slider:

```py
metrics = dtour.compute_metrics(
    X,                    # (n_samples, n_features) float32
    views=tour.views,     # from TourResult
    labels=None,          # cluster/class labels for supervised metrics
    metrics=None,         # list of metric names; defaults to ["silhouette", "trustworthiness"]
    k=7,                  # neighbors for neighborhood-based metrics
    subsample=None,       # int, per-metric dict, or None for built-in defaults
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
    theme=None,                   # "light" | "dark" | None (theme-aware dicts)
    overrides=None,               # per-label color overrides
)
dtour.Widget(data=df, point_color_by="cluster", color_map=cmap)
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

Edit a notebook against the local source with all dev extras:

```sh
uv run --extra dev marimo edit notebooks/demo_immune_cell_markers.py
```

Run the tests:

```sh
uv run --extra dev --extra cev pytest
```

Smoke-test the demo notebooks against your local working copy (from the repo root):

```sh
pnpm test:notebooks                 # all notebooks
pnpm test:notebooks demo_spectral   # a single notebook
```
