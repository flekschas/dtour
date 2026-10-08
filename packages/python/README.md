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
    point_color_by=None,  # column name for categorical coloring
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

Run the tests:

```sh
uv run --extra dev --extra cev pytest
```

Smoke-test the demo notebooks against your local working copy (from the repo root):

```sh
pnpm test:notebooks                 # all notebooks
pnpm test:notebooks demo_spectral   # a single notebook
```
