# dtour Python reference

`pip install dtour` installs an [anywidget](https://anywidget.dev) that works in Jupyter
and marimo. Optional extras:

| Extra | Enables |
|---|---|
| `dtour[umap]` | `umap_little_tour`, `aligned_umap_tour`, `sequential_tour(method="umap")` |
| `dtour[tsne]` | `attraction_repulsion_tour` (default engine), `sequential_tour(method="tsne")` |
| `dtour[pymde]` | `method="pymde"` in sequential / attraction–repulsion tours |
| `dtour[cev]` | the `confusion` quality metric |

## Contents

- [Widget](#widget)
- [Tours](#tours)
- [Quality metrics](#quality-metrics)
- [Coloring](#coloring)
- [Linking widgets](#linking-widgets)
- [Saving and sharing](#saving-and-sharing)
- [Recipes from the example notebooks](#recipes-from-the-example-notebooks)

## Widget

```py
w = dtour.Widget(
    data,                     # the data the tour was computed from, plus columns to color by:
                              # polars/pandas DataFrame, pyarrow/arro3 Table, numpy 2D array,
                              # Arrow IPC bytes, or a path to an Arrow/Parquet file
    tour,                     # TourResult; omit for an auto-generated tour
    height=720,
    preview_count=4,          # 2–32, keyframes of an auto-generated tour
    preview_size="auto",      # "auto" | "small" | "medium" | "large"
    preview_padding=12.0,
    preview_label_content="auto",     # "auto" | "description" | "loadings"
    preview_label_visibility="auto",  # "auto" | "visible" | "interactive" (on hover) | "hidden"
    preview_keyframe_numbers="auto",  # "auto" | "visible" | "hidden"
    point_size="auto",        # float or "auto"
    point_opacity="auto",     # 0–1 or "auto"
    min_point_size=2.0,       # 1–20
    point_color=[0.25, 0.5, 0.9],
    point_color_by=None,      # column name: string → categorical, numeric → continuous
    color_map={},             # label → color, see build_color_map()
    tour_by="dimensions",     # "dimensions" | "pca" | "parameter"
    tour_traversal="guided",  # "guided" | "manual" | "grand"
    tour_position=0.0,        # 0–1 along the cyclic tour
    tour_playing=False,
    tour_speed=1.0,
    tour_direction="forward",
    tour_slider_spacing="equal",       # "equal" | "geodesic" (segment width encodes projection distance)
    tour_slider_visibility="visible",  # "visible" | "subtle" | "hidden"
    camera_pan_x=0.0, camera_pan_y=0.0, camera_zoom=1/1.5,
    centering="midrange",     # "midrange" | "mean"
    show_legend=True,
    show_axes=False,          # axis biplot in guided mode
    show_tour_description=False,
    theme_mode="dark",        # "light" | "dark" | "system"
    metric_tracks=[],         # radial metric chart config, see Quality metrics
    metric_bar_width="full",  # "full" or int
)
```

Every viewer setting is a synced [traitlet](https://traitlets.readthedocs.io/en/stable/): read
it, set it (`w.tour_playing = True`), or `observe` it from the notebook. Changes made in
the UI reach Python debounced (~250 ms).

Methods:

```py
w.set_data(df)                 # replace data
w.set_tour(tour)               # replace tour
w.set_metrics(metric_result)   # radial quality charts on the slider
w.select([0, 5, 9])            # select by row index
w.select_by_labels(["B cell"]) # select by labels of the active color column
w.clear_selection()
w.save_spec_to_parquet()       # see Saving and sharing
w.tour_family                  # "hyperdimensional" | "sequential" | None
```

Selection state is synced in both directions: `w.selected_indices` and
`w.selected_labels`.

### How data maps onto a tour

- Numeric columns (float or int) become dimensions. String/dictionary columns become
  categories that you can color by. Cast integer labels to string.
- Without a tour, the viewer builds its own from all numeric columns. `tour_by`
  chooses dimension pairs (`"dimensions"`) or in-browser PCA (`"pca"`).
- A precomputed tour uses only its own columns; other numeric columns (e.g. raw marker
  values) can stay in `data` for coloring. For an auto-generated tour, set
  `tour_dimensions` to the columns to use, or uncheck columns in the toolbar's column
  menu (the PCA tour always uses all numeric columns).
- Every numeric column is loaded onto the GPU and must not contain missing values. For
  wide data, pass only the columns you need; an embedding tour needs no input columns
  at all (`Widget(df.select("cell_type"), tour)`).
- Pass the data the tour was computed from. A tour finds its columns by name: `little_tour`
  records the DataFrame's numeric column names in `tour.feature_names`, and the widget
  adds the `tour.embedding` columns of all other tours itself, named after
  `tour.embedding_names`. `w.tour_dimensions` shows the columns the tour projects. For
  unnamed input, like numpy arrays, the tour uses the first `tour.n_dims` numeric
  columns. Rows are matched by position, so keep them in the order the tour was
  computed with; for sequential tours, pass one table with one row per point.
- The widget keeps a snapshot of `data`; call `set_data` to show changes.
- A tour can have any number of keyframes. The gallery previews up to 32 of them, evenly
  spaced along the tour, and fewer when space is short. `preview_count` only applies
  to auto-generated tours.
- `tour_by="pca"` overrides a passed tour. Leave `tour_by` alone when passing one. For
  sequential tours, `set_tour` switches it to `"parameter"` automatically.

## Tours

Every tour function returns a `TourResult`:

| Field | Meaning |
|---|---|
| `keyframes` | list of `(p, 2)` float32 orthonormal bases, one per keyframe |
| `n_keyframes`, `n_dims` | number of keyframes, p |
| `embedding`, `embedding_names` | `(n, p)` matrix the bases project and its column names (e.g. `LE1`, `UMAP1`, `frame1_x`), or `None` when they project the input columns (`little_tour`) |
| `feature_names` | input feature names. For `little_tour`, the columns the bases project |
| `feature_loadings`, `feature_r2` | correlations between tour dims and input features (`le_tour`). Drive the loading labels under the previews |
| `explained_variance_ratio` | PCA tours |
| `tour_family` | `"hyperdimensional"` or `"sequential"` |
| `description`, `keyframe_descriptions` | text shown in the description bar and per keyframe |

Build one from your own bases with `dtour.TourResult(keyframes, n_dims=p)` (the other
fields are keyword-only). Persist with `tour.save("t.npz")` / `dtour.TourResult.load("t.npz")`, or read a tour
embedded in Parquet with `dtour.TourResult.from_parquet(path_or_table)`.

All tour functions accept numpy arrays, pandas/polars DataFrames, or pyarrow Tables
with numeric columns.

### Hyperdimensional tours

```py
dtour.little_tour(X, n_components=None)
```
PCA, then consecutive component pairs, wrapping around: [PC1,PC2] → [PC2,PC3] → … →
[PCk,PC1]. `n_components` defaults to `min(n_features, 10)`. The bases live in the
**input** space (`embedding is None`) and project the input columns by name.

```py
dtour.umap_little_tour(X, n_components=10, umap_kwargs=None)   # dtour[umap]
```
UMAP to `n_components` dims, then a little tour over that embedding.

```py
dtour.le_tour(
    X, n_components=8, n_neighbors=15, feature_names=None, random_state=None,
    subsample=None,      # fit the spectral embedding on a subsample, extend the rest via kNN (>100K rows)
    n_frames=None,       # number of frames; computes n_frames + 1 eigenvectors (don't set n_components too)
    n_remove=0,          # progressively drop low-frequency eigenvectors after build-up (local detail)
    labels=None,         # class-aware signed Laplacian: same-label edges attract, cross-label repel
    discriminative=False,# with labels: spectral Fisher, eigenvectors ordered by class separation
    alpha=1.0,           # repulsion strength of the signed Laplacian
    affinity="symmetric_knn",  # or "mutual_knn" (sparser)
    adaptive_sigma=False, normalize_alpha=None, se_kwargs=None,
)
```
A Laplacian Eigenmaps tour. Each frame adds one more eigenvector through a fixed
circular basis, so the tour moves from global to local structure. It fills in loadings
against `feature_names` (taken from DataFrame columns when X is one).

### Sequential tours

These build one 2D embedding per frame, Procrustes-align the frames, and stack them into
`tour.embedding` with shape `(n, 2K)`. Rows must be the same entities, in the same order,
in every frame.

```py
dtour.sequential_tour(
    [X_t0, X_t1, X_t2],          # one dataset per frame, same row count
    method="umap",               # "umap" | "tsne" | "pymde" | callable(data, prev, **kw) -> (n, 2)
    method_kwargs=None,
    steps=None,                  # list[dtour.EmbeddingStep(method=..., kwargs=...)] per-frame overrides
    init=None, feature_names=None, keyframe_descriptions=None, description=None,
    random_state=None,
)
```
Each frame is warm-started from the previous one. To tour precomputed 2D layouts
(e.g. one UMAP per embedding model), pass the layouts as the data and an identity
callable: `dtour.sequential_tour(layouts, method=lambda data, prev, **kw: data)`.

```py
dtour.attraction_repulsion_tour(
    X, n_frames=4, rhos=None, n_neighbors=15, init="le",
    method="tsne",               # or "pymde" (with regularization=0.5–5 to reduce jitter)
    regularization=0.0, feature_names=None, random_state=None,
)
```
Sweeps the exaggeration ρ from high (pure attraction, LE-like) to low (t-SNE). Based on
Böhm, Berens & Kobak, JMLR 2022.

```py
dtour.aligned_umap_tour([X_t0, X_t1], relations=None, umap_kwargs=None, ...)   # dtour[umap]
```
Optimizes all frames jointly with UMAP's `AlignedUMAP` instead of warm-starting them.
`umap_kwargs` takes `alignment_regularisation` (how strongly points are held in place
across frames) and `alignment_window_size`.

**`sequential_tour` or `aligned_umap_tour`?**

| | `sequential_tour` | `aligned_umap_tour` |
|---|---|---|
| How frames relate | Each frame is its own embedding, warm-started from the previous one, then rotated/scaled onto it (Procrustes) | All frames are optimized together, with a penalty for moving corresponding points between adjacent frames |
| Each keyframe is… | a faithful standalone embedding | a compromise between fitting its own data and matching its neighbors |
| Methods | UMAP, t-SNE, PyMDE, any callable, precomputed layouts | UMAP only |
| Use for | comparing models, methods, or hyperparameters, where differences are the point | smooth series of closely related slices (time points, gradual parameter sweeps), where stable motion matters more than per-frame fidelity |

The default is `sequential_tour`. `aligned_umap_tour` can hide real changes when the
regularization is strong.

### Data for embedding tours

Every tour except `little_tour` projects `tour.embedding`, not the input columns. The
widget puts the embedding columns first and keeps the columns of `data` for coloring,
tooltips, and labels:

```py
df = pl.read_parquet("cells.parquet")          # marker columns + a "cell_type" string column
markers = [c for c in df.columns if c != "cell_type"]

tour = dtour.le_tour(df.select(markers), n_frames=8, random_state=42)
w = dtour.Widget(df, tour, point_color_by="cell_type")   # sends LE1…LE9, the markers, cell_type
```

`dtour.Widget(tour=tour)` alone shows just the embedding.

dtour 0.4.4 and older don't add the embedding. There, build the data from it, with the
embedding columns first:

```py
emb = pl.DataFrame({f"le_{i}": tour.embedding[:, i] for i in range(tour.n_dims)})
w = dtour.Widget(data=emb.with_columns(df["cell_type"]), tour=tour, point_color_by="cell_type")
```

Newer versions still accept such data, with a warning.

## Quality metrics

```py
m = dtour.compute_metrics(
    X,                 # the matrix the keyframes project: tour.embedding, or the input columns for little_tour
    tour.keyframes,
    labels=None,       # needed for silhouette, calinski_harabasz, neighborhood_hit, confusion
    metrics=None,      # default ["silhouette", "trustworthiness"]
    k=7, subsample=None, exclude_labels=None,
)
w.set_metrics(m)
```
Available metrics: `silhouette`, `trustworthiness`, `calinski_harabasz`,
`neighborhood_hit`, `confusion` (needs `dtour[cev]`), and `hdbscan_score`. They are
drawn as radial bars per keyframe. Configure the bars with
`metric_tracks=[{"metric": "confusion", "height": 64, "domain": [0, 1]}]`. Metrics can
be slow; cache `m.values` and rebuild with
`dtour.MetricResult(values=..., metric_names=[...])`.

## Coloring

`point_color_by` takes any column:

- **String column**: categorical colors with a legend. Clicking legend entries selects
  those labels.
- **Numeric column**: a continuous colormap.
- **Two numeric columns**: a 2D colormap, where position in a 2D color square encodes
  both values. It is currently set in the toolbar's column menu ("2D" toggle, then
  pick two columns), not from Python. Entering 2D mode preselects the first two numeric
  columns.

A 2D colormap on one keyframe's coordinates is a strong tool for **sequential tours**.
Color by the x/y columns of one frame (e.g. `frame1_x` and `frame1_y`; entering 2D mode
preselects the first frame's pair), then scrub to the others. Points keep the color of where they sat
in the reference frame, so a group whose structure changed stands out, e.g. bright red
points in a dark blue neighborhood. The paper uses this to find a cluster of physics
education papers that one embedding model pulls together and the others spread out.

### Color maps

```py
cmap = dtour.build_color_map(
    sorted(labels.unique()),   # same order the engine uses (alphabetical)
    theme="dark",              # "light" | "dark" | None (returns {"light": .., "dark": ..} per label)
    overrides={"Unassigned": "#888888"},
)
dtour.Widget(data=df, point_color_by="cell_type", color_map=cmap)
```
It assigns Okabe–Ito colors first, then Glasbey, matching the engine's own
auto-assignment. Use one map across several widgets so their colors agree.

## Linking widgets

Widget state is [traitlets](https://traitlets.readthedocs.io/en/stable/), so the usual
tools apply:

- `w.observe(fn, names="selected_indices")` runs `fn(change)` whenever the trait changes
  (`change.new` holds the new value).
- `traitlets.link((w1, "point_color_by"), (w2, "point_color_by"))` keeps the same trait
  equal on two widgets. `traitlets.dlink` does the same in one direction, with an
  optional transform.

Selection is the trait you will link most often. `selected_indices` holds lasso or
programmatic selections. `selected_labels` holds legend selections (and clears
`selected_indices`).

### dtour + jupyter-scatter: validate a UMAP against a PCA tour

Tour the PCA space that UMAP was computed from, next to the 2D UMAP, and sync selections
both ways. Cells that cluster in UMAP but scatter in every PCA keyframe point to
structure that UMAP introduced (`pip install jupyter-scatter`):

```py
import dtour
import jscatter
import pandas as pd

# df: PCA coordinates (pc_cols) + a string "cell_type" column; umap_2d: (n, 2) UMAP of df[pc_cols]
cmap = dtour.build_color_map(sorted(df["cell_type"].unique()), theme="dark")

tour = dtour.little_tour(df[pc_cols])
w = dtour.Widget(df, tour, point_color_by="cell_type", color_map=cmap)

umap_df = pd.DataFrame({"x": umap_2d[:, 0], "y": umap_2d[:, 1], "cell_type": df["cell_type"]})
s = jscatter.Scatter(data=umap_df, x="x", y="y", color_by="cell_type", color_map=cmap)

def dtour_to_umap(change):
    if set(change.new) != set(s.selection()):   # skip echoes to avoid a feedback loop
        s.selection(change.new or None)

def umap_to_dtour(change):
    rows = list(change.new)
    if set(rows) != set(w.selected_indices):
        w.select(rows)

w.observe(dtour_to_umap, names="selected_indices")
s.widget.observe(umap_to_dtour, names="selection")

# Show side by side: ipywidgets.HBox([w, s.show()]) in Jupyter, mo.hstack([w, s.widget]) in marimo
```

To also mirror legend selections, observe `selected_labels` and map labels to row
indices (`df.index[df["cell_type"].isin(labels)]`). `demo_brain_atlas.py` in the repo is
the complete version.

Why two tools? dtour is built for tours. For a single 2D embedding, jupyter-scatter has
more 2D features (tooltips, axes, size and opacity encodings), so the two
complement each other.

## Saving and sharing

Embed the widget's current settings and the tour you set from Python in Parquet
metadata. dtour.dev and the React viewer restore them when they open the file. In Python,
`dtour.TourResult.from_parquet(path)` recovers the tour and `dtour.read_spec_from_parquet(path)`
recovers the settings. Saving replaces any existing dtour metadata: a widget opened from
an annotated file without a Python tour, e.g. `Widget("tour.pq")`, saves no tour, and the
columns checked for an auto-generated tour aren't saved either. To keep a file's tour,
pass it explicitly: `Widget(path, dtour.TourResult.from_parquet(path))`.

```py
import pyarrow as pa, pyarrow.parquet as pq

annotated = w.save_spec_to_parquet()   # the widget's data, incl. embedding columns, as an arro3 Table
pq.write_table(pa.table(annotated), "tour.pq", compression="zstd")
```

Or build the metadata without a widget. The table must contain the columns the tour
projects, so this suits `little_tour`, whose columns are in your data:

```py
meta = dtour.build_dtour_metadata(
    tour=tour, point_color_by="label", point_color_map=cmap,
    preview_count=8, camera_zoom=0.5, theme_mode="light",
)
df.write_parquet("tour.pq", metadata={"dtour": meta})    # polars
```

`tour_dimensions` in the metadata names the columns the tour projects. It defaults to
`tour.feature_names` for `little_tour`. For embedding tours, `build_dtour_metadata`
needs it explicitly, because `tour.feature_names` holds the *input* features there.

## Recipes from the example notebooks

The repo's `packages/python/notebooks/` directory has self-contained marimo notebooks.
Run one with `uvx marimo edit --sandbox <file>.py`.

- **UMAP validation** (`demo_brain_atlas.py`): `little_tour` over PC1–PC8 of a
  single-cell atlas, next to a jupyter-scatter 2D UMAP of the same PCs. Lasso in dtour
  highlights the same cells in the UMAP.
- **Spectral structure** (`demo_spectral.py`): `le_tour` on standardized CyTOF markers.
  It includes vanilla, signed (`labels=`), and Fisher (`discriminative=True`) variants,
  plus a heatmap of eigenvector ↔ marker correlations from `tour.feature_loadings`.
- **UMAP tour + metrics** (`demo_immune_cell_markers.py`): `umap_little_tour` to 8D,
  per-keyframe `confusion` metric bars, and label selection.
- **Attraction–repulsion** (`demo_attraction_repulsion.py`): Fashion-MNIST reduced to
  50D with PCA, then `attraction_repulsion_tour(n_frames=4, init="le")`. Selected points
  show their images in a side panel.
- **Precomputed tour from Parquet** (`demo_image_embedding.py`):
  `TourResult.from_parquet(table)` plus `Widget(data=table, tour=tour)`.

Patterns the notebooks share: standardize the features, cache tours with
`save`/`load`, and use `point_opacity=0.5` for dense data.
