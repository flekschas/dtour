---
name: dtour
description: Visually explore high-dimensional data and embeddings through interactive tours (guided, manual, and grand) of 2D projections as scatter plots in Jupyter/marimo, React, dtour.dev, or chat apps via the dtour MCP server. Use when someone wants to look at data with more than two numeric dimensions beyond a single 2D scatter, check whether UMAP/t-SNE structure is real, compare embeddings across models, hyperparameters, or time points, or is working with the `dtour` Python package or `@dtour/viewer`.
---

# dtour

dtour shows high-dimensional data as a **tour**: a cyclic sequence of 2D projections
(*keyframes*) that users can smoothly scrub through. One central scatter is surrounded by
keyframe previews on a circular slider. Users can switch between three traversal modes at
any time:

- **guided**: scrub, scroll, or play along a precomputed keyframe path
- **manual**: drag a dimension's axis handle to steer the projection
- **grand**: an endless random walk through projection space

Lasso and label selection persist across projections, so a typical workflow is to select
points in one view and then follow them through the others. Rendering is smooth for up
to 10M points and still interactive up to 20M points.

## When to use it

Use dtour when no single 2D view can tell the story:

- the data has 4+ numeric dimensions (features, PCs, or a higher-dimensional embedding)
- you want to check whether clusters/structure in a UMAP/t-SNE layout are real or artifacts
- you want to compare several 2D embeddings of the same points, e.g. across models,
  hyperparameters, or time

Data that is already 2D has nothing to tour. dtour shows two numeric columns as a static
scatter, which is handy next to a tour with linked selections (`dtour.link`).

## Pick a surface

| Situation | Surface |
|---|---|
| The dtour MCP server's `visualize` tool is available, and a local file or URL at hand | Call `visualize(path, tour, columns, settings)`. The viewer opens in the chat, or the tool returns a browser link. Read the file's columns first, and leave IDs and integer-coded labels out of `columns` |
| No code, a Parquet/Arrow/CSV file at hand | https://dtour.dev: drop the file in, or open `https://dtour.dev/?url=<encoded public file URL>` with settings as further parameters (see [Link to a configured view](#link-to-a-configured-view)). The file's host must allow CORS; if the start page stays up, that is the likely cause |
| Python analysis in a Marimo or Jupyter notebook | `pip install dtour` → `dtour.Widget(...)`. See [references/python.md](references/python.md) |
| A React app | `npm install @dtour/viewer` → `<Dtour data={buffer} />`. See [references/javascript.md](references/javascript.md) |

Without a precomputed tour, the viewer generates one itself: dimension pairs by default,
or PCA with `tour_by="pca"` / `tourBy: "pca"`. Start there for a quick first look.

## Pick a tour (Python)

| Question | Tour | Notes |
|---|---|---|
| What does this table look like from different angles? | `little_tour(X)` | PCA, consecutive PC pairs. Like walking along the PC1–PC2, PC2–PC3, … panels of a scatter plot matrix, but with smooth transitions between them. Cheap; a good default |
| What nonlinear manifold or cluster structure is there? | `le_tour(X, n_frames=8)` | Laplacian Eigenmaps, coarse → fine. `subsample=` for >100K rows |
| Which directions separate my known labels? | `le_tour(X, labels=y, discriminative=True)` | Spectral Fisher. Keyframes are ordered by how well they separate classes |
| Too many dims for a linear tour? | `umap_little_tour(X, n_components=8)` | Needs `dtour[umap]` |
| Is my UMAP/t-SNE structure real? | `little_tour` over the PCA space UMAP was fit on, next to the 2D UMAP with linked selections | A group that separates in UMAP but overlaps in every PCA keyframe is worth checking: inspect its neighbors and distances in the full PCA space before calling it an artifact, since the tour shows only consecutive PC pairs. Example: [validate a UMAP against a PCA tour](references/python.md#validate-a-umap-against-a-pca-tour) |
| How do clusters form as repulsion increases? | `attraction_repulsion_tour(X)` | Moves from LE-like through UMAP-like to t-SNE. Needs `dtour[tsne]` |
| How do models, methods, or hyperparameters differ? | `sequential_tour([X1, X2, ...])` | Each keyframe is its own embedding (UMAP, t-SNE, PyMDE, or precomputed layouts), aligned to the previous one. Keyframes stay faithful, so differences show up |
| How does a gradual series evolve (e.g. time points)? | `aligned_umap_tour([X1, X2, ...])` | UMAP only. Optimizes all frames jointly and penalizes moving points, giving smoother motion. Strong alignment can hide real change |

Tours can have any number of keyframes. The gallery previews up to 32 of them, evenly
spaced along the tour (fewer when space is short), and the slider keeps a tick for every
keyframe.

The first four are **hyperdimensional** tours: every frame, including the in-between
ones, is a real projection of one high-dimensional space. The last three are
**sequential** tours: only the keyframes are real embeddings, and the in-between frames
are morphs between them. Never read structure from in-between frames of a sequential
tour.

## Minimal Python recipe

```py
import dtour
import polars as pl

df = pl.read_parquet("https://github.com/uwdata/mosaic/raw/main/data/athletes.parquet")
features = ["height", "weight", "gold", "silver", "bronze"]
df = df.drop_nulls(features)

tour = dtour.little_tour(df.select(features))

w = dtour.Widget(
    df,  # the data the tour was computed from, plus columns to color by
    tour,
    point_color_by="sex",
    color_map=dtour.build_color_map(sorted(df["sex"].unique())),
)
```

Pass the same kind of `data` for every tour. `little_tour` projects the input columns,
which the widget finds by name. All other tours compute their own embedding
(`tour.embedding`), and the widget adds its columns to the data itself:

```py
tour = dtour.le_tour(df.select(features), n_frames=8, random_state=42)
dtour.Widget(df, tour, point_color_by="sex")
```

This needs a dtour release newer than 0.4.4. With 0.4.4, put the tour's columns first in
`data`, and for embedding tours build that data from `tour.embedding` (see
[references/python.md](references/python.md#data-for-embedding-tours)).

## Data rules (most common mistakes)

1. **Numeric columns are dimensions; string columns are categories.** An integer label
   column becomes a tour dimension, so cast labels to string. A precomputed tour uses
   only its own columns. For an auto-generated tour, set `tour_dimensions` to the columns
   to use, or uncheck columns in the toolbar's column menu (the PCA tour always uses all
   numeric columns).
2. **Pass the data the tour was computed from.** Tours find their columns by name. Only
   for unnamed input, like numpy arrays, the tour uses the first `tour.n_dims` numeric
   columns.
3. **Keep the default `tour_by` when passing a tour.** `tour_by="pca"` replaces your
   tour with an in-browser PCA tour. Sequential tours switch to `"parameter"`
   automatically.
4. **Scale features with different units** (e.g. `StandardScaler`) before computing a
   tour. Otherwise one feature dominates every projection.
5. **Drop or impute missing values** in every numeric column you pass, not only the tour
   columns. A missing value in any numeric column can break the projection.
6. **Keep wide data out of `data`.** Every numeric column is loaded onto the GPU. For an
   embedding tour over thousands of features, pass only the columns to color by, e.g.
   `Widget(df.select("cell_type"), tour)`.
7. **Computing a tour can be slow; rendering is not.** `le_tour` (kNN graph +
   eigensolver) and the UMAP/t-SNE-based tours take minutes on large data, and so do
   quality metrics. `little_tour` is fast. Cache results with `tour.save(path)` /
   `dtour.TourResult.load(path)`, and use `le_tour(subsample=...)` above ~100K rows.

## Viewer settings

The same settings exist on every surface: Python widget traitlets and Parquet export
arguments (snake_case), and the React `spec` prop and Parquet metadata (camelCase). Every
setting has a traitlet. The most useful ones:

| Python | React / Parquet | Values |
|---|---|---|
| `tour_traversal` | `tourTraversal` | `"guided"` \| `"manual"` \| `"grand"` |
| `tour_position`, `tour_playing`, `tour_speed` | `tourPosition`, `tourPlaying`, `tourSpeed` | 0–1, bool, 0.1–5 |
| `point_color_by`, `color_map` (export: `point_color_map`) | `pointColorBy`, `pointColorMap` | column name or `[x, y]` pair for a 2D colormap, label → color |
| `point_color_map_2d` | `pointColorMap2d` | `"schumann"` \| `"bremm"` \| `"steiger"` \| `"ziegler"` \| `"teulingfig2"` \| `"cubediagonal"` \| `"oklab_polar"` |
| `tour_dimensions` | `tourDimensions` | columns of an auto-generated tour; two show a static scatter, the first on the x-axis |
| `link` | `link` | views in the same browser with the same id share their selection (`dtour.link()` sets it) |
| `point_size`, `point_opacity` | `pointSize`, `pointOpacity` | number or `"auto"` |
| `preview_count` | `previewCount` | 2–32, keyframes of an auto-generated tour |
| `preview_size` | `previewSize` | `"auto"` \| `"small"` \| `"medium"` \| `"large"` |
| `preview_label_content` | `previewLabelContent` | `"auto"` \| `"description"` \| `"loadings"` |
| `preview_label_visibility` | `previewLabelVisibility` | `"auto"` \| `"visible"` \| `"interactive"` (on hover) \| `"hidden"` |
| `show_axes` | `showAxes` | axis biplot in guided mode |
| `camera_zoom`, `camera_pan_x/y` | `cameraZoom`, `cameraPanX/Y` | numbers |
| `theme_mode` | `themeMode` | `"light"` \| `"dark"` \| `"system"` |

## Link to a configured view

dtour.dev reads every setting from URL parameters named like the React/Parquet fields, so you can hand someone a link that opens the data the way they
need it:

```
https://dtour.dev/?url=<encoded file URL>&pointColorBy=cell_type&tourTraversal=manual
```

- Load the data with `url=<public file URL>` or `dataset=<example>` (`gaussian-blobs`,
  `linked-rings`, `lorenz`, `fashion-mnist`, `news-headlines`, `single-cell`,
  `single-cell-rna-seq`, `image-caption-clip`, `arxiv-papers`).
- Write strings as plain text and everything else as JSON: `previewCount=8`,
  `showAxes=true`, `pointColorBy=["x","y"]`, `tourDimensions=["a","b","c"]`,
  `pointColorMap={"setosa":"#e69f00"}`. Write a string that reads as JSON, e.g. a
  column named `1`, as a JSON string: `pointColorBy="1"`.
- Percent-encode each value, e.g. with `encodeURIComponent` or
  `urllib.parse.urlencode`.
- Name only columns that exist in the file. Column names are case-sensitive, so read
  the file's schema first if you don't know them.
- An invalid value is ignored and logged to the browser console. Unknown parameters are
  ignored silently, so check the spelling against the settings table above.
- A link shows exactly its settings: those it sets override the file's embedded
  settings, including `null` (`pointColorBy=null` turns off coloring the file sets).
  Settings it omits use the file's embedded settings or the defaults, never what the
  user last used for that file. Without `themeMode`, the view is dark.

```py
from urllib.parse import urlencode
import json

params = {
    "url": "https://example.org/cells.parquet",
    "pointColorBy": json.dumps(["UMAP_1", "UMAP_2"]),
    "pointColorMap2d": "bremm",
    "tourTraversal": "manual",
}
print(f"https://dtour.dev/?{urlencode(params)}")
```

As the user changes settings, dtour.dev writes them to the URL, leaving out those equal
to the defaults or the file's embedded settings. Copying the address bar shares these
settings and the guided tour position, but not a projection the user dragged in manual
mode, the current grand tour projection, or the viridis/magma choice for numeric
coloring.

## Reading a tour

- Scrub slowly between keyframes. The motion of point groups is the signal. A tour
  shows a limited set of 2D views, so treat what you see as a lead to check, not proof:
  a group can be real yet overlap in every keyframe.
- **Select, then explore**: lasso a group in one keyframe, scrub to the others, then
  switch to manual mode and drag axes to find which dimensions separate it.
- Keyframe previews can show the top-loading features (`le_tour` fills these in).
  Use them to name what each view shows.
- **Color to track change.** Color by a numeric column for a continuous colormap, or by
  two columns for a 2D colormap (toolbar column menu → "2D"). In a sequential tour,
  apply the 2D colormap to one keyframe's x/y columns. Points keep their reference
  colors, so a group whose position changed stands out, e.g. bright red in a dark blue
  neighborhood. See [Coloring](references/python.md#coloring).
- A cluster that is tight in t-SNE-like keyframes but scatters toward the LE-like end of
  an attraction–repulsion tour may be a repulsion artifact; check its members (e.g. the
  raw items or their neighbors in the input space) before concluding so. Boundary points
  that stay put across the spectrum are more trustworthy.

## Sharing a result

Embed the tour and the widget settings into the Parquet file's metadata with
`w.save_spec_to_parquet()`, which saves the widget's data, including any embedding
columns (details in [references/python.md](references/python.md#saving-and-sharing)).
The file then opens with the same tour and settings on dtour.dev or in React. In
Python, `dtour.TourResult.from_parquet(path)` recovers the tour. Once the file is
public, a [dtour.dev link](#link-to-a-configured-view) can point at it with different
settings.

## References

Load these only when needed:

- [references/python.md](references/python.md): the full Widget API, tour function
  signatures, coloring, quality metrics, linking widgets (with a jupyter-scatter
  example), Parquet export
- [references/javascript.md](references/javascript.md): the `<Dtour>` React component,
  `DtourSpec`, the imperative handle, keyframe matrix layout
- [references/paper.md](references/paper.md): the dtour paper. Covers the background
  (tour theory, the design of the steerability spectrum, interpolation math, the tour
  strategies) and the usage scenarios on Fashion-MNIST, single-cell, and arXiv data.
  Read it when explaining *why* dtour works the way it does, or when interpreting tours
  in depth. When quoting its validation claims, add that a disagreement between views is
  a lead to check, not proof.
