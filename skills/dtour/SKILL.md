---
name: dtour
description: Explore high-dimensional data and embeddings through steerable tours (guided, manual, and grand) of 2D projections as scatter plots in Jupyter/marimo, React, or dtour.dev. Use when someone wants to look at data with more than two numeric dimensions beyond a single 2D scatter, check whether UMAP/t-SNE structure is real, compare embeddings across models, hyperparameters, or time points, or is working with the `dtour` Python package or `@dtour/viewer`.
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

For data that is already 2D, use a regular scatter plot (e.g.
[jupyter-scatter](https://jupyter-scatter.dev)) instead.

## Pick a surface

| Situation | Surface |
|---|---|
| No code, a Parquet/Arrow/CSV file at hand | https://dtour.dev: drop the file in, or open `https://dtour.dev/?url=<encoded public file URL>`. The file's host must allow CORS; if the start page stays up, that is the likely cause |
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
| Is my UMAP/t-SNE structure real? | `little_tour` over the PCA space UMAP was fit on, next to the 2D UMAP with linked selections | Structure present in UMAP but in no PCA keyframe was probably introduced by UMAP. Example: [dtour + jupyter-scatter](references/python.md#dtour--jupyter-scatter-validate-a-umap-against-a-pca-tour) |
| How do clusters form as repulsion increases? | `attraction_repulsion_tour(X)` | Moves from LE-like through UMAP-like to t-SNE. Needs `dtour[tsne]` |
| How do models, methods, or hyperparameters differ? | `sequential_tour([X1, X2, ...])` | Each keyframe is its own embedding (UMAP, t-SNE, PyMDE, or precomputed layouts), aligned to the previous one. Keyframes stay faithful, so differences show up |
| How does a gradual series evolve (e.g. time points)? | `aligned_umap_tour([X1, X2, ...])` | UMAP only. Optimizes all frames jointly and penalizes moving points, giving smoother motion. Strong alignment can hide real change |

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

# little_tour projects the input columns directly
tour = dtour.little_tour(df.select(features))

dtour.Widget(
    data=df.select([*features, "sex"]),  # tour columns first, labels after
    tour=tour,
    point_color_by="sex",
    color_map=dtour.build_color_map(sorted(df["sex"].unique())),
)
```

Every other tour computes its own embedding and projects `tour.embedding`, not the
input columns. Passing the tour alone is not enough: the data you pass must start with
the embedding columns.

```py
tour = dtour.le_tour(df.select(features), n_frames=8, random_state=42)
emb = pl.DataFrame({f"le_{i}": tour.embedding[:, i] for i in range(tour.n_dims)})
dtour.Widget(data=emb.with_columns(df["sex"]), tour=tour, point_color_by="sex")
```

## Data rules (most common mistakes)

1. **Numeric columns are dimensions; string columns are categories.** An integer label
   column becomes a tour dimension, so cast labels to string. To keep other numeric
   columns out of the tour, leave them out of `data` or uncheck them in the toolbar's
   column menu (auto-generated tours only).
2. **With a precomputed tour, the viewer projects the first `tour.n_dims` numeric
   columns, in order.** Put the tour columns first and any extra numeric columns after
   them.
3. **Keep the default `tour_by` when passing a tour.** `tour_by="pca"` replaces your
   tour with an in-browser PCA tour. Sequential tours switch to `"parameter"`
   automatically.
4. **Scale features with different units** (e.g. `StandardScaler`) before computing a
   tour. Otherwise one feature dominates every projection.
5. **Drop or impute nulls** in the tour columns.
6. **Computing a tour can be slow; rendering is not.** `le_tour` (kNN graph +
   eigensolver) and the UMAP/t-SNE-based tours take minutes on large data, and so do
   quality metrics. `little_tour` is fast. Cache results with `tour.save(path)` /
   `dtour.TourResult.load(path)`, and use `le_tour(subsample=...)` above ~100K rows.

## Viewer settings

The same settings exist on every surface: Python widget traitlets (snake_case), the
React `spec` prop and Parquet metadata (camelCase). The most useful ones:

| Python | React / Parquet | Values |
|---|---|---|
| `tour_traversal` | `tourTraversal` | `"guided"` \| `"manual"` \| `"grand"` |
| `tour_position`, `tour_playing`, `tour_speed` | `tourPosition`, `tourPlaying`, `tourSpeed` | 0–1, bool, 0.1–5 |
| `point_color_by`, `color_map` | `pointColorBy`, `pointColorMap` | column name, label → color |
| `point_size`, `point_opacity` | `pointSize`, `pointOpacity` | number or `"auto"` |
| `preview_count`, `preview_size` | `previewCount`, `previewScale` | 2–16; small/medium/large ↔ 0.5/0.75/1 |
| `camera_zoom`, `camera_pan_x/y` | `cameraZoom`, `cameraPanX/Y` | numbers |
| `theme` | `themeMode` | `"light"` \| `"dark"` \| `"system"` |

dtour.dev has no way to set these through the link yet. To share a configured view,
embed the settings in the file (see Sharing a result).

## Reading a tour

- Scrub slowly between keyframes. The motion of point groups is the signal: clusters
  that stay together in every view are probably real.
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
  an attraction–repulsion tour is likely a repulsion artifact. Boundary points that stay
  put across the spectrum are more trustworthy.

## Sharing a result

Embed the tour and the widget settings into the Parquet file's metadata with
`w.save_spec_to_parquet(table)` (details in [references/python.md](references/python.md)).
The file then opens with the same tour and settings on dtour.dev or in React. In
Python, `dtour.TourResult.from_parquet(path)` recovers the tour.

## References

Load these only when needed:

- [references/python.md](references/python.md): the full Widget API, tour function
  signatures, coloring, quality metrics, linking widgets (with a jupyter-scatter
  example), Parquet export
- [references/javascript.md](references/javascript.md): the `<Dtour>` React component,
  `DtourSpec`, the imperative handle, view matrix layout
- [references/paper.md](references/paper.md): the dtour paper. Covers the background
  (tour theory, the design of the steerability spectrum, interpolation math, the tour
  strategies) and the usage scenarios on Fashion-MNIST, single-cell, and arXiv data.
  Read it when explaining *why* dtour works the way it does, or when interpreting tours
  in depth.
