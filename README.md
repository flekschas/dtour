<p align="center">
  <img src="https://storage.googleapis.com/dtour/dtour-logo-plate.svg" alt="dtour" width="240">
</p>
<p align="center">
  <em>take a detour from your usual 2D embedding visualization → <a href="https://dtour.dev" target="_blank">dtour.dev</a></em>
</p>
<p align="center">
  <em>Still work in progress but feel free to play with it.</em>
</p>

---

**dtour** is a visualization tool for exploring high-dimensional data through guided, manual, and grand tours.

- ⚡ **Fast**: built with WebGPU, Web Workers, and OffscreenCanvas to scale to millions of data points
- 🔄 **Flexible**: explore data with the web app, integrate the viewer in your own React app, or use as a Python widget for data analysis
- 🖱️ **Fingertrippy**: play and rewind tours, manipulate axes, or get hypnotized by an endless grand tour animation

<p align="center">
  <img src="https://storage.googleapis.com/dtour/dtour-teaser-60fps.gif" alt="dtour teaser" width="480">
</p>

A single 2D projection can only capture a fraction of high-dimensional structure. That's not a flaw of the embedding, it's a constraint of two axes. dtour lets you fly through multiple projections so you can build a sense for the full space.

## Web

Go to https://dtour.dev and drop a Parquet or Arrow file into the app. That's it 🚀

#### Modes

dtour has three viewing modes:

- **Guided** — play through a precomputed tour of optimized 2D projections. Use the circular slider to scrub through keyframes, or hit play and watch the data rotate between views.
- **Manual** — drag individual axes to build your own projection from scratch. Good for hypothesis-driven exploration when you know which dimensions matter.
- **Grand** — sit back and watch an infinite random tour through projection space. Useful for serendipitous discovery (or as a screensaver).

## Python

**dtour** integrates with Jupyter and Marimo notebooks through [anywidget](https://github.com/manzt/anywidget).

#### Install

```sh
pip install dtour
```

#### Quick start

Load a dataset and instantiate the widget:

```py
import dtour
import polars as pl

df = pl.read_parquet("https://github.com/uwdata/mosaic/raw/main/data/athletes.parquet")

dtour.Widget(data=df)
```

#### Widget API

```py
dtour.Widget(
    data=...,             # DataFrame, pyarrow Table, Arrow IPC bytes, or file path
    tour=...,             # TourResult from little_tour() / little_umap_tour()
    # display
    height=720,           # canvas height in pixels
    preview_count=4,      # keyframe previews: 4 | 8 | 12 | 16
    preview_size="large", # "small" | "medium" | "large"
    preview_padding=12.0, # gap between previews
    # point style
    point_size="auto",    # point radius or "auto"
    point_opacity="auto", # point alpha or "auto"
    point_color=[0.25, 0.5, 0.9],  # RGB list or column name for categorical coloring
    color_map={},         # label → color mapping (see build_color_map())
    # tour playback
    tour_by="dimensions", # "dimensions" | "pca"
    tour_position=0.0,    # 0–1 position along the tour
    tour_playing=False,   # auto-play on load
    tour_speed=1.0,       # playback speed multiplier
    tour_direction="forward",  # "forward" | "backward"
    # camera
    camera_pan_x=0.0,
    camera_pan_y=0.0,
    camera_zoom=1.0,
    # mode & appearance
    view_mode="guided",   # "guided" | "manual" | "grand"
    show_legend=True,     # show/hide color legend
    theme="dark",         # "light" | "dark" | "system"
)
```

#### Widget methods

```py
w = dtour.Widget(data=X, tour=tour)
w.set_metrics(metrics)            # display radial quality charts
w.select([0, 1, 2])              # select points by index
w.clear_selection()               # clear selection
```

#### Tour computation

dtour ships with two tour generators:

```py
# PCA-based: cycles through consecutive pairs of principal components
tour = dtour.little_tour(
    X,                    # (n_samples, n_features) array or DataFrame
    n_components=None,    # defaults to min(n_features, 10)
)

# UMAP + PCA: reduce to n_components with UMAP first (pip install dtour[umap])
tour = dtour.little_umap_tour(
    X,
    n_components=10,
    umap_kwargs=None,     # extra kwargs passed to umap.UMAP
)
```

Both return a `TourResult` with `.views` (list of p×2 float32 arrays), `.n_views`, `.n_dims`, `.explained_variance_ratio`, and `.save(path)` / `TourResult.load(path)` for persistence.

#### Quality metrics

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

Supported metrics: `silhouette`, `trustworthiness`, `calinski_harabasz`, `neighborhood_hit`, `confusion` (require `labels`), `hdbscan_score` (unsupervised).

#### Color maps

Build a label → color mapping that matches the engine's auto-assignment:

```py
cmap = dtour.build_color_map(
    labels=sorted_unique_labels,  # same order the engine sees
    theme=None,                   # "light" | "dark" | None (theme-aware dicts)
    overrides=None,               # per-label color overrides
)
dtour.Widget(data=df, point_color="cluster", color_map=cmap)
```

## JavaScript

**dtour** is also published as a ready-to-use React component.

#### Install

```sh
npm install dtour
```

#### Quick start

```tsx
import { Dtour } from "dtour";

<Dtour data={arrowBuffer} />
```

#### Component API

```tsx
<Dtour
  data={arrowBuffer}          // Arrow IPC or Parquet ArrayBuffer
  views={views}               // Float32Array[] of p×2 column-major view matrices
  metrics={metricsBuffer}     // Arrow IPC ArrayBuffer with per-view quality metrics
  metricTracks={tracks}       // RadialTrackConfig[] for radial bar chart customization
  metricBarWidth="full"       // "full" | number — global bar width for radial charts
  colorMap={colorMap}         // Record<string, string | {light, dark}> per-label colors
  spec={spec}                 // partial DtourSpec to control component state
  onSpecChange={handleSpec}   // fires on state change (debounced ~250ms)
  onStatus={handleStatus}     // called on every renderer status event
  onSelectionChange={fn}      // fires when legend selection changes (label names)
  onLoadData={fn}             // called when user loads a file via the toolbar
  onReady={fn}                // called with a DtourHandle for programmatic control
  hideToolbar={false}         // hide the top toolbar
/>
```

#### DtourSpec

All fields are optional. Omitted fields use defaults.

```ts
type DtourSpec = {
  tourBy?: "dimensions" | "pca";      // default "dimensions"
  tourPosition?: number;              // 0–1, default 0
  tourPlaying?: boolean;              // default false
  tourSpeed?: number;                 // 0.1–5, default 1
  tourDirection?: "forward" | "backward";
  previewCount?: 4 | 8 | 12 | 16;    // default 4
  previewScale?: 1 | 0.75 | 0.5;     // default 1
  previewPadding?: number;            // default 12
  pointSize?: number | "auto";        // default "auto"
  pointOpacity?: number | "auto";     // 0–1, default "auto"
  pointColor?: [number, number, number] | string;  // RGB or column name
  cameraPanX?: number;                // default 0
  cameraPanY?: number;                // default 0
  cameraZoom?: number;                // default 1/1.5
  viewMode?: "guided" | "manual" | "grand";
  showLegend?: boolean;               // default true
  themeMode?: "light" | "dark" | "system"; // default "dark"
};
```

## Why Take a _Tour de Vis_ Through High-Dimensional Data?

Making sense of high-dimensional data is hard. Non-linear embedding tools like
UMAP do a great job at representing the high-dimensional manifold as best as
possible in a 2D space. However, by virtue of only having two dimensions for
laying out a scatter of points, distortions are introduced. One can have endless
debates about the stability and correctness of clusters that emerge in such 2D
embedding visualization.

The obvious solution to this problem is to take detour from your usual static 2D
projection and examine different perspectives: i.e., a _tour_ of different
projections/views. Such tours can be visualized statically as scatter plot
matrices when the number of views is limited or as an animation like the
[grand tour](https://en.wikipedia.org/wiki/Grand_Tour_(data_visualisation)),
which randomly rotates the high-dimensional space.

The idea of **dtour** is to take a middle ground and offer a highly interactive
interface for deterministically transitioning through a handful of
curated views that reveal more than a static 2D projection and are less
overwhelming than a grand tour. In other terms, dtour, wants to tighten the
exploration of high-dimensional data by replacing the random wandering of a
grand tour animation with deterministic, optimized navigation through, what's
called _guided tours_.
