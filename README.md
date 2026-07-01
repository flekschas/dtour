<p align="center">
  <img src="https://data.dtour.dev/dtour-logo-plate.svg" alt="dtour" width="240">
</p>
<p align="center">
  <em>take a detour from your usual 2D embedding visualization → <a href="https://dtour.dev" target="_blank">dtour.dev</a></em>
</p>

---

**dtour** is a visualization tool for exploring high-dimensional data through guided, manual, and grand tours.

- ⚡ **Fast**: built with WebGPU, Web Workers, and OffscreenCanvas to scale to millions of data points
- 🔄 **Flexible**: explore data with the web app, integrate the viewer in your own React app, or use as a Python widget for data analysis
- 🖱️ **Fingertrippy**: play and rewind tours, manipulate axes, or get hypnotized by an endless grand tour animation

<p align="center">
  <img src="https://data.dtour.dev/dtour-teaser-60fps.gif" alt="dtour teaser" width="480">
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

> [!TIP]
> See the [Python package](packages/python) for details and several [example notebooks](packages/python/notebooks).

**dtour** integrates with Jupyter and Marimo notebooks through [anywidget](https://github.com/manzt/anywidget).

```sh
pip install dtour
```

Load a dataset and instantiate the widget:

```py
import dtour
import polars as pl

df = pl.read_parquet("https://github.com/uwdata/mosaic/raw/main/data/athletes.parquet")

dtour.Widget(data=df)
```

That's the simplest hello-world example to get started and visualize a high-dimensional dataset. For something more interesting, compute a **PCA tour** over the numeric columns and color points by a category:

```py
import dtour
import polars as pl

df = pl.read_parquet(
    "https://github.com/uwdata/mosaic/raw/main/data/athletes.parquet"
).drop_nulls(["height", "weight"])

# A little tour cycles through consecutive pairs of principal components.
features = ["height", "weight", "gold", "silver", "bronze"]
tour = dtour.little_tour(df.select(features))

# Color points by a categorical column, with a matching legend color map.
dtour.Widget(
    data=df,
    tour=tour,
    point_color_by="sex",
    color_map=dtour.build_color_map(sorted(df["sex"].unique()), theme="dark"),
)
```

Swap `little_tour` for `umap_little_tour` (needs `pip install "dtour[umap]"`) to tour a UMAP embedding instead.

For more details see the [Python package](packages/python) and [example notebooks](packages/python/notebooks).

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
  onPointSelectionChange={fn} // fires when point selection changes (Uint32Array mask)
  onLoadData={fn}             // called when user loads a file via the toolbar
  onReady={fn}                // called with a DtourHandle for programmatic control
  hideToolbar={false}         // hide the top toolbar
  backend="webgpu"            // "webgpu" | "webgl"
  tourFamily="hyperdimensional" // "hyperdimensional" | "sequential"
  keyframeDescriptions={[...]} // per-keyframe descriptions or template string
  keyframeLoadings={[...]}    // per-keyframe feature loadings
/>
```

#### DtourSpec

All fields are optional. Omitted fields use defaults.

```ts
type DtourSpec = {
  tourTraversal?: "guided" | "manual" | "grand"; // default "guided"
  tourBy?: "dimensions" | "pca" | "parameter";   // default "dimensions"
  tourPosition?: number;              // 0–1, default 0
  tourPlaying?: boolean;              // default false
  tourSpeed?: number;                 // 0.1–5, default 1
  tourDirection?: "forward" | "backward";
  tourSliderSpacing?: "equal" | "geodesic"; // default "equal"
  tourSliderVisibility?: "visible" | "subtle" | "hidden";
  previewCount?: 2–16;               // default 4
  previewScale?: 1 | 0.75 | 0.5;     // default 1
  previewPadding?: number;            // default 12
  pointSize?: number | "auto";        // default "auto"
  pointOpacity?: number | "auto";     // 0–1, default "auto"
  minPointSize?: number;              // 1–20, default 2
  pointColor?: [number, number, number]; // default [0.25, 0.5, 0.9]
  pointColorBy?: string | null;       // column name for categorical coloring
  pointColorMap?: Record<string, string>; // label → hex color
  cameraPanX?: number;                // default 0
  cameraPanY?: number;                // default 0
  cameraZoom?: number;                // default 1/1.5
  centering?: "midrange" | "mean";    // default "midrange"
  showLegend?: boolean;               // default true
  showAxes?: boolean;                 // default false
  showKeyframeNumbers?: boolean;      // default false
  showKeyframeLoadings?: boolean;     // default true
  showTourDescription?: boolean | null; // default null
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

## Paper

For details about the design of dtour, the tour interpolation math, and usage scenarios spanning text, image, and single-cell data, see our preprint at https://arxiv.org/abs/2605.04306.

```bibtex
@misc{lekschas2026dtour,
  author        = {Fritz Lekschas and Nezar Abdennur},
  title         = {dtour: a steerable tour de vis through high-dimensional data},
  year          = {2026},
  eprint        = {2605.04306},
  archivePrefix = {arXiv},
  primaryClass  = {cs.HC},
  doi           = {10.48550/arXiv.2605.04306},
  url           = {https://arxiv.org/abs/2605.04306}
}
```
