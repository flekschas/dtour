# dtour JavaScript reference

dtour's frontend is two packages:

- `@dtour/viewer`: the React component with the full tour UI. Use this one.
- `@dtour/scatter`: the framework-agnostic WebGPU/WebGL2 rendering engine, for when you
  need only the renderer (see its README in the dtour repo).

## Install and minimal use

```sh
npm install @dtour/viewer   # react and react-dom >= 18 are peer dependencies
```

```tsx
import { Dtour } from "@dtour/viewer";
import "@dtour/viewer/dist/viewer.css";

const buffer = await fetch(url).then((r) => r.arrayBuffer());
<Dtour data={buffer} />;
```

`data` is an Arrow IPC or Parquet `ArrayBuffer`. **Ownership is transferred to a
worker**, so the buffer is detached afterwards. Pass a copy (`buffer.slice(0)`) if you
need it again. Parquet files with dtour metadata (written by the Python
`save_spec_to_parquet` / `build_dtour_metadata`) restore their tour and settings
automatically.

Numeric columns become dimensions; string columns become categories that you can color
by. Without `keyframes` or an embedded tour, the viewer generates a tour from the numeric
columns: dimension pairs by default, or PCA with `spec.tourBy = "pca"`.

## Props

```tsx
<Dtour
  data={buffer}
  keyframes={keyframes}          // Float32Array[], one p×2 basis per keyframe (see below)
  tourDimensions={names}         // columns the keyframes project; without keyframes, the
                                 //   columns checked in the toolbar's column menu
  tourFamily="hyperdimensional"  // or "sequential" for stacked 2D embeddings
  tourDescription={null}         // text for the description bar
  keyframeDescriptions={[...]}   // string[] or a template using {primary} {secondary} {relation}
  keyframeLoadings={[...]}       // [{ primary: [name, r], secondary: [name, r] }, ...]
  colorMap={colorMap}            // label → "#hex" or { light, dark }
  metrics={metricsIpc}           // Arrow IPC buffer with per-keyframe quality metrics
  metricTracks={tracks}          // radial bar chart config
  metricBarWidth="full"
  spec={spec}                    // partial DtourSpec, see below
  onSpecChange={(spec) => {}}    // full resolved spec, debounced (~250 ms)
  onSelectionChange={(labels) => {}}       // legend/label selection
  onPointSelectionChange={(mask) => {}}    // lasso selection, bit-packed Uint32Array
  onReady={(handle) => {}}       // DtourHandle for programmatic control
  onStatus={(status) => {}}      // renderer status events
  onLoadData={(data, fileName) => {}}      // user loaded a file from the toolbar
  onLogoClick={() => {}}
  hideToolbar={false}
  backend="auto"                 // "auto" | "webgpu" | "webgl"
  portalContainer={el}           // portal popups here, for Shadow DOM hosts
/>
```

### Keyframe matrices

Each keyframe is a `Float32Array` of length `2p`, laid out column-major as
`[x_0 … x_{p-1}, y_0 … y_{p-1}]`, with orthonormal x and y columns. The keyframes project the
numeric columns named in `tourDimensions`, or the **first p numeric columns** of `data`
when it is omitted. If they don't name one existing column per keyframe dimension, the
viewer logs an error and shows an auto-generated tour instead. For a sequential tour, pass
`tourFamily="sequential"`, with columns `[f0_x, f0_y, f1_x, f1_y, …]` and keyframe `k`
selecting the pair for frame `k`.

### Lasso selection mask

Point `i` is selected when `(mask[i >> 5] >>> (i & 31)) & 1` is `1`.

### DtourHandle

```ts
handle.select([0, 5, 9]);                         // indices
handle.select(mask, { isBitPacked: true });       // bit-packed Uint32Array
handle.selectByLabels(["B cell"]);                // labels of the active color column
handle.clearSelection();
```

## DtourSpec

All fields are optional. `DTOUR_DEFAULTS` holds the defaults, and `dtourSpecSchema` is
the Zod schema.

```ts
{
  tourTraversal: "guided" | "manual" | "grand",     // "guided"
  tourBy: "dimensions" | "pca" | "parameter",       // "dimensions"
  tourPosition: number,          // 0–1
  tourPlaying: boolean,
  tourSpeed: number,             // 0.1–5
  tourDirection: "forward" | "backward",
  tourSliderSpacing: "equal" | "geodesic",   // geodesic: segment width encodes projection distance
  tourSliderVisibility: "visible" | "subtle" | "hidden",
  previewCount: 2..32,           // 4, keyframes of an auto-generated tour
  previewSize: "auto" | "small" | "medium" | "large",
  previewPadding: number,
  previewKeyframeNumbers: "auto" | "visible" | "hidden",     // auto: when some keyframes have no preview
  previewLabelContent: "auto" | "description" | "loadings",  // auto: loadings when available
  previewLabelVisibility: "auto" | "visible" | "interactive" | "hidden", // auto: visible up to 16 previews, on hover above
  pointSize: number | "auto",
  pointOpacity: number | "auto",
  minPointSize: number,          // 1–20
  pointColor: [r, g, b],         // 0–1
  pointColorBy: string | null,   // column name
  pointColorMap: Record<string, string>,
  cameraPanX: number, cameraPanY: number, cameraZoom: number,
  centering: "midrange" | "mean",
  showLegend, showAxes: boolean,
  showTourDescription: boolean | null,
  themeMode: "light" | "dark" | "system",
}
```

When `spec` changes, its set fields are pushed into the component. Settings embedded in a
Parquet file apply only to fields that `spec` leaves unset. To fully reset the component
for new data, remount it with `key={fileName}` (the dtour.dev webapp does this). Persist
user changes from `onSpecChange`.

## Advanced

- `<DtourViewer>` together with the exported Jotai atoms (`tourPositionAtom`,
  `pointColorByAtom`, `cameraZoomAtom`, `metadataAtom`, …) lets you drive state from
  your own Jotai `Provider`.
- `CircularSlider`, `DtourToolbar`, and `RadialChart`/`parseMetrics` are the individual
  UI pieces.
- `parseEmbeddedConfig` reads the Parquet `dtour` metadata.

## dtour.dev

The hosted app (`packages/webapp` in the repo) accepts Parquet, Arrow, and CSV files by
drag-and-drop or file picker. `https://dtour.dev/?url=<encoded URL>` opens a remote
file directly, skipping the intro. The file's host has to allow CORS. A failed fetch only
logs to the browser console and leaves the start page up.
