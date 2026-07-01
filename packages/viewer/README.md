# dtour: Viewer

This is the dtour React component: the full tour UI — circular keyframe selector, preview gallery, playback controls, color legend, and radial quality-metric charts — built on top of the [`@dtour/scatter`](../scatter) rendering engine. It's the ready-to-use way to embed dtour in a React app, and it's what the [Python widget](../python) wraps under the hood.

## Install

```sh
npm install @dtour/viewer
```

`react` and `react-dom` are peer dependencies. Import the stylesheet once:

```tsx
import "@dtour/viewer/dist/viewer.css";
```

## Quick start

```tsx
import { Dtour } from "@dtour/viewer";
import "@dtour/viewer/dist/viewer.css";

<Dtour data={arrowBuffer} />
```

`data` is an Arrow IPC or Parquet `ArrayBuffer`. With no `views`, a tour is auto-generated.

## Component API

```tsx
<Dtour
  data={arrowBuffer}          // Arrow IPC or Parquet ArrayBuffer (ownership transferred)
  views={views}               // Float32Array[] of p×2 column-major view matrices
  metrics={metricsBuffer}     // Arrow IPC ArrayBuffer with per-view quality metrics
  metricTracks={tracks}       // RadialTrackConfig[] for radial bar chart customization
  metricBarWidth="full"       // "full" | number — global bar width for radial charts
  colorMap={colorMap}         // Record<string, string | {light, dark}> per-label colors
  spec={spec}                 // partial DtourSpec to control component state
  onSpecChange={handleSpec}   // fires on state change (debounced ~250ms), full resolved spec
  onStatus={handleStatus}     // called on every renderer status event
  onSelectionChange={fn}      // fires when legend selection changes (label names)
  onPointSelectionChange={fn} // fires when lasso selection changes (Uint32Array bit mask)
  onLoadData={fn}             // called when user loads a file via the toolbar (data, fileName)
  onLogoClick={fn}            // called when the user clicks the toolbar logo
  onReady={fn}                // called with a DtourHandle for programmatic control
  hideToolbar={false}         // hide the top toolbar
  backend="auto"              // "auto" (default) | "webgpu" | "webgl"
  tourFamily="hyperdimensional" // "hyperdimensional" | "sequential"
  tourDescription={null}      // human-readable description shown in the sub-bar
  keyframeDescriptions={[...]} // string[] literals, or a template string with
                               //   {primary}, {secondary}, {relation} placeholders
  keyframeLoadings={[...]}    // per-keyframe feature loadings
  portalContainer={el}        // portal Radix popups here (Shadow DOM isolation)
/>
```

### DtourSpec

The `spec` prop drives component state. All fields are optional; omitted fields use defaults (`DTOUR_DEFAULTS`).

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

### DtourHandle

`onReady` hands back an imperative handle for programmatic selection:

```ts
type DtourHandle = {
  select: (indicesOrMask: number[] | Int32Array | Uint32Array, opts?: { isBitPacked?: boolean }) => void;
  selectByLabels: (labels: string[]) => void; // resolve labels against the active color column
  clearSelection: () => void;
};
```

## Advanced exports

For granular control beyond the self-contained `<Dtour>`:

- **`<DtourViewer>`** (`DtourViewerProps`) — the composable viewer without the built-in [Jotai](https://jotai.org) store, for wiring into your own `Provider`.
- **Jotai atoms** — `tourPositionAtom`, `pointColorByAtom`, `cameraZoomAtom`, `themeModeAtom`, … (the full set backing `DtourSpec`), for reading/driving state directly.
- **`RadialChart` / `parseMetrics`** (`RadialChartProps`, `RadialTrackConfig`, `ParsedTrack`) — the quality-metrics visualization used on the circular slider.
- **`CircularSlider`** (`CircularSliderHandle`, `CircularSliderProps`) and **`DtourToolbar`** — the individual UI pieces.
- **`PortalContainerContext`** — portal target for Shadow DOM isolation (used by the anywidget/marimo integration).
- **`DTOUR_DEFAULTS`, `dtourSpecSchema`, `parseEmbeddedConfig`, `createDefaultViews`** — defaults, the [Zod](https://zod.dev) schema for `DtourSpec`, and helpers for embedded config and fallback views.

## Development

```sh
pnpm --filter @dtour/viewer build   # bundle to dist/ (+ viewer.css, index.d.ts)
pnpm --filter @dtour/viewer dev     # watch mode
```
