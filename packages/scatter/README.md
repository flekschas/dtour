# dtour: Scatter

This is the dtour rendering engine: a framework-agnostic, high-performance scatter renderer that projects high-dimensional data to 2D and interpolates between tour keyframes on the GPU. It has no UI and no framework dependency — it's the core that [`@dtour/viewer`](../viewer) (React) and, transitively, the [Python widget](../python) are built on.

It uses a **three-thread architecture**: GPU work (rendering) runs in one Web Worker, data work (Arrow/Parquet parsing, normalization, selection) in another, and the main thread stays free for UI. Rendering targets **WebGPU** and falls back to **WebGL2**.

## Install

```sh
npm install @dtour/scatter
```

## Quick start

```ts
import { createScatter } from "@dtour/scatter";

const canvas = document.querySelector("canvas")!;
const scatter = createScatter({ canvas });

scatter.loadData(arrowBuffer); // Arrow IPC or Parquet ArrayBuffer (ownership transferred)
scatter.render();
```

For automatic backend selection with a WebGL2 fallback:

```ts
import { createScatter, createScatterWebGL, detectBackend } from "@dtour/scatter";

const backend = await detectBackend(); // "webgpu" | "webgl"
const scatter =
  backend === "webgpu" ? createScatter({ canvas }) : createScatterWebGL({ canvas });
```

## API

### `createScatter(options) → ScatterInstance`

Also `createScatterWebGL(options)` (same signature and return type, WebGL2 backend).

```ts
type ScatterOptions = {
  canvas: HTMLCanvasElement; // the canvas to render into
  zoom?: number;             // initial camera zoom (default 1)
  dpr?: number;              // device pixel ratio (default window.devicePixelRatio ?? 1)
};
```

### `ScatterInstance`

**Data & tour**

```ts
scatter.loadData(buffer);                 // Arrow IPC or Parquet ArrayBuffer (ownership transferred)
scatter.setBases(bases, tourFamily?);     // Float32Array[] of p×2 column-major keyframe bases;
                                          //   tourFamily: "hyperdimensional" (default) | "sequential"
scatter.setTourPosition(position);        // scrub along the arc-length path, [0, 1]
scatter.setDirectBasis(basis);            // set a single basis directly (manual/zen modes)
scatter.startPlayback(speed, direction);  // worker-driven rAF playback; direction: 1 | -1
scatter.stopPlayback();
```

**Camera & style**

```ts
scatter.setCamera({ pan, zoom, insetOffsetY, insetZoom });
scatter.setCentering("midrange" | "mean");
scatter.setStyle({ pointSize, opacity, color, minPointSize, fillTarget }); // sizes accept "auto"
scatter.setBackgroundColor([r, g, b]);    // RGB 0–1
scatter.resize(viewIndex, width, height, dpr?);
scatter.setMaxPoints(n);                  // decimate for huge datasets; 0 = render all
scatter.render();
```

**Color encoding**

```ts
scatter.encodeColor(column, palette?, theme?, colorMap?); // categorical or numeric column
scatter.encodeColor2D(columnX, columnY, colormap?);       // two numeric columns → 2D colormap
scatter.clearColor();
```

**Selection**

```ts
scatter.selectByColumn(column, { labelIndices, valueRanges });
scatter.setSelectionMask(mask);   // bit-packed, 1 bit/point (ceil(numPoints / 32) u32s)
scatter.lassoSelect(polygon);     // NDC polygon; GPU point-in-polygon test
scatter.clearSelection();
```

**Previews, 3D, introspection & lifecycle**

```ts
scatter.addPreviewCanvas(id, canvas);     // keyframe preview thumbnails
scatter.resizePreview(id, width, height);
scatter.removePreviewCanvas(id);

scatter.enable3d();                        // 3D camera rotation (adds a residual PC as 3rd axis)
scatter.set3dRotation(matrix);             // 3×3 column-major (9 floats)
scatter.disable3d();

scatter.computePCA();                      // GPU PCA; result arrives via subscribe("pcaResult")
await scatter.getProjectedPositions();     // N×2 interleaved Float32Array (spatial indexing)
await scatter.getPointData(pointIndex);
await scatter.getMetrics();                // GPU memory, dims, JS heap usage
await scatter.benchmark(numFrames?);       // sweep the tour, time each frame

const unsub = scatter.subscribe((status) => { /* ScatterStatus events from both workers */ });
scatter.destroy();                         // terminate workers, release resources
```

### Other exports

- `detectBackend()` → `Promise<"webgpu" | "webgl">`, and the `ScatterBackend` type.
- Palettes: `OKABE_ITO`, `GLASBEY_LIGHT`, `GLASBEY_DARK`, `VIRIDIS_25`, `MAGMA_25`.
- 2D colormaps: `COLORMAP_2D_NAMES`, `COLORMAP_2D_INDEX`, `packColormap2DLut`, and the `Colormap2DName` type.
- Tour math: `computeArcLengths`, `interpolateAtPosition`.
- Selection helper: `bitPackIndices`.
- Types: `ScatterOptions`, `ScatterInstance`, `ScatterStatus`, `Metadata`.

## Development

```sh
pnpm --filter @dtour/scatter build   # bundle to dist/ (workers inlined)
pnpm --filter @dtour/scatter dev     # watch mode
```
