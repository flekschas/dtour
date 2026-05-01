# Changelog

## v0.4.0

### python
- feat: `TourResult.from_parquet()` classmethod to extract tours from Parquet metadata
- feat: `tour_dimensions` traitlet for explicit tour column-name support
- refactor: rename `spectrum_tour` → `attraction_repulsion_tour`
- fix: auto-coerce `tour_by` mismatches instead of raising errors

### scatter
- feat: `tourMode` parameter on `setBases()` to skip orthonormalization for parameter tours
- feat: configurable `minPointSize` and `fillTarget` for density-adaptive point sizing
- feat: conditional zoom-based opacity scaling via `scaleOpacityByZoom`
- feat: 2D colormap rendering in WebGL shaders (LUT and Oklab polar)
- perf: columnar parquet streaming via `onChunk` avoids per-row object allocation for large datasets

### viewer
- feat: predefined tour support — locks column toggles, preview count, and Dims/PCA toggle
- feat: `expandBases()` maps subset-dimension tours into full column space
- feat: `minPointSize` rendering control with spec sync
- feat: zoom control reworked to percentage-based steps (25%–400%)
- feat: smooth guided-mode resume with basis-blend projection transition
- feat: projection-anchored hover highlight with per-point color
- feat: hover tooltip anchored to projection space with directional arrow
- fix: clear hover highlight and tooltip on projection change
- fix: apply resolved theme class to Radix portal container for light-mode support
- style: unify tooltip, popover, and dropdown backgrounds
- perf: spatial index rebuilds use imperative subscriptions to avoid 60fps re-renders in guided mode

### webapp
- feat: show parsing spinner until first render after data load
- feat: `serveDataDir` Vite plugin to serve monorepo `data/` directory in dev

## v0.3.0

### python
- feat: `sequential_tour` for warm-started DR sequences (UMAP, t-SNE, pymde, or custom callables)
- feat: `aligned_umap_tour` using UMAP's joint AlignedUMAP optimisation
- feat: `EmbeddingStep` dataclass for per-frame method/kwarg overrides
- refactor: `spectrum_tour` now delegates to `sequential_tour`

### scatter
- feat: 2D colormap encoding (two numeric columns mapped to procedural 2D colormaps)

### viewer
- feat: 2D colormap mode with 1D/2D toggle and colormap picker
- feat: hover tooltip with lazy point data loading
- feat: kdbush spatial index for sub-millisecond point picking (replaces O(n) GPU scan)
- perf: click-to-select is now synchronous on main thread (no worker round-trip)

### scatter
- feat: `getProjectedPositions()` API for client-side spatial indexing
- feat: `getPointData(index)` API for lazy column value readback
- refactor: remove `pickPoint` in favor of client-side kdbush spatial index
- fix: add `COPY_SRC` to data and categorical GPU buffers for readback

### python
- feat: spectrum tour with configurable parameters
- feat: bidirectional point selection sync via `selected_indices` traitlet
- feat: fine-grained point selections
- refactor: switch PyMDE regularization to concave log penalty
- chore: enforce synced `tourMode` and `tourBy` for parameter tours

### viewer
- feat: support preview counts 2-16 with U-shape and perimeter layouts
- feat: spectrum tour support and updated toolbar/gallery
- feat: bidirectional point selection sync
- fix: align circular slider ticks with gallery layout positions
- fix: account for frame summaries in selector size computation
- fix: suppress spurious `tourBy` coercion warnings
- fix: guard `parseEmbeddedConfig` log behind dev mode
- fix: lasso selection and vertical toolbar offset
- fix: point selection propagation

### scatter
- feat: bidirectional point selection sync
- fix: hardcoded preview canvas resolution -> now track layout size × DPR

### webapp
- feat: add CSV support

## v0.2.0

### python
- feat: LE, signed LE, and spectral Fisher / LDA tours
- feat: embed spec in Parquet files
- feat: tour descriptions and per-frame feature correlations
- fix: signed and Fisher tour correctness

### viewer
- feat: 3D manual rotation around the residual PC
- feat: equal-spacing slider and axis overlay in guided mode
- feat: frame numbers and feature correlation display
- fix: avoid race condition in worker communication

### scatter
- feat: 3D manual rotation around the residual PC
- perf: rendering and color encoding performance
- perf: better memory usage (specifically for the WebGPU backend)

## v0.1.0

Initial release.
