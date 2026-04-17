# Changelog

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
