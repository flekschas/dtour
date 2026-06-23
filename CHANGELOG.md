# Changelog

## v0.4.2

### viewer

- fix: set tour position on first load
- chore: add a visible button to exit grand mode
- chore: improve axis drag handle hover indication from panning
- chore: auto-fade out legend sidebar in grand tour mode
- chore: automatically color the generated "Gaussian blobs" and "Rings" examples
- chore: optimize landing page for portrait small screens (such that it displays nicely on a smartphone)

## v0.4.1

### scatter
- fix: detect WebGPU support (`detectBackend()`) and fall back to the WebGL2 backend when WebGPU or the `float32-blendable` feature is unavailable
- fix: explicitly enable the `EXT_float_blend` extension in the WebGL backend

### viewer
- fix: `backend` now defaults to `'auto'`, falling back to WebGL2 when WebGPU is unsupported (e.g. Firefox) instead of rendering nothing
- fix: logo in Safari which struggles hard with `stroke-dashoffset` and multi-path `<clipPath>`. Sad.

### webapp
- fix: default renderer to auto-detection (WebGPU with WebGL2 fallback)

## v0.4.0

### BREAKING CHANGES

- **Embedded config**: `EmbeddedConfig.tour` fields renamed — `views` → `keyframes`, `tourMode` → `family` (`'hyperdimensional' | 'sequential'`), `tourDescription` → `description`, `frameSummaries`/`tourFrameDescription` → `keyframeDescriptions`, `frameLoadings` → `keyframeLoadings` (now `KeyframeLoading[]` with `{primary, secondary}` shape instead of `[string, number][][]`)
- **Embedded config**: `nViews` and `nDims` removed from `EmbeddedConfig` type (only used internally during parsing)
- **Embedded config**: `tour.family` and `tour.dimensions` are now required — tours without valid values are rejected by the parser
- **DtourSpec**: `viewMode` → `tourTraversal`, `showFrameNumbers` → `showKeyframeNumbers`, `showFrameLoadings` → `showKeyframeLoadings`, `sliderSpacing` → `tourSliderSpacing`, `colorMap` → `pointColorMap`
- **Atoms**: `viewModeAtom` → `tourTraversalAtom`, `tourModeAtom` → `tourFamilyAtom`, `frameLoadingsAtom` → `keyframeLoadingsAtom`, `frameSummariesAtom`/`tourFrameDescriptionAtom` → `keyframeDescriptionsAtom`, `showFrameNumbersAtom` → `showKeyframeNumbersAtom`, `showFrameLoadingsAtom` → `showKeyframeLoadingsAtom`, `sliderSpacingAtom` → `tourSliderSpacingAtom`
- **Component props**: `tourMode` → `tourFamily`, `tourFrameDescription`/`frameSummaries` → `keyframeDescriptions`, `frameLoadings` → `keyframeLoadings`
- **Scatter API**: `setBases(bases, tourMode)` → `setBases(bases, tourFamily)` where `'sequential'` skips orthonormalization
- **Python `TourResult`**: `tour_mode` → `tour_family`, `tour_description` → `description`, `tour_frame_description`/`frame_summaries` → `keyframe_descriptions`
- **Python widget**: `show_frame_loadings` → `show_keyframe_loadings`, `view_mode` → `tour_traversal`
- **Python `build_dtour_metadata`**: `view_mode` → `tour_traversal`, `slider_spacing` → `tour_slider_spacing`, `color_map` → `point_color_map`, `show_frame_numbers` → `show_keyframe_numbers`, `show_frame_loadings` → `show_keyframe_loadings`
- **Python functions**: `sequential_tour`, `aligned_umap_tour` parameters renamed (`frame_summaries`/`tour_description`/`tour_frame_description` → `keyframe_descriptions`/`description`)
- **No backward compatibility**: old Parquet files and `.npz` tours with legacy field names are no longer parsed. Re-export data with the new format.

### python
- feat: `TourResult.from_parquet()` classmethod to extract tours from Parquet metadata
- feat: `tour_dimensions` traitlet for explicit tour column-name support
- feat: `centering` traitlet and spec parameter (`'midrange'` / `'mean'`)
- refactor: rename `spectrum_tour` → `attraction_repulsion_tour`
- fix: auto-coerce `tour_by` mismatches instead of raising errors

### scatter
- feat: configurable projection centering (`setCentering('midrange' | 'mean')`) with consistent normalization across WebGPU, WebGL, PCA, and residual-PC shaders
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
- feat: show color in point tooltip
- feat: add tour slider visibility settings
- feat: add ability to reset spec to default settings
- feat: configurable projection centering (midrange / mean)
- feat: drag-to-pan and zoom-about-cursor with toolbar toggle for scroll semantics
- fix: clear hover highlight and tooltip on projection change
- refactor: make toolbar design more responsive
- refactor: hide origin dot until axes are shown
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
