# Changelog

## Next

### BREAKING CHANGES

Names now use "keyframe" for a stop on the tour and "preview" for its thumbnail in the gallery.

- **DtourSpec**: `previewScale: 1 | 0.75 | 0.5` → `previewSize: 'auto' | 'small' | 'medium' | 'large'`, `showKeyframeNumbers: boolean` → `previewKeyframeNumbers: 'auto' | 'visible' | 'hidden'`, `showKeyframeLoadings: boolean` → `previewLabelContent: 'auto' | 'description' | 'loadings'`
- **Atoms**: `showKeyframeNumbersAtom` → `previewKeyframeNumbersAtom`, `showKeyframeLoadingsAtom` → `previewLabelContentAtom`; `previewSizeAtom` is now exported; `selectedKeyframeAtom` is removed because the active preview now always follows the slider (`currentKeyframeAtom`)
- **Component props**: `Dtour` and `DtourViewer` `views` → `keyframes`
- **Viewer functions**: `createDefaultViews` → `createDefaultKeyframes`
- **Python `TourResult`**: `views` → `keyframes`, `n_views` → `n_keyframes` (now a property), `views_raw` → `keyframes_raw`. The constructor takes `keyframes` as its only positional argument and no longer accepts `views` or `n_views`, e.g. `TourResult(keyframes, n_dims=5)`
- **Python widget**: `show_keyframe_loadings` → `preview_label_content`, `theme` → `theme_mode` (matching the spec's `themeMode`); `preview_size` now also accepts `"auto"`, which is the new default
- **Python `build_dtour_metadata` / `add_spec_to_parquet`**: `preview_scale` → `preview_size`, `show_keyframe_numbers` → `preview_keyframe_numbers`, `show_keyframe_loadings` → `preview_label_content`
- **Python `compute_metrics`**: `views` → `keyframes`
- **Backward compatibility**: Parquet files with the old spec names still load. Old Python names still work but raise a `DeprecationWarning`, except in the `TourResult` constructor. Saved tours (`.npz` and Parquet) keep their format.

### python

- feat: add a read-only `Widget.tour_family` property
- feat: add `preview_keyframe_numbers` and `preview_label_content` traitlets, so every preview setting is available from Python
- feat: add a `preview_label_visibility` traitlet and raise the `preview_count` limit from 16 to 32. `set_tour()` accepts tours of any length; longer tours preview a sample of their keyframes
- feat: add `tour_slider_spacing`, `tour_slider_visibility`, `min_point_size`, and `show_axes` traitlets, and the matching `build_dtour_metadata` arguments where missing
- fix: keep label columns of pandas DataFrames — categorical, string, object, and boolean columns become Arrow string columns (with missing values as nulls), so `point_color_by` works with plain pandas input. Other types, like datetimes, are only included when listed in `from_pandas(columns=...)`. Column names that collide as strings (e.g., `1` and `"1"`) now raise a `ValueError`
- fix: preserve sequential interpolation (no "breathing") and show tour descriptions, keyframe labels, and loadings when a widget view opens, including in marimo. This also removes the tour-family console warning
- fix: `set_tour()` with a sequential tour now switches `tour_by` to `"parameter"` when the widget previously had a hyperdimensional tour
- fix: `preview_size` supports `"auto"` and uses it by default, so widgets pick the preview size from the available space like the web viewer
- chore: explain how to build a missing widget bundle, and warn on import in a repo checkout when the bundle is older than its sources or build configuration
- chore: add `pnpm build:widget` to build the widget bundle together with the `@dtour/scatter` and `@dtour/viewer` packages it bundles
- chore: rename the private widget frontend package from `@dtour/python-build` to `@dtour/python-widget`
- chore: update `uv.lock` to match `pyproject.toml`
- fix: Parquet exports of embedding tours (`le_tour`, `umap_little_tour`, sequential tours) record the embedding columns as the tour dimensions instead of the input features, which broke files with extra numeric columns. `add_spec_to_parquet()` and `Widget.save_spec_to_parquet()` infer them from the table; `build_dtour_metadata()` requires `tour_dimensions` for these tours
- docs: document all tour generators in the README

### scatter

- fix: treat string columns whose first value is null as categorical

### viewer

- feat: previews show keyframe numbers when some keyframes have no preview (`previewKeyframeNumbers: 'auto'`)
- feat: show up to 32 previews. Layouts for up to 16 previews are unchanged; larger counts use a wide perimeter grid with 4–6 rows
- feat: `CircularSlider` and `RadialChart` accept a `startAngle`, so both line up with the gallery that is actually shown
- feat: add `previewLabelVisibility: 'auto' | 'visible' | 'interactive' | 'hidden'` and a matching "Labels" toolbar control. `'interactive'` shows the label inside the preview on hover and for the current keyframe, so labels no longer take space from the previews. `'auto'` uses `'visible'` up to 16 previews and `'interactive'` above
- fix: tours with more keyframes than the gallery can show no longer stack all previews in the top-left corner. The gallery previews the 32 keyframes most evenly spaced along the tour (by normalized geodesic distance), always including the first and last, and the slider keeps a tick for every keyframe
- fix: align radial metric bars with the slider ticks for every preview count. Previously the bars were rotated away from the ticks for counts other than 4, 8, 12, and 16
- fix: show fewer previews instead of unusably small ones in narrow or short containers, such as phones. Each preview stays at least 24px, the shown keyframes are sampled like for long tours, and the gallery hides when not even two previews fit
- fix: the active preview always follows the slider. Clicking a preview moves the slider to it but no longer keeps it highlighted after scrubbing elsewhere
- chore: remove the dev-only warning about `views.length` differing from `previewCount`, which predefined tours no longer need
- chore: add a preview-fit regression check (`pnpm --filter @dtour/viewer check:preview-fit`) and run it in CI

### webapp

- feat: example buttons show a preview video of their dataset that loops while the button is hovered or focused. In light mode the video is inverted with its hues kept
- feat: on screens from 1440px, 1600px, and 1920px wide, the example grid gets wider with larger gaps and taller buttons
- fix: the webapp's responsive and hover styles (e.g., the example grid's `sm:` gap and the drop button's hover background) no longer lose to same-named classes from the viewer's stylesheet
- fix: `?url=` and `?dataset=` links load their data without also needing `&benchmark`

### agents

- ai: add a dtour agent skill (`npx skills add flekschas/dtour`) with usage guidance, API references, and the paper

## v0.4.4

### python

- chore: lower the minimum Python to 3.10 (from 3.12) for broader availability — supports 3.10–3.13 (numpy/scikit-learn floors are set to the last series with 3.10 wheels)
- chore: slim the base dependencies to only what `dtour` imports at runtime (`anywidget`, `arro3-core`, `arro3-io`, `numpy`, `pyamg`, `scikit-learn`)
- chore: make `openTSNE` optional via `pip install 'dtour[tsne]'` — it's only used by the t-SNE tour engine (the default for `attraction_repulsion_tour`), like `umap`/`pymde`
- chore: drop `pyarrow` from the dependencies — all Arrow work (IPC, Parquet) goes through `arro3`; pyarrow is only consumed as an optional input format (duck-typed via `__arrow_c_stream__`), so it needs no declared dependency, same as pandas/polars
- chore: drop `jupyter-scatter`, `marimo`, and `matplotlib` from the package dependencies — none are used by `dtour` itself; they're only needed to run the demo notebooks (`pip install 'dtour[demo]'`). Jupyter/Marimo are host environments the consumer already provides. Dropping `jupyter-scatter` also unblocks Python 3.14, since it pulled in `geoindex-rs`, which has no 3.14 wheels
- chore: make `cev-metrics` optional via `pip install 'dtour[cev]'` — it's only needed for the `confusion` metric and ships wheels only for Python ≤3.12, so keeping it out of the base deps keeps the core install wheel-only on Python 3.13+
- ci: test the base install against Python 3.10, 3.11, 3.12, 3.13, and 3.14 (3.14 is allowed to fail until native deps like `pyamg` ship wheels for it)

## v0.4.3

### python

- chore: make `umap-learn` optional via `pip install 'dtour[umap]'`
- chore: set a minimum numba version for Python 3.12 support

### viewer

- feat: show a between-keyframe indicator for sequential tools to inform the user that between-keyframe projections should not be interpreted structurally for sequential tours

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
