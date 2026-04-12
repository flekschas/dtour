# Changelog

## v0.3.0

### scatter
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
