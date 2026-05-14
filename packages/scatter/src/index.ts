// @dtour/scatter — scatter renderer with three-thread architecture.
// GPU work (rendering) and data work (parsing, normalization) each run in
// dedicated workers, keeping the main thread free for UI.

export type { Colormap2DName } from './data/colormaps2d.ts';
export { COLORMAP_2D_INDEX, COLORMAP_2D_NAMES, packColormap2DLut } from './data/colormaps2d.ts';
export { GLASBEY_DARK, GLASBEY_LIGHT, MAGMA_25, OKABE_ITO, VIRIDIS_25 } from './data/palettes.ts';
export type { Metadata } from './data/types.ts';
export type { ScatterInstance, ScatterOptions, ScatterStatus } from './gpu/client.ts';
export { createScatter } from './gpu/client.ts';
export { bitPackIndices } from './selection.ts';
export { computeArcLengths, interpolateAtPosition } from './tour/arc-length.ts';
export { createScatterWebGL } from './webgl/client.ts';
