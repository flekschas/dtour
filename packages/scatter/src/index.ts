// @dtour/scatter — WebGPU scatter renderer with three-thread architecture.
// GPU work (rendering) and data work (parsing, normalization) each run in
// dedicated workers, keeping the main thread free for UI.

export { createScatter } from './gpu/client.ts';
export type { ScatterOptions, ScatterInstance, ScatterStatus } from './gpu/client.ts';
export type { Metadata } from './data/types.ts';
export { computeArcLengths, interpolateAtPosition } from './tour/arc-length.ts';
