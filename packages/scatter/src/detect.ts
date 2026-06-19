export type ScatterBackend = 'webgpu' | 'webgl';

/**
 * Detect the best available rendering backend.
 *
 * Returns 'webgpu' only when the environment supports everything the WebGPU
 * renderer requires — a WebGPU adapter exposing the `float32-blendable`
 * feature used for HDR blending (see gpu/device.ts, which throws without it).
 * Otherwise returns 'webgl', the WebGL2 backend that runs on any WebGL2-capable
 * browser. As of mid-2026 Firefox falls into this bucket: it either lacks
 * WebGPU or ships it without `float32-blendable`, so a bare `navigator.gpu`
 * check is not sufficient — we must probe the adapter feature.
 *
 * Never throws; any failure resolves to 'webgl'. Must run on a thread with
 * access to `navigator.gpu` (main thread or worker).
 */
export const detectBackend = async (): Promise<ScatterBackend> => {
  try {
    if (typeof navigator === 'undefined' || !navigator.gpu) return 'webgl';
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return 'webgl';
    if (!adapter.features.has('float32-blendable')) return 'webgl';
    return 'webgpu';
  } catch {
    return 'webgl';
  }
};
