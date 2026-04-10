export type GpuContext = {
  adapter: GPUAdapter;
  device: GPUDevice;
};

/**
 * Request WebGPU adapter and device. Must run inside the GPU Worker
 * (or main thread for testing). Throws if WebGPU is unavailable.
 */
export const initDevice = async (): Promise<GpuContext> => {
  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is not supported in this environment. ' +
        'Please use Chrome 113+, Edge 113+, or Firefox Nightly with WebGPU enabled.',
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    throw new Error(
      'No WebGPU adapter found. Your GPU may not support WebGPU, ' +
        'or the browser may have blocked access.',
    );
  }

  if (!adapter.features.has('float32-blendable')) {
    throw new Error(
      'Your GPU does not support float32-blendable, which is required for HDR rendering. ' +
        'Please try a different browser or device with a GPU that supports this WebGPU feature.',
    );
  }

  const device = await adapter.requestDevice({
    label: 'dtour-scatter',
    requiredFeatures: ['float32-blendable'],
    requiredLimits: {
      // Raise buffer limits to the adapter maximum so large datasets (>128 MB) work.
      // Default spec values are only 128 MB / 256 MB; real hardware supports much more.
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  device.lost.then((info) => {
    // Surface device loss as a console error; client.ts can recover
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
  });

  return { adapter, device };
};
