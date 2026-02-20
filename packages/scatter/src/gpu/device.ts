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

  const device = await adapter.requestDevice({
    label: 'dtour-scatter',
  });

  device.lost.then((info) => {
    // Surface device loss as a console error; client.ts can recover
    console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
  });

  return { adapter, device };
};
