export type CanvasView = {
  canvas: OffscreenCanvas;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
};

/**
 * Configure an OffscreenCanvas for WebGPU rendering.
 */
export const configureCanvas = (canvas: OffscreenCanvas, device: GPUDevice): CanvasView => {
  const context = canvas.getContext('webgpu');
  if (!context) {
    throw new Error(
      'Failed to get WebGPU context from canvas. Ensure WebGPU is supported and the canvas is valid.',
    );
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });
  return { canvas, context, format };
};

/**
 * Encode a render pass for numPoints instanced quads into a texture view.
 * The target is typically an HDR (rgba32float) texture for later tone mapping.
 * Returns a GPUCommandBuffer — caller batches with compute commands before submitting.
 */
export const renderPoints = (
  device: GPUDevice,
  target: GPUTextureView,
  pipeline: GPURenderPipeline,
  bindGroup: GPUBindGroup,
  numPoints: number,
  clearColor: [number, number, number] = [0, 0, 0],
): GPUCommandBuffer => {
  const encoder = device.createCommandEncoder({ label: 'render-points' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: target,
        clearValue: { r: clearColor[0], g: clearColor[1], b: clearColor[2], a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      },
    ],
  });

  if (numPoints > 0) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, numPoints);
  }
  pass.end();

  return encoder.finish();
};

/**
 * Encode a fullscreen tone-map pass that reads an HDR texture and writes to a canvas.
 */
export const tonemapToCanvas = (
  device: GPUDevice,
  view: CanvasView,
  tonemapPipeline: GPURenderPipeline,
  tonemapBindGroup: GPUBindGroup,
): GPUCommandBuffer => {
  const encoder = device.createCommandEncoder({ label: 'tonemap' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: view.context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      },
    ],
  });
  pass.setPipeline(tonemapPipeline);
  pass.setBindGroup(0, tonemapBindGroup);
  pass.draw(3);
  pass.end();
  return encoder.finish();
};
