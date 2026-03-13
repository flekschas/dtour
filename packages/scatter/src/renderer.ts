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
 * Encode a render pass for numPoints instanced quads.
 * Returns a GPUCommandBuffer — caller batches with compute commands before submitting.
 */
export const renderPoints = (
  device: GPUDevice,
  view: CanvasView,
  pipeline: GPURenderPipeline,
  bindGroup: GPUBindGroup,
  numPoints: number,
  clearColor: [number, number, number] = [0, 0, 0],
): GPUCommandBuffer => {
  const colorTexture = view.context.getCurrentTexture();
  const colorView = colorTexture.createView();

  const encoder = device.createCommandEncoder({ label: 'render-points' });
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: colorView,
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
