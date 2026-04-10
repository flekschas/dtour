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

// Maximum instances per draw call. Keeps individual GPU submissions short
// enough to avoid Chrome's GPU watchdog timeout (~2-8s depending on
// platform). For 10M points, this produces 5 draws within a single render
// pass — the blend state accumulates correctly across draws.
const MAX_INSTANCES_PER_DRAW = 2_000_000;

/**
 * Encode a render pass for numPoints instanced quads into a texture view.
 * The target is typically an HDR float texture for later tone mapping.
 *
 * Large point counts are split into multiple draw calls using firstInstance
 * offsets so that @builtin(instance_index) covers [0, numPoints) across all
 * draws — storage buffer indexing and decimation hashing are unaffected.
 */
export const renderPoints = (
  encoder: GPUCommandEncoder,
  target: GPUTextureView,
  pipeline: GPURenderPipeline,
  bindGroup: GPUBindGroup,
  numPoints: number,
  clearColor: [number, number, number] = [0, 0, 0],
): void => {
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
    for (let offset = 0; offset < numPoints; offset += MAX_INSTANCES_PER_DRAW) {
      const count = Math.min(MAX_INSTANCES_PER_DRAW, numPoints - offset);
      pass.draw(4, count, 0, offset);
    }
  }
  pass.end();
};

/**
 * Encode a fullscreen tone-map pass that reads an HDR texture and writes to a canvas.
 */
export const tonemapToCanvas = (
  encoder: GPUCommandEncoder,
  view: CanvasView,
  tonemapPipeline: GPURenderPipeline,
  tonemapBindGroup: GPUBindGroup,
): void => {
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
};
