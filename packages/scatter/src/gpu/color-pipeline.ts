import colorCategoricalWgsl from '../shaders/color-categorical.wgsl?raw';
import colorContinuousWgsl from '../shaders/color-continuous.wgsl?raw';
import selectCategoricalWgsl from '../shaders/select-categorical.wgsl?raw';
import selectContinuousWgsl from '../shaders/select-continuous.wgsl?raw';

const WORKGROUP_SIZE = 256;

// Shared bind group layout for all 4 compute operations:
//   binding 0: read-only storage  (data / indices source)
//   binding 1: read-only storage  (colormap / palette / ranges / selected-labels)
//   binding 2: read-write storage (color / mask output)
//   binding 3: uniform            (params)

// Max params struct size across all shaders: 5 fields × 4 = 20 bytes → pad to 32
const PARAMS_SIZE = 32;

export type ColorPipelines = {
  colorContinuous: GPUComputePipeline;
  colorCategorical: GPUComputePipeline;
  selectContinuous: GPUComputePipeline;
  selectCategorical: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  paramsBuffer: GPUBuffer;
};

export const createColorPipelines = (device: GPUDevice): ColorPipelines => {
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'color-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });

  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const makePipeline = (label: string, code: string): GPUComputePipeline =>
    device.createComputePipeline({
      label,
      layout,
      compute: { module: device.createShaderModule({ label, code }), entryPoint: 'main' },
    });

  const paramsBuffer = device.createBuffer({
    label: 'color-params',
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return {
    colorContinuous: makePipeline('color-continuous', colorContinuousWgsl),
    colorCategorical: makePipeline('color-categorical', colorCategoricalWgsl),
    selectContinuous: makePipeline('select-continuous', selectContinuousWgsl),
    selectCategorical: makePipeline('select-categorical', selectCategoricalWgsl),
    bindGroupLayout,
    paramsBuffer,
  };
};

/** Create a bind group for any of the 4 color/selection compute operations. */
export const createColorBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  source: GPUBuffer,
  lookup: GPUBuffer,
  output: GPUBuffer,
  params: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'color-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: source } },
      { binding: 1, resource: { buffer: lookup } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ],
  });

/** Dispatch a color/selection compute shader and return the command buffer. */
export const dispatchColorCompute = (
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  numPoints: number,
): GPUCommandBuffer => {
  const encoder = device.createCommandEncoder({ label: 'color-dispatch' });
  const pass = encoder.beginComputePass({ label: 'color-compute' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
  pass.end();
  return encoder.finish();
};

// ─── Params writers ────────────────────────────────────────────────────────

export const writeContinuousColorParams = (
  device: GPUDevice,
  buf: GPUBuffer,
  numPoints: number,
  columnOffset: number,
  min: number,
  range: number,
  lutSize: number,
): void => {
  const data = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(data);
  const f = new Float32Array(data);
  u[0] = numPoints;
  u[1] = columnOffset;
  f[2] = min;
  f[3] = range;
  u[4] = lutSize;
  device.queue.writeBuffer(buf, 0, data);
};

export const writeCategoricalColorParams = (
  device: GPUDevice,
  buf: GPUBuffer,
  numPoints: number,
  paletteSize: number,
): void => {
  const data = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(data);
  u[0] = numPoints;
  u[1] = paletteSize;
  device.queue.writeBuffer(buf, 0, data);
};

export const writeContinuousSelectParams = (
  device: GPUDevice,
  buf: GPUBuffer,
  numPoints: number,
  columnOffset: number,
  numRanges: number,
): void => {
  const data = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(data);
  u[0] = numPoints;
  u[1] = columnOffset;
  u[2] = numRanges;
  device.queue.writeBuffer(buf, 0, data);
};

export const writeCategoricalSelectParams = (
  device: GPUDevice,
  buf: GPUBuffer,
  numPoints: number,
  numLabels: number,
): void => {
  const data = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(data);
  u[0] = numPoints;
  u[1] = numLabels;
  device.queue.writeBuffer(buf, 0, data);
};

/** Pack an RGB palette [[r,g,b], ...] into Uint32Array of 0xAABBGGRR values. */
export const packPalette = (palette: [number, number, number][]): Uint32Array => {
  const packed = new Uint32Array(palette.length);
  for (let i = 0; i < palette.length; i++) {
    const [r, g, b] = palette[i]!;
    packed[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return packed;
};
