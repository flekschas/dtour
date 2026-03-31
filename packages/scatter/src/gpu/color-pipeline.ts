import selectCategoricalWgsl from '../shaders/select-categorical.wgsl?raw';
import selectContinuousWgsl from '../shaders/select-continuous.wgsl?raw';

const WORKGROUP_SIZE = 256;

// Shared bind group layout for selection compute operations:
//   binding 0: read-only storage  (data / indices source)
//   binding 1: read-only storage  (ranges / selected-labels)
//   binding 2: read-write storage (mask output)
//   binding 3: uniform            (params)

// Max params struct size across selection shaders: 3 fields × 4 = 12 bytes → pad to 32
const PARAMS_SIZE = 32;

export type ColorPipelines = {
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
    selectContinuous: makePipeline('select-continuous', selectContinuousWgsl),
    selectCategorical: makePipeline('select-categorical', selectCategoricalWgsl),
    bindGroupLayout,
    paramsBuffer,
  };
};

/** Create a bind group for a selection compute operation. */
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

// ─── Params writers ────────────────────────────────────────────────────────

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
