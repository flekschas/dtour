import computeShader from './compute-projection.wgsl?raw';

export type ProjectionPipeline = {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  paramsBuffer: GPUBuffer;
};

// Params uniform layout (16 bytes):
//   offset  0: num_points  (u32)
//   offset  4: num_dims    (u32)
//   offset  8: viewport_scale (f32)
//   offset 12: _pad        (f32)
const PARAMS_SIZE = 16;

const WORKGROUP_SIZE = 256;

export const createProjectionPipeline = (device: GPUDevice): ProjectionPipeline => {
  const module = device.createShaderModule({
    label: 'compute-projection',
    code: computeShader,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'projection-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipeline = device.createComputePipeline({
    label: 'projection-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module, entryPoint: 'project' },
  });

  const paramsBuffer = device.createBuffer({
    label: 'projection-params',
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return { pipeline, bindGroupLayout, paramsBuffer };
};

export const writeProjectionParams = (
  device: GPUDevice,
  paramsBuffer: GPUBuffer,
  numPoints: number,
  numDims: number,
  viewportScale: number,
): void => {
  const data = new ArrayBuffer(PARAMS_SIZE);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = numPoints;
  u32[1] = numDims;
  f32[2] = viewportScale;
  f32[3] = 0; // pad
  device.queue.writeBuffer(paramsBuffer, 0, data);
};

export type ProjectionResources = {
  dataBuffer: GPUBuffer; // all dims concatenated
  normParamsBuffer: GPUBuffer;
  basisBuffer: GPUBuffer;
  projectedBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
};

export const createProjectionResources = (
  device: GPUDevice,
  pipeline: ProjectionPipeline,
  numPoints: number,
  numDims: number,
): Omit<ProjectionResources, 'bindGroup'> & { bindGroup: null } => {
  const dataBuffer = device.createBuffer({
    label: 'nd-data',
    size: numPoints * numDims * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const normParamsBuffer = device.createBuffer({
    label: 'norm-params',
    size: numDims * 8, // vec2f per dim
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const basisBuffer = device.createBuffer({
    label: 'basis',
    size: numDims * 2 * 4, // p×2 floats
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const projectedBuffer = device.createBuffer({
    label: 'projected',
    size: numPoints * 2 * 4, // N×2 floats
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  return { dataBuffer, normParamsBuffer, basisBuffer, projectedBuffer, bindGroup: null };
};

export const createProjectionBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  paramsBuffer: GPUBuffer,
  res: Omit<ProjectionResources, 'bindGroup'>,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'projection-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: res.dataBuffer } },
      { binding: 2, resource: { buffer: res.normParamsBuffer } },
      { binding: 3, resource: { buffer: res.basisBuffer } },
      { binding: 4, resource: { buffer: res.projectedBuffer } },
    ],
  });

export const dispatchProjection = (
  device: GPUDevice,
  pipeline: ProjectionPipeline,
  bindGroup: GPUBindGroup,
  numPoints: number,
): GPUCommandBuffer => {
  const encoder = device.createCommandEncoder({ label: 'projection-dispatch' });
  const pass = encoder.beginComputePass({ label: 'project-points' });
  pass.setPipeline(pipeline.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(numPoints / WORKGROUP_SIZE));
  pass.end();
  return encoder.finish();
};
