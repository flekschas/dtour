import computeShader from './compute-pca.wgsl?raw';
import { jacobiEigen } from './jacobi.ts';

export type PcaPipeline = {
  partialPipeline: GPUComputePipeline;
  finalPipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  paramsBuffer: GPUBuffer;
};

// Params uniform: { num_points: u32, num_dims: u32, num_workgroups: u32, _pad: u32 }
const PARAMS_SIZE = 16;
const WORKGROUP_SIZE = 256;
const MAX_DIMS = 32;

export const createPcaPipeline = (device: GPUDevice): PcaPipeline => {
  const module = device.createShaderModule({
    label: 'compute-pca',
    code: computeShader,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'pca-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const partialPipeline = device.createComputePipeline({
    label: 'pca-partial-pipeline',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'reduce_partial' },
  });

  const finalPipeline = device.createComputePipeline({
    label: 'pca-final-pipeline',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'final_reduce' },
  });

  const paramsBuffer = device.createBuffer({
    label: 'pca-params',
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return { partialPipeline, finalPipeline, bindGroupLayout, paramsBuffer };
};

/** Run PCA on data already in GPU buffers. Returns eigenvectors sorted by descending eigenvalue. */
export const runPCA = async (
  device: GPUDevice,
  pipeline: PcaPipeline,
  dataBuffer: GPUBuffer,
  normParamsBuffer: GPUBuffer,
  numPoints: number,
  numDims: number,
): Promise<{ eigenvalues: Float32Array; eigenvectors: Float32Array[] }> => {
  const d = Math.min(numDims, MAX_DIMS);
  const numCov = (d * (d + 1)) / 2;
  const numTerms = d + numCov;
  const numWorkgroups = Math.ceil(numPoints / WORKGROUP_SIZE);

  // Write params
  const paramsData = new ArrayBuffer(PARAMS_SIZE);
  const u32 = new Uint32Array(paramsData);
  u32[0] = numPoints;
  u32[1] = d;
  u32[2] = numWorkgroups;
  u32[3] = 0;
  device.queue.writeBuffer(pipeline.paramsBuffer, 0, paramsData);

  // Ephemeral buffers
  const partialsBuffer = device.createBuffer({
    label: 'pca-partials',
    size: numWorkgroups * numTerms * 4,
    usage: GPUBufferUsage.STORAGE,
  });

  const resultSize = numTerms * 4;
  const resultBuffer = device.createBuffer({
    label: 'pca-result',
    size: resultSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const readbackBuffer = device.createBuffer({
    label: 'pca-readback',
    size: resultSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const bindGroup = device.createBindGroup({
    label: 'pca-bg',
    layout: pipeline.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: pipeline.paramsBuffer } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: normParamsBuffer } },
      { binding: 3, resource: { buffer: partialsBuffer } },
      { binding: 4, resource: { buffer: resultBuffer } },
    ],
  });

  // Dispatch both passes + copy to readback
  const encoder = device.createCommandEncoder({ label: 'pca-compute' });

  const pass1 = encoder.beginComputePass({ label: 'pca-partial' });
  pass1.setPipeline(pipeline.partialPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  const pass2 = encoder.beginComputePass({ label: 'pca-final' });
  pass2.setPipeline(pipeline.finalPipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultSize);
  device.queue.submit([encoder.finish()]);

  // Readback
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const resultData = new Float32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();

  // Cleanup ephemeral buffers
  partialsBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  // Unpack upper-triangle covariance into full symmetric d×d matrix
  const covMatrix = new Float32Array(d * d);
  for (let d1 = 0; d1 < d; d1++) {
    for (let d2 = d1; d2 < d; d2++) {
      const ci = d1 * d - (d1 * (d1 - 1)) / 2 + (d2 - d1);
      covMatrix[d1 * d + d2] = resultData[d + ci]!;
      covMatrix[d2 * d + d1] = resultData[d + ci]!;
    }
  }

  return jacobiEigen(covMatrix, d);
};
