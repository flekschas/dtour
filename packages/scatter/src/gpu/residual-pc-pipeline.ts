import computeShader from './compute-residual-pc.wgsl?raw';

// Maximum dimensionality supported by the residual-PC shader.
// Keeping this at 64 bounds the partial-sums buffer to ~33 MB even at N=1M.
export const RESIDUAL_PC_MAX_DIMS = 64;

const PARAMS_SIZE = 16;
const WORKGROUP_SIZE = 256;

export type ResidualPcPipeline = {
  partialPipeline: GPUComputePipeline;
  finalPipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  paramsBuffer: GPUBuffer;
};

export const createResidualPcPipeline = (device: GPUDevice): ResidualPcPipeline => {
  const module = device.createShaderModule({ label: 'compute-residual-pc', code: computeShader });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'residual-pc-bgl',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const partialPipeline = device.createComputePipeline({
    label: 'residual-pc-partial',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'reduce_partial' },
  });

  const finalPipeline = device.createComputePipeline({
    label: 'residual-pc-final',
    layout: pipelineLayout,
    compute: { module, entryPoint: 'final_reduce' },
  });

  const paramsBuffer = device.createBuffer({
    label: 'residual-pc-params',
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return { partialPipeline, finalPipeline, bindGroupLayout, paramsBuffer };
};

/**
 * Compute the first PC of the residual complement on the GPU.
 *
 * The GPU reduces N points to a p×p covariance matrix (two passes).
 * Power iteration (15 steps) runs on the CPU — p is always small (≤64).
 *
 * @param basis  p×2 column-major Float32Array already written to basisBuffer
 * @returns      p-element unit vector orthogonal to both basis columns
 */
export const computeResidualPCGpu = async (
  device: GPUDevice,
  pipeline: ResidualPcPipeline,
  dataBuffer: GPUBuffer,
  normParamsBuffer: GPUBuffer,
  basisBuffer: GPUBuffer,
  basis: Float32Array,
  numPoints: number,
  numDims: number,
): Promise<Float32Array> => {
  const d = Math.min(numDims, RESIDUAL_PC_MAX_DIMS);
  const numCov = (d * (d + 1)) / 2;
  const numTerms = d + numCov;
  const numWorkgroups = Math.ceil(numPoints / WORKGROUP_SIZE);

  const paramsData = new ArrayBuffer(PARAMS_SIZE);
  const u32 = new Uint32Array(paramsData);
  u32[0] = numPoints;
  u32[1] = d;
  u32[2] = numWorkgroups;
  u32[3] = 0;
  device.queue.writeBuffer(pipeline.paramsBuffer, 0, paramsData);

  // Ephemeral buffers — destroyed after readback
  const partialsBuffer = device.createBuffer({
    label: 'residual-pc-partials',
    size: numWorkgroups * numTerms * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const resultBuffer = device.createBuffer({
    label: 'residual-pc-result',
    size: numTerms * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: 'residual-pc-readback',
    size: numTerms * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const bindGroup = device.createBindGroup({
    label: 'residual-pc-bg',
    layout: pipeline.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: pipeline.paramsBuffer } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: normParamsBuffer } },
      { binding: 3, resource: { buffer: basisBuffer } },
      { binding: 4, resource: { buffer: partialsBuffer } },
      { binding: 5, resource: { buffer: resultBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: 'residual-pc' });

  const pass1 = encoder.beginComputePass({ label: 'residual-pc-partial' });
  pass1.setPipeline(pipeline.partialPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  const pass2 = encoder.beginComputePass({ label: 'residual-pc-final' });
  pass2.setPipeline(pipeline.finalPipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, numTerms * 4);
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const raw = new Float32Array(readbackBuffer.getMappedRange()).slice();
  readbackBuffer.unmap();

  partialsBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return powerIteration(raw, d, numDims, basis);
};

// ─── CPU power iteration ─────────────────────────────────────────────────────

/**
 * Find the top eigenvector of the residual covariance via 15 steps of power
 * iteration, orthogonalizing against the basis at each step to prevent drift.
 *
 * @param raw      GPU readback: [mean_0..mean_{d-1}, cov upper-triangle]
 * @param d        effective dims (≤ numDims, may be clamped to MAX_DIMS)
 * @param numDims  original dimension count (result is padded to this length)
 * @param basis    p×2 column-major basis (length = numDims * 2)
 */
const powerIteration = (
  raw: Float32Array,
  d: number,
  numDims: number,
  basis: Float32Array,
): Float32Array => {
  // Unpack upper-triangle into full symmetric d×d covariance
  const cov = new Float64Array(d * d);
  for (let d1 = 0; d1 < d; d1++) {
    for (let d2 = d1; d2 < d; d2++) {
      const ci = d1 * d - (d1 * (d1 - 1)) / 2 + (d2 - d1);
      const v = raw[d + ci]!;
      cov[d1 * d + d2] = v;
      cov[d2 * d + d1] = v;
    }
  }

  // Extract basis columns as Float64 (only first d entries needed)
  const bx = new Float64Array(d);
  const by = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    bx[i] = basis[i]!;
    by[i] = basis[numDims + i]!;
  }

  // Initialize to all-ones, then orthogonalize and normalize
  let vec = new Float64Array(d);
  for (let i = 0; i < d; i++) vec[i] = 1.0;
  orthogonalizeAgainstBasis(vec, bx, by, d);
  normalizeVec(vec, d);

  // 15 steps of power iteration with orthogonalization to prevent drift
  let dst = new Float64Array(d);
  for (let iter = 0; iter < 15; iter++) {
    for (let i = 0; i < d; i++) {
      let sum = 0;
      for (let j = 0; j < d; j++) sum += cov[i * d + j]! * vec[j]!;
      dst[i] = sum;
    }
    orthogonalizeAgainstBasis(dst, bx, by, d);
    normalizeVec(dst, d);
    const tmp = vec;
    vec = dst;
    dst = tmp;
  }

  // Pad to numDims if d was clamped
  const result = new Float32Array(numDims);
  for (let i = 0; i < d; i++) result[i] = vec[i]!;
  return result;
};

const orthogonalizeAgainstBasis = (
  v: Float64Array,
  bx: Float64Array,
  by: Float64Array,
  dims: number,
): void => {
  let dotX = 0;
  let dotY = 0;
  for (let d = 0; d < dims; d++) {
    dotX += v[d]! * bx[d]!;
    dotY += v[d]! * by[d]!;
  }
  for (let d = 0; d < dims; d++) {
    v[d] = v[d]! - (dotX * bx[d]! + dotY * by[d]!);
  }
};

const normalizeVec = (v: Float64Array, dims: number): void => {
  let norm = 0;
  for (let d = 0; d < dims; d++) norm += v[d]! * v[d]!;
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let d = 0; d < dims; d++) v[d] = v[d]! / norm;
  }
};
