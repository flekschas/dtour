import pointShader from '../shaders/point.wgsl?raw';
import tonemapShader from '../shaders/tonemap.wgsl?raw';

export type PointPipeline = {
  /** Additive blend pipeline — accumulates light on dark background. */
  additivePipeline: GPURenderPipeline;
  /** Normal (premultiplied-over) blend pipeline — preserves label colors. */
  normalPipeline: GPURenderPipeline;
  /** Subtractive (reverse-subtract) blend pipeline — subtracts from light background. */
  subtractivePipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
  uniformBuffer: GPUBuffer;
  cameraBuffer: GPUBuffer;
  /** Small 4-byte default buffer used when no per-point colors are loaded. */
  defaultColorBuffer: GPUBuffer;
  /** Small 4-byte default buffer used when no selection mask is set. */
  defaultSelectionBuffer: GPUBuffer;
};

// Uniform layout (64 bytes, 16-byte aligned):
//   offset  0: point_size          (f32)
//   offset  4: opacity             (f32)
//   offset  8: usePerPointColor    (f32)
//   offset 12: useSelectionMask    (f32)
//   offset 16: color               (vec4f)
//   offset 32: useSubtractive      (f32)
//   offset 36: num_points          (u32)
//   offset 40: num_dims            (u32)
//   offset 44: _pad                (u32)
//   offset 48: bias                (vec2f)
//   offset 56-63: padding (struct alignment to 16 bytes)
const UNIFORM_SIZE = 64;

// Camera layout (32 bytes — 7 fields + padding for uniform struct alignment):
//   offset  0: pan_x           (f32)
//   offset  4: pan_y           (f32)
//   offset  8: zoom            (f32)
//   offset 12: aspect          (f32)
//   offset 16: viewport_height (f32)
//   offset 20: inset_offset_y  (f32)
//   offset 24: inset_zoom      (f32)
//   offset 28-31: padding
const CAMERA_SIZE = 32;

/** Resolved style with concrete numeric values, ready for GPU upload. */
export type PointStyle = {
  pointSize: number; // NDC units, e.g. 0.01
  opacity: number; // 0-1
  color: [number, number, number]; // RGB 0-1
};

/** Style as stored in worker state — may contain 'auto' for per-view resolution. */
export type RawPointStyle = {
  pointSize: number | 'auto';
  opacity: number | 'auto';
  color: [number, number, number];
};

export type StyleFlags = {
  usePerPointColor: boolean;
  useSelectionMask: boolean;
};

export type CameraState = {
  panX: number;
  panY: number;
  zoom: number;
  aspect: number;
  viewportHeight: number;
  /** NDC-space Y offset — shifts content to center below toolbar. */
  insetOffsetY: number;
  /** Zoom multiplier — scales content to fit visible area below toolbar. */
  insetZoom: number;
};

const DEFAULT_STYLE: PointStyle = {
  pointSize: 0.012,
  opacity: 0.7,
  color: [0.25, 0.5, 0.9],
};

const DEFAULT_FLAGS: StyleFlags = {
  usePerPointColor: false,
  useSelectionMask: false,
};

const DEFAULT_CAMERA: CameraState = {
  panX: 0,
  panY: 0,
  zoom: 1,
  aspect: 1,
  viewportHeight: 1,
  insetOffsetY: 0,
  insetZoom: 1,
};

const DEFAULT_HDR_FORMAT: GPUTextureFormat = 'rgba32float';

// Pre-allocated typed arrays for per-frame uniform writes (avoids GC pressure).
const uniformBuf = new ArrayBuffer(UNIFORM_SIZE);
const uniformF32 = new Float32Array(uniformBuf);
const uniformU32 = new Uint32Array(uniformBuf);
const cameraBuf = new Float32Array(CAMERA_SIZE / 4);
const tonemapBuf = new Float32Array(4);

export const createPointPipeline = (
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
  hdrFormat: GPUTextureFormat = DEFAULT_HDR_FORMAT,
): PointPipeline => {
  const shaderModule = device.createShaderModule({
    label: 'point-shader',
    code: pointShader,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'point-bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'uniform' },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: 'read-only-storage' },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // Point pipelines target a float FBO (rgba32float or rgba16float) so additive
  // blending accumulates without clamping. A separate tonemap pass maps to canvas.

  // Additive blending — accumulate light on dark background.
  // Dense regions can exceed 1.0 in the float texture; tone mapping resolves.
  const additivePipeline = device.createRenderPipeline({
    label: 'point-pipeline-additive',
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: 'vs_main' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format: hdrFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one' },
            alpha: { srcFactor: 'zero', dstFactor: 'one' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip' },
  });

  // Normal (premultiplied-over) blending — preserves per-point label colors.
  // Values stay in [0,1]; tone mapping acts as identity.
  const normalPipeline = device.createRenderPipeline({
    label: 'point-pipeline-normal',
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: 'vs_main' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format: hdrFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip' },
  });

  // Subtractive (reverse-subtract) blending — subtracts from light background.
  // Shader outputs complement color (1-rgb)*intensity; blend computes dst - src,
  // yielding the original hue on a white/light background (Reusser).
  const subtractivePipeline = device.createRenderPipeline({
    label: 'point-pipeline-subtractive',
    layout: pipelineLayout,
    vertex: { module: shaderModule, entryPoint: 'vs_main' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format: hdrFormat,
          blend: {
            color: {
              operation: 'reverse-subtract',
              srcFactor: 'one',
              dstFactor: 'one',
            },
            alpha: { srcFactor: 'zero', dstFactor: 'one' },
          },
        },
      ],
    },
    primitive: { topology: 'triangle-strip' },
  });

  const uniformBuffer = device.createBuffer({
    label: 'point-uniforms',
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const cameraBuffer = device.createBuffer({
    label: 'camera-uniforms',
    size: CAMERA_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Small default buffers (4 bytes each) so bind groups are always valid
  const defaultColorBuffer = device.createBuffer({
    label: 'default-color-buf',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const defaultSelectionBuffer = device.createBuffer({
    label: 'default-selection-buf',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Write defaults
  writeUniforms(device, uniformBuffer, DEFAULT_STYLE, DEFAULT_FLAGS);
  writeCamera(device, cameraBuffer, DEFAULT_CAMERA);

  return {
    additivePipeline,
    normalPipeline,
    subtractivePipeline,
    bindGroupLayout,
    uniformBuffer,
    cameraBuffer,
    defaultColorBuffer,
    defaultSelectionBuffer,
  };
};

export const writeUniforms = (
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  style: PointStyle,
  flags: StyleFlags = DEFAULT_FLAGS,
  useSubtractive = false,
  numPoints = 0,
  numDims = 0,
  biasX = 0,
  biasY = 0,
): void => {
  uniformF32[0] = style.pointSize;
  uniformF32[1] = style.opacity;
  uniformF32[2] = flags.usePerPointColor ? 1.0 : 0.0;
  uniformF32[3] = flags.useSelectionMask ? 1.0 : 0.0;
  uniformF32[4] = style.color[0];
  uniformF32[5] = style.color[1];
  uniformF32[6] = style.color[2];
  uniformF32[7] = 1.0; // color.a
  uniformF32[8] = useSubtractive ? 1.0 : 0.0;
  uniformU32[9] = numPoints;
  uniformU32[10] = numDims;
  uniformU32[11] = 0; // _pad
  uniformF32[12] = biasX;
  uniformF32[13] = biasY;
  uniformF32[14] = 0; // padding
  uniformF32[15] = 0; // padding
  device.queue.writeBuffer(uniformBuffer, 0, uniformBuf);
};

export const writeCamera = (
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  camera: CameraState,
): void => {
  cameraBuf[0] = camera.panX;
  cameraBuf[1] = camera.panY;
  cameraBuf[2] = camera.zoom;
  cameraBuf[3] = camera.aspect;
  cameraBuf[4] = camera.viewportHeight;
  cameraBuf[5] = camera.insetOffsetY;
  cameraBuf[6] = camera.insetZoom;
  device.queue.writeBuffer(cameraBuffer, 0, cameraBuf);
};

export const createPointBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  dataBuffer: GPUBuffer,
  cameraBuffer: GPUBuffer,
  colorBuffer: GPUBuffer,
  selectionBuffer: GPUBuffer,
  adjBasisBuffer: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'point-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: cameraBuffer } },
      { binding: 3, resource: { buffer: colorBuffer } },
      { binding: 4, resource: { buffer: selectionBuffer } },
      { binding: 5, resource: { buffer: adjBasisBuffer } },
    ],
  });

// ─── Tonemap pipeline ────────────────────────────────────────────────────

export type TonemapPipeline = {
  pipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
  /** Shared 16-byte uniform buffer: [mode(f32), pad, pad, pad]. */
  paramsBuffer: GPUBuffer;
};

/** Tonemap mode: 0 = exponential (additive), 1 = clamp (over/subtractive). */
export const writeTonemapParams = (
  device: GPUDevice,
  paramsBuffer: GPUBuffer,
  mode: number,
): void => {
  tonemapBuf[0] = mode;
  device.queue.writeBuffer(paramsBuffer, 0, tonemapBuf);
};

export const createTonemapPipeline = (
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
): TonemapPipeline => {
  const module = device.createShaderModule({
    label: 'tonemap-shader',
    code: tonemapShader,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    label: 'tonemap-bgl',
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'unfilterable-float' },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    label: 'tonemap-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module, entryPoint: 'vs' },
    fragment: {
      module,
      entryPoint: 'fs',
      targets: [{ format: canvasFormat }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const paramsBuffer = device.createBuffer({
    label: 'tonemap-params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  return { pipeline, bindGroupLayout, paramsBuffer };
};

export const createTonemapBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  hdrTextureView: GPUTextureView,
  paramsBuffer: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'tonemap-bg',
    layout,
    entries: [
      { binding: 0, resource: hdrTextureView },
      { binding: 1, resource: { buffer: paramsBuffer } },
    ],
  });
