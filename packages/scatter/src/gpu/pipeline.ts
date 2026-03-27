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

// Uniform layout (48 bytes, 16-byte aligned):
//   offset  0: point_size          (f32)
//   offset  4: opacity             (f32)
//   offset  8: usePerPointColor    (f32)
//   offset 12: useSelectionMask    (f32)
//   offset 16: color               (vec4f)
//   offset 32: useSubtractive      (f32)
//   offset 36-47: padding (struct alignment to 16 bytes)
const UNIFORM_SIZE = 48;

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

export const createPointPipeline = (
  device: GPUDevice,
  canvasFormat: GPUTextureFormat,
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
    ],
  });

  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  // Both point pipelines target rgba32float (HDR FBO) so additive blending
  // accumulates without clamping.  A separate tonemap pass maps to the canvas.
  const hdrFormat: GPUTextureFormat = 'rgba32float';

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
    primitive: { topology: 'triangle-list' },
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
    primitive: { topology: 'triangle-list' },
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
    primitive: { topology: 'triangle-list' },
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
): void => {
  const data = new Float32Array(12);
  data[0] = style.pointSize;
  data[1] = style.opacity;
  data[2] = flags.usePerPointColor ? 1.0 : 0.0;
  data[3] = flags.useSelectionMask ? 1.0 : 0.0;
  data[4] = style.color[0];
  data[5] = style.color[1];
  data[6] = style.color[2];
  data[7] = 1.0; // color.a
  data[8] = useSubtractive ? 1.0 : 0.0;
  device.queue.writeBuffer(uniformBuffer, 0, data);
};

export const writeCamera = (
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  camera: CameraState,
): void => {
  const data = new Float32Array(8);
  data[0] = camera.panX;
  data[1] = camera.panY;
  data[2] = camera.zoom;
  data[3] = camera.aspect;
  data[4] = camera.viewportHeight;
  data[5] = camera.insetOffsetY;
  data[6] = camera.insetZoom;
  device.queue.writeBuffer(cameraBuffer, 0, data);
};

export const createPointBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  projectedBuffer: GPUBuffer,
  cameraBuffer: GPUBuffer,
  colorBuffer: GPUBuffer,
  selectionBuffer: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'point-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: projectedBuffer } },
      { binding: 2, resource: { buffer: cameraBuffer } },
      { binding: 3, resource: { buffer: colorBuffer } },
      { binding: 4, resource: { buffer: selectionBuffer } },
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
  const data = new Float32Array(4);
  data[0] = mode;
  device.queue.writeBuffer(paramsBuffer, 0, data);
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
