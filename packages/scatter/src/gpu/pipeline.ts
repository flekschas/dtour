import pointShader from '../shaders/point.wgsl?raw';
import tonemapShader from '../shaders/tonemap.wgsl?raw';

export type PointPipeline = {
  /** Additive blend pipeline — accumulates light on dark background. */
  additivePipeline: GPURenderPipeline;
  /** Normal (premultiplied-over) pipeline — standard alpha compositing for per-point colors. */
  normalPipeline: GPURenderPipeline;
  /** Subtractive (reverse-subtract) blend pipeline — subtracts from light background. */
  subtractivePipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
  uniformBuffer: GPUBuffer;
  cameraBuffer: GPUBuffer;
  /** Small 4-byte default buffer used when no colormap LUT is loaded. */
  defaultColorLutBuffer: GPUBuffer;
  /** Small 4-byte default buffer used when no selection mask is set. */
  defaultSelectionBuffer: GPUBuffer;
  /** Small 4-byte default buffer used when no categorical indices are bound. */
  defaultCatIndicesBuffer: GPUBuffer;
};

// Uniform layout (96 bytes, 16-byte aligned):
//   offset  0: point_size            (f32)
//   offset  4: opacity               (f32)
//   offset  8: color_mode            (u32) — 0=uniform, 1=continuous, 2=categorical, 3=2D colormap
//   offset 12: useSelectionMask      (f32)
//   offset 16: color                 (vec4f)
//   offset 32: useSubtractive        (f32)
//   offset 36: num_points            (u32)
//   offset 40: num_dims              (u32)
//   offset 44: max_points            (u32) — 0 = disabled
//   offset 48: bias                  (vec2f)
//   offset 56: color_column_offset   (u32)
//   offset 60: color_min             (f32)
//   offset 64: color_range           (f32)
//   offset 68: color_num_stops       (u32)
//   offset 72: bias_z                (f32) — z-axis bias (3D mode only)
//   offset 76: color_column_offset_v (u32) — 2D colormap second column
//   offset 80: color_min_v           (f32) — 2D colormap second column min
//   offset 84: color_range_v         (f32) — 2D colormap second column range
//   offset 88: color2d_map_index     (u32) — which 2D colormap (0-6)
//   offset 92-95: padding
const UNIFORM_SIZE = 96;

// Camera layout (80 bytes — 8 scalar fields + 3×3 rotation matrix padded to 3×vec4f):
//   offset  0: pan_x           (f32)
//   offset  4: pan_y           (f32)
//   offset  8: zoom            (f32)
//   offset 12: aspect          (f32)
//   offset 16: viewport_height (f32)
//   offset 20: inset_offset_y  (f32)
//   offset 24: inset_zoom      (f32)
//   offset 28: use_3d          (f32) — 0.0 = 2D, 1.0 = 3D rotation active
//   offset 32: rot_col0        (vec3f + pad)
//   offset 48: rot_col1        (vec3f + pad)
//   offset 64: rot_col2        (vec3f + pad)
const CAMERA_SIZE = 80;

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

/** Color mode: 0 = uniform color, 1 = continuous, 2 = categorical, 3 = 2D colormap. */
export type ColorMode = 0 | 1 | 2 | 3;

/** Color mapping state for the shader LUT approach. */
export type ColorState = {
  mode: ColorMode;
  columnOffset: number; // columnIndex * numPoints (continuous only)
  min: number;
  range: number;
  numStops: number; // LUT size
  // 2D colormap fields (used when mode == 3)
  columnOffsetV: number; // second column index * numPoints
  minV: number;
  rangeV: number;
  mapIndex: number; // which 2D colormap (0-6)
};

export type StyleFlags = {
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
  /** Whether 3D camera rotation is active. */
  use3d: boolean;
  /** 3×3 rotation matrix (column-major, 9 floats). null = identity. */
  rotation: Float32Array | null;
};

const DEFAULT_STYLE: PointStyle = {
  pointSize: 0.012,
  opacity: 0.7,
  color: [0.25, 0.5, 0.9],
};

const DEFAULT_FLAGS: StyleFlags = {
  useSelectionMask: false,
};

const DEFAULT_COLOR: ColorState = {
  mode: 0,
  columnOffset: 0,
  min: 0,
  range: 1,
  numStops: 0,
  columnOffsetV: 0,
  minV: 0,
  rangeV: 1,
  mapIndex: 0,
};

const DEFAULT_CAMERA: CameraState = {
  panX: 0,
  panY: 0,
  zoom: 1,
  aspect: 1,
  viewportHeight: 1,
  insetOffsetY: 0,
  insetZoom: 1,
  use3d: false,
  rotation: null,
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
      {
        binding: 6,
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

  // Normal (premultiplied-over) blending — standard alpha compositing for per-point colors.
  // Background is baked into the clear color; each point composites on top.
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
  const defaultColorLutBuffer = device.createBuffer({
    label: 'default-color-lut-buf',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const defaultSelectionBuffer = device.createBuffer({
    label: 'default-selection-buf',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const defaultCatIndicesBuffer = device.createBuffer({
    label: 'default-cat-indices-buf',
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Write defaults
  writeUniforms(device, uniformBuffer, DEFAULT_STYLE, DEFAULT_FLAGS, DEFAULT_COLOR);
  writeCamera(device, cameraBuffer, DEFAULT_CAMERA);

  return {
    additivePipeline,
    normalPipeline,
    subtractivePipeline,
    bindGroupLayout,
    uniformBuffer,
    cameraBuffer,
    defaultColorLutBuffer,
    defaultSelectionBuffer,
    defaultCatIndicesBuffer,
  };
};

export const writeUniforms = (
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  style: PointStyle,
  flags: StyleFlags = DEFAULT_FLAGS,
  colorState: ColorState = DEFAULT_COLOR,
  useSubtractive = false,
  numPoints = 0,
  numDims = 0,
  biasX = 0,
  biasY = 0,
  maxPoints = 0,
  biasZ = 0,
): void => {
  uniformF32[0] = style.pointSize;
  uniformF32[1] = style.opacity;
  uniformU32[2] = colorState.mode;
  uniformF32[3] = flags.useSelectionMask ? 1.0 : 0.0;
  uniformF32[4] = style.color[0];
  uniformF32[5] = style.color[1];
  uniformF32[6] = style.color[2];
  uniformF32[7] = 1.0; // color.a
  uniformF32[8] = useSubtractive ? 1.0 : 0.0;
  uniformU32[9] = numPoints;
  uniformU32[10] = numDims;
  uniformU32[11] = maxPoints;
  uniformF32[12] = biasX;
  uniformF32[13] = biasY;
  uniformU32[14] = colorState.columnOffset;
  uniformF32[15] = colorState.min;
  uniformF32[16] = colorState.range;
  uniformU32[17] = colorState.numStops;
  uniformF32[18] = biasZ;
  uniformU32[19] = colorState.columnOffsetV;
  uniformF32[20] = colorState.minV;
  uniformF32[21] = colorState.rangeV;
  uniformU32[22] = colorState.mapIndex;
  uniformF32[23] = 0; // padding
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
  cameraBuf[7] = camera.use3d ? 1.0 : 0.0;
  // 3×3 rotation matrix, column-major, each column padded to vec4f (16 bytes)
  const rot = camera.rotation;
  // col0 at offset 32 (index 8)
  cameraBuf[8] = rot ? rot[0]! : 1.0;
  cameraBuf[9] = rot ? rot[1]! : 0.0;
  cameraBuf[10] = rot ? rot[2]! : 0.0;
  cameraBuf[11] = 0; // pad
  // col1 at offset 48 (index 12)
  cameraBuf[12] = rot ? rot[3]! : 0.0;
  cameraBuf[13] = rot ? rot[4]! : 1.0;
  cameraBuf[14] = rot ? rot[5]! : 0.0;
  cameraBuf[15] = 0; // pad
  // col2 at offset 64 (index 16)
  cameraBuf[16] = rot ? rot[6]! : 0.0;
  cameraBuf[17] = rot ? rot[7]! : 0.0;
  cameraBuf[18] = rot ? rot[8]! : 1.0;
  cameraBuf[19] = 0; // pad
  device.queue.writeBuffer(cameraBuffer, 0, cameraBuf);
};

export const createPointBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  dataBuffer: GPUBuffer,
  cameraBuffer: GPUBuffer,
  colorLutBuffer: GPUBuffer,
  selectionBuffer: GPUBuffer,
  adjBasisBuffer: GPUBuffer,
  catIndicesBuffer: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'point-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: dataBuffer } },
      { binding: 2, resource: { buffer: cameraBuffer } },
      { binding: 3, resource: { buffer: colorLutBuffer } },
      { binding: 4, resource: { buffer: selectionBuffer } },
      { binding: 5, resource: { buffer: adjBasisBuffer } },
      { binding: 6, resource: { buffer: catIndicesBuffer } },
    ],
  });

// ─── Tonemap pipeline ────────────────────────────────────────────────────

export type TonemapPipeline = {
  pipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
  /** 4-byte uniform buffer: [mode]. */
  paramsBuffer: GPUBuffer;
};

/** Tonemap mode: 0 = exponential (additive), 1 = clamp (normal / subtractive). */
export const writeTonemapParams = (
  device: GPUDevice,
  paramsBuffer: GPUBuffer,
  mode: number,
): void => {
  tonemapBuf[0] = mode;
  device.queue.writeBuffer(paramsBuffer, 0, tonemapBuf, 0, 1);
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
    size: 4,
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
