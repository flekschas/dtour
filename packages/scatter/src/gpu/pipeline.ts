import pointShader from '../shaders/point.wgsl?raw';

export type PointPipeline = {
  pipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
  uniformBuffer: GPUBuffer;
  cameraBuffer: GPUBuffer;
};

// Uniform layout (32 bytes, 16-byte aligned):
//   offset  0: point_size (f32)
//   offset  4: opacity    (f32)
//   offset  8: _pad0      (f32)
//   offset 12: _pad1      (f32)
//   offset 16: color      (vec4f)
const UNIFORM_SIZE = 32;

// Camera layout (16 bytes):
//   offset  0: pan_x  (f32)
//   offset  4: pan_y  (f32)
//   offset  8: zoom   (f32)
//   offset 12: aspect (f32)
const CAMERA_SIZE = 16;

export type PointStyle = {
  pointSize: number; // NDC units, e.g. 0.01
  opacity: number; // 0-1
  color: [number, number, number]; // RGB 0-1
};

export type CameraState = {
  panX: number;
  panY: number;
  zoom: number;
  aspect: number;
};

const DEFAULT_STYLE: PointStyle = {
  pointSize: 0.012,
  opacity: 0.7,
  color: [0.25, 0.5, 0.9],
};

const DEFAULT_CAMERA: CameraState = {
  panX: 0,
  panY: 0,
  zoom: 1,
  aspect: 1,
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
    ],
  });

  const pipeline = device.createRenderPipeline({
    label: 'point-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [
        {
          format: canvasFormat,
          blend: {
            // Premultiplied alpha blending
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
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

  // Write defaults
  writeUniforms(device, uniformBuffer, DEFAULT_STYLE);
  writeCamera(device, cameraBuffer, DEFAULT_CAMERA);

  return { pipeline, bindGroupLayout, uniformBuffer, cameraBuffer };
};

export const writeUniforms = (
  device: GPUDevice,
  uniformBuffer: GPUBuffer,
  style: PointStyle,
): void => {
  const data = new Float32Array(8);
  data[0] = style.pointSize;
  data[1] = style.opacity;
  data[2] = 0; // pad
  data[3] = 0; // pad
  data[4] = style.color[0];
  data[5] = style.color[1];
  data[6] = style.color[2];
  data[7] = 1.0; // color.a
  device.queue.writeBuffer(uniformBuffer, 0, data);
};

export const writeCamera = (
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  camera: CameraState,
): void => {
  const data = new Float32Array(4);
  data[0] = camera.panX;
  data[1] = camera.panY;
  data[2] = camera.zoom;
  data[3] = camera.aspect;
  device.queue.writeBuffer(cameraBuffer, 0, data);
};

export const createPointBindGroup = (
  device: GPUDevice,
  layout: GPUBindGroupLayout,
  uniformBuffer: GPUBuffer,
  projectedBuffer: GPUBuffer,
  cameraBuffer: GPUBuffer,
): GPUBindGroup =>
  device.createBindGroup({
    label: 'point-bg',
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: projectedBuffer } },
      { binding: 2, resource: { buffer: cameraBuffer } },
    ],
  });
