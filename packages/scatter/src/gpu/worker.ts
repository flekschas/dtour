/// <reference lib="webworker" />

import type { DataToGpu } from '../data/messages.ts';
import { type PcaPipeline, createPcaPipeline, runPCA } from '../pca/pipeline.ts';
import { type CanvasView, configureCanvas, renderPoints, tonemapToCanvas } from '../renderer.ts';
import { computeResidualPC } from '../residual-pc.ts';
import { computeArcLengths, interpolateAtPosition } from '../tour/arc-length.ts';
import {
  type ColorPipelines,
  createColorBindGroup,
  createColorPipelines,
  writeCategoricalSelectParams,
  writeContinuousSelectParams,
} from './color-pipeline.ts';
import { initDevice } from './device.ts';
import type { GpuToMain, MainToGpu } from './messages.ts';
import {
  type CameraState,
  type ColorState,
  type PointStyle,
  type RawPointStyle,
  type StyleFlags,
  type TonemapPipeline,
  createPointBindGroup,
  createPointPipeline,
  createTonemapBindGroup,
  createTonemapPipeline,
  writeCamera,
  writeTonemapParams,
  writeUniforms,
} from './pipeline.ts';
import {
  type ProjectionResources,
  createProjectionBindGroup,
  createProjectionPipeline,
  createProjectionResources,
  dispatchProjection,
  writeProjectionParams,
} from './projection.ts';

// ─── Mutable worker state ──────────────────────────────────────────────────

type TourState = {
  bases: Float32Array[];
  arcLengths: Float32Array;
  dims: number;
  position: number;
  interpolatedBasis: Float32Array;
};

type GpuState = {
  device: GPUDevice;
  views: CanvasView[];
  // Render pipeline
  pointPipeline: ReturnType<typeof createPointPipeline>;
  renderBindGroup: GPUBindGroup | null;
  // Compute pipelines (projection kept for on-demand lasso use only)
  projectionPipeline: ReturnType<typeof createProjectionPipeline>;
  projectionResources: ProjectionResources | null;
  projectionBindGroup: GPUBindGroup | null;
  colorPipelines: ColorPipelines;
  // Data
  numPoints: number;
  numDims: number;
  // Categorical index buffers on GPU, keyed by column name
  categoricalBuffers: Map<string, GPUBuffer>;
  // Tour
  tour: TourState | null;
  // Style
  style: RawPointStyle;
  styleFlags: StyleFlags;
  camera: CameraState;
  // Color mapping — tiny LUT + uniforms (replaces N-element color buffer)
  colorLutBuffer: GPUBuffer | null;
  colorState: ColorState;
  activeCatColumnName: string | null;
  // Selection
  selectionBuffer: GPUBuffer | null;
  // Sticky direct basis — when set, renderMainView/renderAllViews use this
  // for the main view instead of tour interpolation. Cleared by setTourPosition.
  directBasis: Float32Array | null;
  // Background clear color (RGB 0–1)
  backgroundColor: [number, number, number];
  // Device pixel ratio — used by auto-style for minimum visible point size
  dpr: number;
  // HDR rendering — points render to float FBO, then tone-map to canvas
  hdrFormat: GPUTextureFormat;
  tonemapPipeline: TonemapPipeline;
  hdrTextures: (GPUTexture | null)[];
  hdrTextureViews: (GPUTextureView | null)[];
  tonemapBindGroups: (GPUBindGroup | null)[];
  // Vertex-shader decimation: 0 = disabled (render all)
  maxPoints: number;
  // Inline projection — adjusted basis buffer + per-dim norm params
  adjBasisBuffer: GPUBuffer | null;
  normMins: Float32Array | null;
  normRanges: Float32Array | null;
  // 3D camera rotation mode
  is3dActive: boolean;
  rotation3d: Float32Array | null; // 9-element 3×3 column-major
  residualPC: Float32Array | null; // p-element 3rd basis vector
  // CPU-side copy of column-major data for residual PC computation
  cpuData: Float32Array | null;
};

let state: GpuState | null = null;

// Messages received before init completes are buffered here and replayed
// in order once state is ready. Without this, the async initDevice() causes
// the onmessage handler to yield, and subsequent messages hit the
// `if (!state) return` guard and are silently dropped.
let pendingMessages: MainToGpu[] | null = [];

// Lazy-created on first computePCA call
let pcaPipeline: PcaPipeline | null = null;

const postMain = (msg: GpuToMain, transfers?: Transferable[]): void => {
  self.postMessage(msg, transfers ?? []);
};

// ─── Auto-style (Reusser color budget) ───────────────────────────────────

const COLOR_BUDGET = 0.5;

/** Compute ideal point size (NDC) and opacity for a given canvas and point count.
 *  All dimensions are physical pixels; pass dpr=1 when canvas sizes are already physical. */
const computeAutoStyle = (
  rowCount: number,
  canvasWidth: number,
  canvasHeight: number,
  dpr: number,
): { pointSize: number; opacity: number } => {
  if (rowCount === 0 || canvasWidth === 0 || canvasHeight === 0) {
    return { pointSize: 0.012, opacity: 0.7 };
  }

  const physW = canvasWidth * dpr;
  const physH = canvasHeight * dpr;
  const totalBudget = physW * physH * COLOR_BUDGET;
  const perPoint = totalBudget / rowCount;
  const idealRadius = Math.sqrt(perPoint / Math.PI);
  const minRadius = dpr; // 1 CSS pixel

  let radius: number;
  let opacity: number;

  if (idealRadius >= minRadius) {
    radius = idealRadius;
    opacity = 1.0;
  } else {
    radius = minRadius;
    opacity = Math.max(0.01, perPoint / (Math.PI * minRadius * minRadius));
  }

  const pointSize = (2 * radius) / physH;
  return { pointSize, opacity };
};

/** Convert a CSS-pixel diameter to the NDC convention used by the shader. */
const cssToNdc = (cssPx: number, viewIndex: number): number => {
  const { views, dpr } = state!;
  const physH = views[viewIndex]!.canvas.height;
  return (cssPx * dpr) / physH;
};

/** Resolve 'auto' point size/opacity for a specific view's canvas. */
const resolveStyleForView = (viewIndex: number): PointStyle => {
  const { style, numPoints, views, dpr } = state!;

  // Fast path: no 'auto' values — still need px→NDC conversion for pointSize
  if (style.pointSize !== 'auto' && style.opacity !== 'auto') {
    return {
      pointSize: cssToNdc(style.pointSize, viewIndex),
      opacity: style.opacity,
      color: style.color,
    };
  }

  // Canvas dimensions are physical pixels; computeAutoStyle expects CSS pixels + dpr
  const canvas = views[viewIndex]!.canvas;
  const auto = computeAutoStyle(numPoints, canvas.width / dpr, canvas.height / dpr, dpr);

  return {
    pointSize: style.pointSize === 'auto' ? auto.pointSize : cssToNdc(style.pointSize, viewIndex),
    opacity: style.opacity === 'auto' ? auto.opacity : style.opacity,
    color: style.color,
  };
};

// ─── HDR texture management ──────────────────────────────────────────────

/** Ensure HDR texture exists and matches canvas size. */
const ensureHdrTexture = (viewIndex: number): void => {
  const { device, views, tonemapPipeline: tp, hdrTextures } = state!;
  const canvas = views[viewIndex]!.canvas;
  const w = canvas.width;
  const h = canvas.height;

  const existing = hdrTextures[viewIndex];
  if (existing && existing.width === w && existing.height === h) return;

  existing?.destroy();
  const hdrTex = device.createTexture({
    label: `hdr-${viewIndex}`,
    size: [w, h],
    format: state!.hdrFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  const hdrView = hdrTex.createView();
  state!.hdrTextures[viewIndex] = hdrTex;
  state!.hdrTextureViews[viewIndex] = hdrView;

  state!.tonemapBindGroups[viewIndex] = createTonemapBindGroup(
    device,
    tp.bindGroupLayout,
    hdrView,
    tp.paramsBuffer,
  );
};

// ─── Buffer helpers ───────────────────────────────────────────────────────

/** Byte size for a bit-packed selection buffer (1 bit per point, packed into u32s). */
const selectionBufferSize = (numPoints: number): number => Math.ceil(numPoints / 32) * 4;

/** Ensure the selection buffer exists at the right size, creating if needed. */
const ensureSelectionBuffer = (): GPUBuffer => {
  if (!state) throw new Error('state not initialized');
  const size = selectionBufferSize(state.numPoints);
  if (!state.selectionBuffer || state.selectionBuffer.size !== size) {
    state.selectionBuffer?.destroy();
    state.selectionBuffer = state.device.createBuffer({
      label: 'selection-mask',
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }
  return state.selectionBuffer;
};

// ─── Inline projection helpers ────────────────────────────────────────────

/** Viewport scale applied to projected coordinates (maps [-0.5,0.5] → [-1,1]). */
const VIEWPORT_SCALE = 2.0;

/** Pre-allocated working buffer for adjusted basis weights (reused across frames). */
let adjBasisWeights: Float32Array | null = null;

/**
 * Fold normalization into the basis so the vertex shader inner loop is just:
 *   x += raw * adjBasis[d];  y += raw * adjBasis[dims + d];
 *
 * adjBasis[d]       = basis[d]       / range[d] * VIEWPORT_SCALE
 * adjBasis[dims+d]  = basis[dims+d]  / range[d] * VIEWPORT_SCALE
 * biasX = Σ ((-min[d]/range[d] - 0.5) * basis[d])       * VIEWPORT_SCALE
 * biasY = Σ ((-min[d]/range[d] - 0.5) * basis[dims+d])  * VIEWPORT_SCALE
 *
 * This is algebraically equivalent to compute-projection.wgsl which does
 * per-point normalization on GPU. Both yield: ((raw-min)/range - 0.5) * basis * VS.
 * If you change the math here, update compute-projection.wgsl to match.
 */
/**
 * Fold normalization into basis. When `cols` is 3, also computes z-axis weights.
 * The `basis` layout for 3 columns: [x0..xp-1, y0..yp-1, z0..zp-1].
 */
const computeAdjustedBasis = (
  basis: Float32Array,
  mins: Float32Array,
  ranges: Float32Array,
  dims: number,
  cols = 2,
): { weights: Float32Array; biasX: number; biasY: number; biasZ: number } => {
  const needed = dims * cols;
  if (!adjBasisWeights || adjBasisWeights.length !== needed) {
    adjBasisWeights = new Float32Array(needed);
  }

  let biasX = 0;
  let biasY = 0;
  let biasZ = 0;

  for (let d = 0; d < dims; d++) {
    const range = Math.max(ranges[d]!, 1e-6);
    const bx = basis[d]!;
    const by = basis[dims + d]!;
    const invRange = VIEWPORT_SCALE / range;

    adjBasisWeights[d] = bx * invRange;
    adjBasisWeights[dims + d] = by * invRange;

    const normOffset = -mins[d]! / range - 0.5;
    biasX += normOffset * bx * VIEWPORT_SCALE;
    biasY += normOffset * by * VIEWPORT_SCALE;

    if (cols === 3) {
      const bz = basis[2 * dims + d]!;
      adjBasisWeights[2 * dims + d] = bz * invRange;
      biasZ += normOffset * bz * VIEWPORT_SCALE;
    }
  }

  return { weights: adjBasisWeights, biasX, biasY, biasZ };
};

// ─── Worker-driven playback ───────────────────────────────────────────────

type PlaybackState = {
  speed: number;
  direction: 1 | -1;
  prevTime: number | null;
  rafId: number;
};

let playbackState: PlaybackState | null = null;
let lastBroadcastTime = 0;
/** Throttle position broadcasts to main thread (~30fps). */
const BROADCAST_INTERVAL_MS = 33;
/** GPU backpressure — skip frames when the GPU can't keep up. */
let gpuReady = true;
/** When true, a render was skipped due to GPU backpressure and should be retried. */
let renderPending = false;

/** Submit a render if the GPU is ready; otherwise mark it pending for later. */
const throttledRenderMainView = (): void => {
  if (!gpuReady) {
    renderPending = true;
    return;
  }
  gpuReady = false;
  renderPending = false;
  renderMainView();
  state!.device.queue.onSubmittedWorkDone().then(() => {
    gpuReady = true;
    // If a render was deferred while the GPU was busy, do it now
    if (renderPending && state) {
      renderPending = false;
      throttledRenderMainView();
    }
  });
};

const playbackTick = (time: number): void => {
  if (!playbackState || !state?.tour) return;

  if (playbackState.prevTime !== null) {
    const dt = (time - playbackState.prevTime) / 1000;
    // Full tour cycle = 20s at speed=1
    const delta = (dt * playbackState.speed * playbackState.direction) / 20;
    let next = state.tour.position + delta;
    next = next - Math.floor(next);
    state.tour.position = next;
  }
  playbackState.prevTime = time;

  state.directBasis = null;
  throttledRenderMainView();

  // Throttle position broadcasts to main thread for UI sync
  if (time - lastBroadcastTime >= BROADCAST_INTERVAL_MS) {
    postMain({ type: 'playbackTick', position: state.tour.position });
    lastBroadcastTime = time;
  }

  playbackState.rafId = requestAnimationFrame(playbackTick);
};

const startPlayback = (speed: number, direction: 1 | -1): void => {
  if (playbackState) {
    // Already playing — just update speed/direction
    playbackState.speed = speed;
    playbackState.direction = direction;
  } else {
    gpuReady = true;
    playbackState = { speed, direction, prevTime: null, rafId: 0 };
    playbackState.rafId = requestAnimationFrame(playbackTick);
  }
};

const stopPlayback = (): void => {
  if (playbackState) {
    cancelAnimationFrame(playbackState.rafId);
    // Post final position so main thread is in sync
    if (state?.tour) {
      postMain({ type: 'playbackTick', position: state.tour.position });
    }
    playbackState = null;
  }
};

// ─── Render helpers ───────────────────────────────────────────────────────

/** Compute adjusted basis, upload uniforms, and render one view (no compute pass). */
const renderView = (
  basis: Float32Array,
  viewIndex: number,
  camera: CameraState,
  renBindGroup: GPUBindGroup,
): void => {
  if (
    !state ||
    !state.projectionResources ||
    !state.adjBasisBuffer ||
    !state.normMins ||
    !state.normRanges
  )
    return;

  const { device, numPoints, numDims } = state;

  // Build the effective basis: p×2 normally, p×3 when 3D is active
  const is3d = state.is3dActive && state.residualPC;
  const cols = is3d ? 3 : 2;
  let effectiveBasis: Float32Array;
  if (is3d) {
    // Combine the p×2 basis with the residual PC as the 3rd column
    effectiveBasis = new Float32Array(numDims * 3);
    effectiveBasis.set(basis.subarray(0, numDims * 2));
    effectiveBasis.set(state.residualPC!, numDims * 2);
  } else {
    effectiveBasis = basis;
  }

  // Fold normalization into basis on CPU (trivial for typical dim counts)
  const { weights, biasX, biasY, biasZ } = computeAdjustedBasis(
    effectiveBasis,
    state.normMins,
    state.normRanges,
    numDims,
    cols,
  );

  // Upload adjusted basis weights (ensure buffer is large enough)
  const neededBytes = numDims * cols * 4;
  if (state.adjBasisBuffer.size < neededBytes) {
    state.adjBasisBuffer.destroy();
    state.adjBasisBuffer = device.createBuffer({
      label: 'adj-basis',
      size: neededBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    rebuildBindGroups();
  }
  device.queue.writeBuffer(state.adjBasisBuffer, 0, weights as Float32Array<ArrayBuffer>);

  // Resolve per-view style (handles 'auto' → concrete values for this canvas size)
  const resolved = resolveStyleForView(viewIndex);

  // Select blend pipeline and tonemap mode based on coloring and background luminance.
  // Per-point colors → normal (premultiplied-over), uniform → additive (dark) or subtractive (light).
  const bg = state.backgroundColor;
  const bgLuminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2];
  const useNormal = state.colorState.mode > 0;
  const useSubtractive = !useNormal && bgLuminance > 0.5;

  let activePipeline: GPURenderPipeline;
  let tonemapMode: number;
  if (useNormal) {
    activePipeline = state.pointPipeline.normalPipeline;
    tonemapMode = 1; // clamp (over-compositing stays in [0,1])
  } else if (useSubtractive) {
    activePipeline = state.pointPipeline.subtractivePipeline;
    tonemapMode = 1; // clamp
  } else {
    activePipeline = state.pointPipeline.additivePipeline;
    tonemapMode = 0; // exponential
  }

  writeUniforms(
    device,
    state.pointPipeline.uniformBuffer,
    resolved,
    state.styleFlags,
    state.colorState,
    useSubtractive,
    numPoints,
    numDims,
    biasX,
    biasY,
    state.maxPoints,
    biasZ,
  );
  writeTonemapParams(device, state.tonemapPipeline.paramsBuffer, tonemapMode);

  // Write the caller-supplied camera (with aspect + viewport height from the target canvas)
  const canvas = state.views[viewIndex]!.canvas;
  const aspect = canvas.width / canvas.height || 1;
  writeCamera(device, state.pointPipeline.cameraBuffer, {
    ...camera,
    aspect,
    viewportHeight: canvas.height,
  });

  // Ensure HDR texture matches canvas size (lazy create/resize)
  ensureHdrTexture(viewIndex);

  // Two-pass in a single encoder: points → float FBO, then tone-map → canvas
  const encoder = device.createCommandEncoder({ label: 'render-frame' });
  renderPoints(
    encoder,
    state.hdrTextureViews[viewIndex]!,
    activePipeline,
    renBindGroup,
    numPoints,
    bg,
  );
  tonemapToCanvas(
    encoder,
    state.views[viewIndex]!,
    state.tonemapPipeline.pipeline,
    state.tonemapBindGroups[viewIndex]!,
  );
  device.queue.submit([encoder.finish()]);
};

// ─── Deferred preview rendering ───────────────────────────────────────────
// Preview rendering is coalesced via setTimeout(0) so rapid-fire state
// updates (setBases, setStyle, render, …) only produce a single
// getCurrentTexture + submit per preview canvas, avoiding expired-texture
// issues that can silently drop frames on some WebGPU implementations.

let previewRenderPending = false;

/** Render a single preview view. No ephemeral buffers needed — inline
 *  projection reuses the shared adjBasisBuffer (written before each submit). */
const renderOnePreview = (viewIndex: number, basis: Float32Array): void => {
  if (!state?.projectionResources || !state.renderBindGroup) return;

  renderView(
    basis,
    viewIndex,
    {
      panX: 0,
      panY: 0,
      zoom: 1,
      aspect: 1,
      viewportHeight: 1,
      insetOffsetY: 0,
      insetZoom: 1,
      use3d: false,
      rotation: null,
    },
    state.renderBindGroup,
  );
  postMain({ type: 'rendered', viewIndex });
};

/** Render all preview views synchronously.
 *  Each preview targets a different canvas, so no texture conflicts. */
const renderPreviewViews = (): void => {
  if (!state?.projectionResources || !state.tour) return;

  const { views, tour } = state;
  const count = Math.min(views.length - 1, tour.bases.length);

  for (let pi = 0; pi < count; pi++) {
    renderOnePreview(pi + 1, tour.bases[pi]!);
  }
};

/** Schedule preview rendering for the next macrotask (coalesces rapid updates). */
const schedulePreviewRender = (): void => {
  if (previewRenderPending) return;
  previewRenderPending = true;
  setTimeout(() => {
    previewRenderPending = false;
    renderPreviewViews();
  }, 0);
};

/** Render all views: main immediately, previews deferred to avoid texture churn. */
const renderAllViews = (): void => {
  if (!state?.projectionResources) return;
  renderMainView();
  schedulePreviewRender();
};

/** Re-render only the main view at the current tour position. */
const renderMainView = (): void => {
  if (!state?.projectionResources || !state.renderBindGroup) return;

  if (state.directBasis) {
    renderView(state.directBasis, 0, state.camera, state.renderBindGroup);
    postMain({ type: 'rendered', viewIndex: 0 });
    return;
  }

  if (!state.tour) return;
  const { tour } = state;
  interpolateAtPosition(
    tour.interpolatedBasis,
    tour.bases,
    tour.arcLengths,
    tour.dims,
    tour.position,
  );
  renderView(tour.interpolatedBasis, 0, state.camera, state.renderBindGroup);
  postMain({ type: 'rendered', viewIndex: 0 });
};

// ─── Build bind groups after data + pipeline are ready ────────────────────

const rebuildBindGroups = (): void => {
  if (!state || !state.projectionResources || !state.adjBasisBuffer) return;

  const { device, projectionPipeline, pointPipeline, projectionResources } = state;
  const colorLutBuf = state.colorLutBuffer ?? pointPipeline.defaultColorLutBuffer;
  const selBuf = state.selectionBuffer ?? pointPipeline.defaultSelectionBuffer;

  // Categorical index buffer — only bound when categorical coloring is active
  let catBuf = pointPipeline.defaultCatIndicesBuffer;
  if (state.colorState.mode === 2 && state.activeCatColumnName) {
    catBuf = state.categoricalBuffers.get(state.activeCatColumnName) ?? catBuf;
  }

  // Projection bind group — kept for on-demand lasso readback
  state.projectionBindGroup = createProjectionBindGroup(
    device,
    projectionPipeline.bindGroupLayout,
    projectionPipeline.paramsBuffer,
    projectionResources,
  );

  // Render bind group — uses raw data + adjusted basis (inline projection)
  state.renderBindGroup = createPointBindGroup(
    device,
    pointPipeline.bindGroupLayout,
    pointPipeline.uniformBuffer,
    projectionResources.dataBuffer,
    pointPipeline.cameraBuffer,
    colorLutBuf,
    selBuf,
    state.adjBasisBuffer,
    catBuf,
  );
};

/** After color LUT/state changes, rebuild render bind groups and re-render. */
const applyColorUpdate = (): void => {
  if (!state) return;
  rebuildBindGroups();
  if ((state.tour || state.directBasis) && state.projectionResources) {
    renderAllViews();
  }
};

const applySelectionUpdate = (): void => {
  if (!state) return;
  state.styleFlags.useSelectionMask = true;
  rebuildBindGroups();
  if ((state.tour || state.directBasis) && state.projectionResources) {
    renderAllViews();
  }
};

// ─── Data from Data Worker ─────────────────────────────────────────────────

// Track the current dataset version to discard stale color/selection messages.
let currentDataVersion = 0;

const onDataMessage = (event: MessageEvent<DataToGpu>): void => {
  if (!state) return;
  const { device } = state;

  // ── Continuous color mapping (upload tiny LUT + set uniforms) ──
  if (event.data.type === 'setColorContinuous') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { columnIndex, min, range, colormap } = event.data;

    // Upload the small colormap LUT (~25 entries = ~100 bytes)
    state.colorLutBuffer?.destroy();
    state.colorLutBuffer = device.createBuffer({
      label: 'color-lut',
      size: colormap.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(state.colorLutBuffer, 0, colormap as Uint32Array<ArrayBuffer>);

    state.colorState = {
      mode: 1,
      columnOffset: columnIndex * state.numPoints,
      min,
      range,
      numStops: colormap.length,
    };
    state.activeCatColumnName = null;

    applyColorUpdate();
    return;
  }

  // ── Categorical color mapping (upload tiny palette LUT + bind cat indices) ──
  if (event.data.type === 'setColorCategorical') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { catColumnName, palette } = event.data;
    if (!state.categoricalBuffers.has(catColumnName)) return;

    // Upload the small palette LUT
    state.colorLutBuffer?.destroy();
    state.colorLutBuffer = device.createBuffer({
      label: 'color-lut',
      size: palette.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(state.colorLutBuffer, 0, palette as Uint32Array<ArrayBuffer>);

    state.colorState = {
      mode: 2,
      columnOffset: 0,
      min: 0,
      range: 1,
      numStops: palette.length,
    };
    state.activeCatColumnName = catColumnName;

    applyColorUpdate();
    return;
  }

  // ── Continuous selection (GPU compute) ──
  if (event.data.type === 'selectContinuous') {
    if (event.data.dataVersion !== currentDataVersion) return;
    if (!state.projectionResources) return;

    const { columnIndex, ranges } = event.data;
    const { numPoints, colorPipelines } = state;

    const selBuf = ensureSelectionBuffer();

    const rangesBuf = device.createBuffer({
      label: 'select-ranges',
      size: Math.max(ranges.byteLength, 4), // min 4 bytes for empty
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(rangesBuf, 0, ranges as Float32Array<ArrayBuffer>);

    writeContinuousSelectParams(
      device,
      colorPipelines.paramsBuffer,
      numPoints,
      columnIndex * numPoints,
      ranges.length / 2,
    );

    const bg = createColorBindGroup(
      device,
      colorPipelines.bindGroupLayout,
      state.projectionResources.dataBuffer,
      rangesBuf,
      selBuf,
      colorPipelines.paramsBuffer,
    );

    // Clear mask (atomicOr can only set bits) then dispatch compute
    const encoder = device.createCommandEncoder({ label: 'select-continuous' });
    encoder.clearBuffer(selBuf, 0, selBuf.size);
    const pass = encoder.beginComputePass({ label: 'select-continuous' });
    pass.setPipeline(colorPipelines.selectContinuous);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(numPoints / 256));
    pass.end();
    device.queue.submit([encoder.finish()]);
    rangesBuf.destroy();

    applySelectionUpdate();
    return;
  }

  // ── Categorical selection (GPU compute) ──
  if (event.data.type === 'selectCategorical') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { catColumnName, selectedLabels } = event.data;
    const indexBuf = state.categoricalBuffers.get(catColumnName);
    if (!indexBuf) return;

    const { numPoints, colorPipelines } = state;
    const selBuf = ensureSelectionBuffer();

    const selLabelsBuf = device.createBuffer({
      label: 'selected-labels',
      size: Math.max(selectedLabels.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(selLabelsBuf, 0, selectedLabels as Uint32Array<ArrayBuffer>);

    writeCategoricalSelectParams(
      device,
      colorPipelines.paramsBuffer,
      numPoints,
      selectedLabels.length,
    );

    const bg = createColorBindGroup(
      device,
      colorPipelines.bindGroupLayout,
      indexBuf,
      selLabelsBuf,
      selBuf,
      colorPipelines.paramsBuffer,
    );

    // Clear mask (atomicOr can only set bits) then dispatch compute
    const encoder = device.createCommandEncoder({ label: 'select-categorical' });
    encoder.clearBuffer(selBuf, 0, selBuf.size);
    const pass = encoder.beginComputePass({ label: 'select-categorical' });
    pass.setPipeline(colorPipelines.selectCategorical);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(numPoints / 256));
    pass.end();
    device.queue.submit([encoder.finish()]);
    selLabelsBuf.destroy();

    applySelectionUpdate();
    return;
  }

  // ── Dataset load ──
  if (event.data.type !== 'data') return;

  const { dataVersion, dims, rows, buffers, mins, ranges, categoricalColumns } = event.data;
  currentDataVersion = dataVersion;

  if (dims < 2 || rows === 0 || buffers.length < 2) {
    postMain({ type: 'error', message: 'Need at least 2 dimensions to render' });
    return;
  }

  // Clean up previous resources
  if (state.projectionResources) {
    state.projectionResources.dataBuffer.destroy();
    state.projectionResources.normParamsBuffer.destroy();
    state.projectionResources.basisBuffer.destroy();
    state.projectionResources.projectedBuffer.destroy();
  }
  if (state.adjBasisBuffer) {
    state.adjBasisBuffer.destroy();
    state.adjBasisBuffer = null;
  }
  // Clean up previous color LUT / selection buffers
  if (state.colorLutBuffer) {
    state.colorLutBuffer.destroy();
    state.colorLutBuffer = null;
  }
  if (state.selectionBuffer) {
    state.selectionBuffer.destroy();
    state.selectionBuffer = null;
  }
  // Clean up previous categorical buffers
  for (const buf of state.categoricalBuffers.values()) {
    buf.destroy();
  }
  state.categoricalBuffers.clear();

  state.styleFlags = { useSelectionMask: false };
  state.colorState = { mode: 0, columnOffset: 0, min: 0, range: 1, numStops: 0 };
  state.activeCatColumnName = null;
  state.is3dActive = false;
  state.rotation3d = null;
  state.residualPC = null;
  state.camera.use3d = false;
  state.camera.rotation = null;

  const { projectionPipeline } = state;

  // Create new projection resources
  const res = createProjectionResources(device, projectionPipeline, rows, dims);

  // Upload each column contiguously into the concatenated data buffer
  // Also retain a CPU-side copy for residual PC computation (3D mode)
  const cpuData = new Float32Array(dims * rows);
  for (let d = 0; d < dims; d++) {
    device.queue.writeBuffer(
      res.dataBuffer,
      d * rows * 4,
      buffers[d]! as Float32Array<ArrayBuffer>,
    );
    cpuData.set(buffers[d]!, d * rows);
  }
  state.cpuData = cpuData;

  // Upload norm params: [min, range] pairs as vec2f array
  const normData = new Float32Array(dims * 2);
  for (let d = 0; d < dims; d++) {
    normData[d * 2] = mins[d]!;
    normData[d * 2 + 1] = ranges[d]!;
  }
  device.queue.writeBuffer(res.normParamsBuffer, 0, normData);

  // Upload categorical index buffers
  for (const cat of categoricalColumns) {
    const buf = device.createBuffer({
      label: `cat-indices-${cat.name}`,
      size: cat.indices.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, cat.indices as Uint32Array<ArrayBuffer>);
    state.categoricalBuffers.set(cat.name, buf);
  }

  // Write projection params (used for on-demand lasso compute dispatch)
  writeProjectionParams(device, projectionPipeline.paramsBuffer, rows, dims, 2.0);

  // Create adjusted basis buffer for inline vertex projection
  state.adjBasisBuffer = device.createBuffer({
    label: 'adj-basis',
    size: dims * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Store norm params on CPU for per-frame basis adjustment
  state.normMins = new Float32Array(mins);
  state.normRanges = new Float32Array(ranges);

  state.projectionResources = { ...res, bindGroup: null as unknown as GPUBindGroup };
  state.numPoints = rows;
  state.numDims = dims;

  rebuildBindGroups();

  // If we already have tour bases, render with them
  if (state.tour && state.tour.dims === dims) {
    renderAllViews();
  } else {
    // Create default bases covering ALL preview views so none stay blank.
    // Each basis projects a consecutive pair of dimensions.
    const count = Math.max(1, state.views.length - 1);
    const defaultBases: Float32Array[] = [];
    for (let k = 0; k < count; k++) {
      const basis = new Float32Array(dims * 2);
      const d = Math.floor((k / count) * dims);
      basis[d] = 1; // dim d → x
      basis[dims + ((d + 1) % dims)] = 1; // dim d+1 → y
      defaultBases.push(basis);
    }

    state.tour = {
      bases: defaultBases,
      arcLengths: computeArcLengths(defaultBases, dims),
      dims,
      position: 0,
      interpolatedBasis: new Float32Array(dims * 2),
    };
    renderAllViews();
  }
};

// ─── Main thread messages ──────────────────────────────────────────────────

/** Process a single non-init message. Requires state to be initialized. */
const handleMessage = (msg: MainToGpu): void => {
  if (!state) return;

  if (msg.type === 'setBases') {
    const { bases } = msg;
    if (bases.length === 0) return;
    const dims = bases[0]!.length / 2;

    const arcLengths = computeArcLengths(bases, dims);

    state.tour = {
      bases,
      arcLengths,
      dims,
      position: state.tour?.position ?? 0,
      interpolatedBasis: new Float32Array(dims * 2),
    };

    if (state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setTourPosition') {
    if (!state.tour) return;
    state.tour.position = msg.position;
    state.directBasis = null; // Back in tour mode — use interpolation
    throttledRenderMainView();
    return;
  }

  if (msg.type === 'setStyle') {
    state.style = { pointSize: msg.pointSize, opacity: msg.opacity, color: msg.color };
    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setCamera') {
    state.camera = {
      panX: msg.pan[0],
      panY: msg.pan[1],
      zoom: msg.zoom,
      aspect: state.camera.aspect,
      viewportHeight: state.camera.viewportHeight,
      insetOffsetY: msg.insetOffsetY,
      insetZoom: msg.insetZoom,
      use3d: state.camera.use3d,
      rotation: state.camera.rotation,
    };
    renderMainView();
    return;
  }

  if (msg.type === 'resize') {
    if (msg.dpr !== undefined) {
      state.dpr = msg.dpr;
    }
    const view = state.views[msg.viewIndex];
    if (view) {
      view.canvas.width = msg.width;
      view.canvas.height = msg.height;
      if ((state.tour || state.directBasis) && state.projectionResources) {
        if (msg.viewIndex === 0) {
          renderMainView();
        } else if (state.tour) {
          const basisIndex = msg.viewIndex - 1;
          if (basisIndex < state.tour.bases.length) {
            renderOnePreview(msg.viewIndex, state.tour.bases[basisIndex]!);
          }
        }
      }
    }
    return;
  }

  if (msg.type === 'render') {
    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setDirectBasis') {
    if (!state.projectionResources || !state.renderBindGroup) return;
    state.directBasis = msg.basis;
    // Recompute residual PC when 3D mode is active and basis changes
    if (state.is3dActive && state.cpuData && state.normMins && state.normRanges) {
      state.residualPC = computeResidualPC(
        msg.basis,
        state.cpuData,
        state.normMins,
        state.normRanges,
        state.numDims,
        state.numPoints,
      );
      postMain({ type: 'residualPC', residualPC: new Float32Array(state.residualPC) });
    }
    renderView(msg.basis, 0, state.camera, state.renderBindGroup);
    postMain({ type: 'rendered', viewIndex: 0 });
    return;
  }

  if (msg.type === 'clearColors') {
    if (state.colorLutBuffer) {
      state.colorLutBuffer.destroy();
      state.colorLutBuffer = null;
    }
    state.colorState = { mode: 0, columnOffset: 0, min: 0, range: 1, numStops: 0 };
    state.activeCatColumnName = null;
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setBackgroundColor') {
    state.backgroundColor = msg.color;
    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setMaxPoints') {
    state.maxPoints = msg.maxPoints;
    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setSelectionMask') {
    const { mask } = msg;

    if (state.selectionBuffer) {
      state.selectionBuffer.destroy();
    }

    state.selectionBuffer = state.device.createBuffer({
      label: 'selection-mask',
      size: mask.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    state.device.queue.writeBuffer(state.selectionBuffer, 0, mask as Uint32Array<ArrayBuffer>);

    state.styleFlags.useSelectionMask = true;
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'clearSelectionMask') {
    if (state.selectionBuffer) {
      state.selectionBuffer.destroy();
      state.selectionBuffer = null;
    }
    state.styleFlags.useSelectionMask = false;
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'enable3d') {
    if (!state.cpuData || !state.normMins || !state.normRanges) return;
    // Get current basis (direct or interpolated)
    const basis = state.directBasis ?? state.tour?.interpolatedBasis;
    if (!basis) return;
    state.is3dActive = true;
    state.residualPC = computeResidualPC(
      basis,
      state.cpuData,
      state.normMins,
      state.normRanges,
      state.numDims,
      state.numPoints,
    );
    postMain({ type: 'residualPC', residualPC: new Float32Array(state.residualPC) });
    state.camera.use3d = true;
    // Resize adj basis buffer for 3 columns
    if (state.adjBasisBuffer) {
      const needed = state.numDims * 3 * 4;
      if (state.adjBasisBuffer.size < needed) {
        state.adjBasisBuffer.destroy();
        state.adjBasisBuffer = state.device.createBuffer({
          label: 'adj-basis',
          size: needed,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        rebuildBindGroups();
      }
    }
    renderMainView();
    return;
  }

  if (msg.type === 'disable3d') {
    state.is3dActive = false;
    state.residualPC = null;
    state.rotation3d = null;
    state.camera.use3d = false;
    state.camera.rotation = null;
    renderMainView();
    return;
  }

  if (msg.type === 'set3dRotation') {
    state.rotation3d = msg.matrix;
    state.camera.rotation = msg.matrix;
    renderMainView();
    return;
  }

  if (msg.type === 'startPlayback') {
    if (!state.tour) return;
    startPlayback(msg.speed, msg.direction);
    return;
  }

  if (msg.type === 'stopPlayback') {
    stopPlayback();
    return;
  }

  if (msg.type === 'computePCA') {
    if (!state.projectionResources || state.numDims < 2) {
      postMain({ type: 'error', message: 'No data loaded for PCA' });
      return;
    }

    const { device, projectionResources, numPoints, numDims } = state;

    if (!pcaPipeline) {
      pcaPipeline = createPcaPipeline(device);
    }

    runPCA(
      device,
      pcaPipeline,
      projectionResources.dataBuffer,
      projectionResources.normParamsBuffer,
      numPoints,
      numDims,
    )
      .then(({ eigenvalues, eigenvectors }) => {
        postMain({
          type: 'pcaResult',
          eigenvectors,
          eigenvalues,
          numDims: eigenvectors.length,
        });
      })
      .catch((err) => {
        postMain({
          type: 'error',
          message: `PCA failed: ${err instanceof Error ? err.message : String(err)}`,
        });
      });
    return;
  }

  if (msg.type === 'lassoSelect') {
    if (!state.projectionResources || !state.projectionBindGroup || state.numPoints === 0) return;

    const { device, numPoints, projectionPipeline, projectionResources, camera } = state;
    const { polygon } = msg;
    const numVertices = polygon.length / 2;
    if (numVertices < 3) return;

    // Get the current basis to populate projectedBuffer on-demand
    let currentBasis: Float32Array | null = null;
    if (state.directBasis) {
      currentBasis = state.directBasis;
    } else if (state.tour) {
      const { tour } = state;
      interpolateAtPosition(
        tour.interpolatedBasis,
        tour.bases,
        tour.arcLengths,
        tour.dims,
        tour.position,
      );
      currentBasis = tour.interpolatedBasis;
    }
    if (!currentBasis) return;

    // Dispatch compute projection on-demand (not in the render hot path)
    device.queue.writeBuffer(
      projectionResources.basisBuffer,
      0,
      currentBasis as Float32Array<ArrayBuffer>,
    );
    const computeCmd = dispatchProjection(
      device,
      projectionPipeline,
      state.projectionBindGroup,
      numPoints,
    );
    device.queue.submit([computeCmd]);

    // Transform the polygon from NDC space (what the user drew) into
    // projection space (what projectedBuffer contains) using the inverse
    // of the vertex shader's camera transform:
    //   ndc_x = (proj_x + panX) * zoom * iz / aspect
    //   ndc_y = (proj_y + panY) * zoom * iz + insetOffsetY
    // Inverted:
    //   proj_x = ndc_x * aspect / (zoom * iz) - panX
    //   proj_y = (ndc_y - insetOffsetY) / (zoom * iz) - panY
    const canvas = state.views[0]!.canvas;
    const aspect = canvas.width / canvas.height || 1;
    const iz = camera.insetZoom;
    const zoomIz = camera.zoom * iz;
    const projPolygon = new Float32Array(polygon.length);
    for (let v = 0; v < numVertices; v++) {
      projPolygon[v * 2] = (polygon[v * 2]! * aspect) / zoomIz - camera.panX;
      projPolygon[v * 2 + 1] = (polygon[v * 2 + 1]! - camera.insetOffsetY) / zoomIz - camera.panY;
    }

    // Read projected positions back from GPU
    const projSize = numPoints * 2 * 4; // 2 floats per point
    const readBuffer = device.createBuffer({
      label: 'proj-readback',
      size: projSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(projectionResources.projectedBuffer, 0, readBuffer, 0, projSize);
    device.queue.submit([encoder.finish()]);

    readBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const projected = new Float32Array(readBuffer.getMappedRange());

      // CPU point-in-polygon (ray casting) — bit-packed mask
      const mask = new Uint32Array(Math.ceil(numPoints / 32));
      for (let i = 0; i < numPoints; i++) {
        const px = projected[i * 2]!;
        const py = projected[i * 2 + 1]!;
        if (pointInPolygon(px, py, projPolygon, numVertices)) {
          const w = i >> 5;
          mask[w] = (mask[w] ?? 0) | (1 << (i & 31));
        }
      }

      readBuffer.unmap();
      readBuffer.destroy();

      // Apply selection mask
      if (state!.selectionBuffer) {
        state!.selectionBuffer.destroy();
      }

      state!.selectionBuffer = device.createBuffer({
        label: 'selection-mask',
        size: mask.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(state!.selectionBuffer, 0, mask as Uint32Array<ArrayBuffer>);

      state!.styleFlags.useSelectionMask = true;
      rebuildBindGroups();

      if ((state!.tour || state!.directBasis) && state!.projectionResources) {
        renderAllViews();
      }
    });
    return;
  }

  if (msg.type === 'benchmark') {
    handleBenchmark(msg.numPoints, msg.numDims, msg.numFrames);
    return;
  }

  if (msg.type === 'benchmarkExisting') {
    handleBenchmarkExisting(msg.numFrames);
    return;
  }
};

// ─── Benchmark ───────────────────────────────────────────────────────────

const handleBenchmark = async (
  numPoints: number,
  numDims: number,
  numFrames: number,
): Promise<void> => {
  if (!state) return;
  const { device } = state;

  // Generate random data columns
  const buffers: Float32Array[] = [];
  for (let d = 0; d < numDims; d++) {
    const col = new Float32Array(numPoints);
    for (let i = 0; i < numPoints; i++) col[i] = Math.random();
    buffers.push(col);
  }

  // Clean up previous projection resources
  if (state.adjBasisBuffer) {
    state.adjBasisBuffer.destroy();
    state.adjBasisBuffer = null;
  }

  // Create projection resources and upload data
  const res = createProjectionResources(device, state.projectionPipeline, numPoints, numDims);
  for (let d = 0; d < numDims; d++) {
    device.queue.writeBuffer(
      res.dataBuffer,
      d * numPoints * 4,
      buffers[d]! as Float32Array<ArrayBuffer>,
    );
  }

  // Upload norm params: [min=0, range=1] pairs
  const normData = new Float32Array(numDims * 2);
  for (let d = 0; d < numDims; d++) {
    normData[d * 2] = 0;
    normData[d * 2 + 1] = 1;
  }
  device.queue.writeBuffer(res.normParamsBuffer, 0, normData);

  writeProjectionParams(device, state.projectionPipeline.paramsBuffer, numPoints, numDims, 2.0);

  state.adjBasisBuffer = device.createBuffer({
    label: 'adj-basis-bench',
    size: numDims * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  state.normMins = new Float32Array(numDims);
  state.normRanges = new Float32Array(numDims).fill(1);
  state.projectionResources = { ...res, bindGroup: null as unknown as GPUBindGroup };
  state.numPoints = numPoints;
  state.numDims = numDims;
  state.styleFlags = { useSelectionMask: false };
  state.colorState = { mode: 0, columnOffset: 0, min: 0, range: 1, numStops: 0 };
  state.activeCatColumnName = null;
  state.maxPoints = 0;

  rebuildBindGroups();

  if (!state.renderBindGroup) {
    postMain({ type: 'error', message: 'Failed to build bind groups for benchmark' });
    return;
  }

  const basis = new Float32Array(numDims * 2);
  const camera: CameraState = {
    panX: 0,
    panY: 0,
    zoom: 1,
    aspect: 1,
    viewportHeight: state.views[0]!.canvas.height,
    insetOffsetY: 0,
    insetZoom: 1,
    use3d: false,
    rotation: null,
  };

  // Warmup (5 frames)
  for (let i = 0; i < 5; i++) {
    const angle = (i / 5) * Math.PI * 2;
    basis[0] = Math.cos(angle);
    basis[1] = Math.sin(angle);
    basis[numDims] = -Math.sin(angle);
    basis[numDims + 1] = Math.cos(angle);
    renderView(basis, 0, camera, state.renderBindGroup);
    await device.queue.onSubmittedWorkDone();
  }

  // Timed frames
  const frameTimes = new Float64Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    const angle = (i / numFrames) * Math.PI * 2;
    basis[0] = Math.cos(angle);
    basis[1] = Math.sin(angle);
    basis[numDims] = -Math.sin(angle);
    basis[numDims + 1] = Math.cos(angle);

    const t0 = performance.now();
    renderView(basis, 0, camera, state.renderBindGroup);
    await device.queue.onSubmittedWorkDone();
    frameTimes[i] = performance.now() - t0;
  }

  postMain({ type: 'benchmarkResult', frameTimes, numPoints }, [frameTimes.buffer]);
};

const handleBenchmarkExisting = async (numFrames: number): Promise<void> => {
  if (!state || !state.tour || !state.renderBindGroup || state.numPoints === 0) {
    postMain({ type: 'error', message: 'No data/tour loaded for benchmark' });
    return;
  }
  const { device, numPoints, tour } = state;

  // Save and restore tour position after benchmark
  const savedPosition = tour.position;
  state.directBasis = null;

  // Warmup (5 frames)
  for (let i = 0; i < 5; i++) {
    tour.position = i / 5;
    interpolateAtPosition(
      tour.interpolatedBasis,
      tour.bases,
      tour.arcLengths,
      tour.dims,
      tour.position,
    );
    renderView(tour.interpolatedBasis, 0, state.camera, state.renderBindGroup);
    await device.queue.onSubmittedWorkDone();
  }

  // Timed frames — sweep through full tour
  const frameTimes = new Float64Array(numFrames);
  for (let i = 0; i < numFrames; i++) {
    tour.position = i / numFrames;
    interpolateAtPosition(
      tour.interpolatedBasis,
      tour.bases,
      tour.arcLengths,
      tour.dims,
      tour.position,
    );

    const t0 = performance.now();
    renderView(tour.interpolatedBasis, 0, state.camera, state.renderBindGroup);
    await device.queue.onSubmittedWorkDone();
    frameTimes[i] = performance.now() - t0;
  }

  // Restore original position
  tour.position = savedPosition;

  postMain({ type: 'benchmarkResult', frameTimes, numPoints }, [frameTimes.buffer]);
};

/** Ray-casting point-in-polygon test. */
const pointInPolygon = (
  px: number,
  py: number,
  polygon: Float32Array,
  numVertices: number,
): boolean => {
  let inside = false;
  for (let i = 0, j = numVertices - 1; i < numVertices; j = i++) {
    const xi = polygon[i * 2]!;
    const yi = polygon[i * 2 + 1]!;
    const xj = polygon[j * 2]!;
    const yj = polygon[j * 2 + 1]!;

    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
};

// ─── Main thread messages ──────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<MainToGpu>): Promise<void> => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      const { device } = await initDevice();

      const views = msg.canvases.map((canvas) => configureCanvas(canvas, device));
      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

      const hdrFormat: GPUTextureFormat = 'rgba32float';
      const pointPipeline = createPointPipeline(device, hdrFormat);
      const projectionPipeline = createProjectionPipeline(device);
      const colorPipelines = createColorPipelines(device);
      const tonemapPipeline = createTonemapPipeline(device, canvasFormat);

      state = {
        device,
        views,
        pointPipeline,
        renderBindGroup: null,
        projectionPipeline,
        projectionResources: null,
        projectionBindGroup: null,
        colorPipelines,
        numPoints: 0,
        numDims: 0,
        categoricalBuffers: new Map(),
        tour: null,
        style: { pointSize: 'auto', opacity: 'auto', color: [0.25, 0.5, 0.9] },
        styleFlags: { useSelectionMask: false },
        camera: {
          panX: 0,
          panY: 0,
          zoom: msg.zoom,
          aspect: 1,
          viewportHeight: 1,
          insetOffsetY: 0,
          insetZoom: 1,
          use3d: false,
          rotation: null,
        },
        colorLutBuffer: null,
        colorState: { mode: 0, columnOffset: 0, min: 0, range: 1, numStops: 0 },
        activeCatColumnName: null,
        selectionBuffer: null,
        directBasis: null,
        backgroundColor: [0, 0, 0],
        dpr: msg.dpr,
        hdrFormat,
        tonemapPipeline,
        hdrTextures: [],
        hdrTextureViews: [],
        tonemapBindGroups: [],
        maxPoints: 0,
        adjBasisBuffer: null,
        normMins: null,
        normRanges: null,
        is3dActive: false,
        rotation3d: null,
        residualPC: null,
        cpuData: null,
      };

      msg.dataPort.onmessage = onDataMessage;

      // Replay any messages that arrived while init was awaiting
      const buffered = pendingMessages;
      pendingMessages = null;
      if (buffered) {
        for (const m of buffered) {
          handleMessage(m);
        }
      }

      postMain({ type: 'ready' });
    } catch (err) {
      postMain({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

  // If init hasn't completed yet, buffer the message for replay
  if (pendingMessages) {
    pendingMessages.push(msg);
    return;
  }

  handleMessage(msg);
};
