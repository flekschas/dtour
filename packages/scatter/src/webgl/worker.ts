/// <reference lib="webworker" />

import type { DataToGpu } from '../data/messages.ts';
import type { GpuToMain, MainToGpu } from '../gpu/messages.ts';
import pointFrag from '../shaders/point.frag?raw';
import pointVert from '../shaders/point.vert?raw';
import tonemapFrag from '../shaders/tonemap.frag?raw';
import tonemapVert from '../shaders/tonemap.vert?raw';
import { computeArcLengths, interpolateAtPosition } from '../tour/arc-length.ts';

// ─── Types ──────────────────────────────────────────────────────────────────

type TourState = {
  bases: Float32Array[];
  arcLengths: Float32Array;
  dims: number;
  position: number;
  interpolatedBasis: Float32Array;
};

type MainView = {
  canvas: OffscreenCanvas;
  gl: WebGL2RenderingContext;
};

type PreviewEntry = {
  canvas: OffscreenCanvas;
  ctx2d: OffscreenCanvasRenderingContext2D;
};

type GLState = {
  mainView: MainView;
  previewCanvases: Map<number, PreviewEntry>;
  // Point program
  pointProgram: WebGLProgram;
  pointLocs: PointLocations;
  // Tonemap program
  tonemapProgram: WebGLProgram;
  tonemapLocs: TonemapLocations;
  // HDR FBOs per view
  hdrFbos: (WebGLFramebuffer | null)[];
  hdrTextures: (WebGLTexture | null)[];
  hdrSizes: ([number, number] | null)[];
  // Blit FBO (RGBA8) for preview readback via readPixels
  blitFbo: WebGLFramebuffer | null;
  blitTexture: WebGLTexture | null;
  blitSize: [number, number] | null;
  // Data texture (R32F, width=texWidth, height=numDims*tileRows)
  dataTexture: WebGLTexture | null;
  // Color LUT texture (R32UI, width=numStops, height=1) — tiny colormap/palette
  colorLutTexture: WebGLTexture | null;
  // Categorical index textures (R32UI, tiled like data tex) — one per cat column
  catIndexTextures: Map<string, WebGLTexture>;
  // Selection texture (R32UI, width=ceil(N/32), height=1)
  selectionTexture: WebGLTexture | null;
  // Data
  numPoints: number;
  numDims: number;
  // Raw data buffers (kept on CPU for lasso/PCA)
  dataBuffers: Float32Array[];
  normMins: Float32Array | null;
  normRanges: Float32Array | null;
  // Categorical index buffers on CPU, keyed by column name
  categoricalBuffers: Map<string, Uint32Array>;
  // Tour
  tour: TourState | null;
  // Style
  style: RawPointStyle;
  styleFlags: StyleFlags;
  // Color mapping state (LUT approach)
  colorMode: number; // 0=uniform, 1=continuous, 2=categorical
  colorColumnIndex: number;
  colorMin: number;
  colorRange: number;
  colorNumStops: number;
  activeCatColumnName: string | null;
  camera: CameraState;
  // Direct basis
  directBasis: Float32Array | null;
  // Background
  backgroundColor: [number, number, number];
  dpr: number;
  // Decimation
  maxPoints: number;
  // Tiling: data texture width (min(numPoints, MAX_TEXTURE_SIZE))
  texWidth: number;
};

type RawPointStyle = {
  pointSize: number | 'auto';
  opacity: number | 'auto';
  color: [number, number, number];
};

type PointStyle = {
  pointSize: number;
  opacity: number;
  color: [number, number, number];
};

type StyleFlags = {
  useSelectionMask: boolean;
};

type CameraState = {
  panX: number;
  panY: number;
  zoom: number;
  aspect: number;
  viewportHeight: number;
  insetOffsetY: number;
  insetZoom: number;
};

type PointLocations = {
  u_data: WebGLUniformLocation;
  u_adjBasisX: WebGLUniformLocation;
  u_adjBasisY: WebGLUniformLocation;
  u_bias: WebGLUniformLocation;
  u_pointSize: WebGLUniformLocation;
  u_opacity: WebGLUniformLocation;
  u_color: WebGLUniformLocation;
  u_useSubtractive: WebGLUniformLocation;
  u_pan: WebGLUniformLocation;
  u_zoom: WebGLUniformLocation;
  u_aspect: WebGLUniformLocation;
  u_viewportHeight: WebGLUniformLocation;
  u_insetOffsetY: WebGLUniformLocation;
  u_insetZoom: WebGLUniformLocation;
  u_numPoints: WebGLUniformLocation;
  u_numDims: WebGLUniformLocation;
  u_maxPoints: WebGLUniformLocation;
  u_texWidth: WebGLUniformLocation;
  u_colorMode: WebGLUniformLocation;
  u_colorColumnIndex: WebGLUniformLocation;
  u_colorMin: WebGLUniformLocation;
  u_colorRange: WebGLUniformLocation;
  u_colorNumStops: WebGLUniformLocation;
  u_colorLutTex: WebGLUniformLocation;
  u_catIndexTex: WebGLUniformLocation;
  u_useSelectionMask: WebGLUniformLocation;
  u_selectionTex: WebGLUniformLocation;
};

type TonemapLocations = {
  u_hdrTexture: WebGLUniformLocation;
  u_mode: WebGLUniformLocation;
};

// ─── State ──────────────────────────────────────────────────────────────────

let state: GLState | null = null;
let pendingMessages: MainToGpu[] | null = [];

const postMain = (msg: GpuToMain, transfers?: Transferable[]): void => {
  self.postMessage(msg, transfers ?? []);
};

// ─── GL Helpers ──────────────────────────────────────────────────────────────

const compileShader = (gl: WebGL2RenderingContext, type: number, source: string): WebGLShader => {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${log}`);
  }
  return shader;
};

const linkProgram = (
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram => {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${log}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
};

const getUniform = (
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation => {
  const loc = gl.getUniformLocation(program, name);
  // Null is valid for optimized-out uniforms
  return loc!;
};

// ─── Auto-style (Reusser color budget) ───────────────────────────────────

const COLOR_BUDGET = 0.5;

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
  const minRadius = dpr;

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

const cssToNdc = (cssPx: number, canvasHeight: number): number => {
  return (cssPx * state!.dpr) / canvasHeight;
};

const resolveStyleForCanvas = (canvasWidth: number, canvasHeight: number): PointStyle => {
  const { style, numPoints, dpr } = state!;

  if (style.pointSize !== 'auto' && style.opacity !== 'auto') {
    return {
      pointSize: cssToNdc(style.pointSize, canvasHeight),
      opacity: style.opacity,
      color: style.color,
    };
  }

  const auto = computeAutoStyle(numPoints, canvasWidth / dpr, canvasHeight / dpr, dpr);

  return {
    pointSize:
      style.pointSize === 'auto' ? auto.pointSize : cssToNdc(style.pointSize, canvasHeight),
    opacity: style.opacity === 'auto' ? auto.opacity : style.opacity,
    color: style.color,
  };
};

// ─── Inline projection helpers ────────────────────────────────────────────

const VIEWPORT_SCALE = 2.0;
let adjBasisWeights: Float32Array | null = null;

const computeAdjustedBasis = (
  basis: Float32Array,
  mins: Float32Array,
  ranges: Float32Array,
  dims: number,
): { weights: Float32Array; biasX: number; biasY: number } => {
  if (!adjBasisWeights || adjBasisWeights.length !== dims * 2) {
    adjBasisWeights = new Float32Array(dims * 2);
  }

  let biasX = 0;
  let biasY = 0;

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
  }

  return { weights: adjBasisWeights, biasX, biasY };
};

// ─── HDR FBO management ──────────────────────────────────────────────────

/** Ensure an HDR FBO exists at the given slot index with the specified dimensions. */
const ensureHdrFbo = (slotIndex: number, w: number, h: number): void => {
  if (!state) return;
  const { hdrFbos, hdrTextures, hdrSizes } = state;
  const gl = state.mainView.gl;

  const existing = hdrSizes[slotIndex];
  if (existing && existing[0] === w && existing[1] === h) return;

  // Clean up old
  if (hdrFbos[slotIndex]) gl.deleteFramebuffer(hdrFbos[slotIndex]);
  if (hdrTextures[slotIndex]) gl.deleteTexture(hdrTextures[slotIndex]);

  // Create float texture
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  state.hdrFbos[slotIndex] = fbo;
  state.hdrTextures[slotIndex] = tex;
  state.hdrSizes[slotIndex] = [w, h];
};

// ─── Blit FBO for preview readback ────────────────────────────────────────

const ensureBlitFbo = (w: number, h: number): void => {
  if (!state) return;
  const existing = state.blitSize;
  if (existing && existing[0] === w && existing[1] === h) return;

  const gl = state.mainView.gl;

  if (state.blitFbo) gl.deleteFramebuffer(state.blitFbo);
  if (state.blitTexture) gl.deleteTexture(state.blitTexture);

  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  state.blitFbo = fbo;
  state.blitTexture = tex;
  state.blitSize = [w, h];
};

// ─── Worker-driven playback ───────────────────────────────────────────────

type PlaybackState = {
  speed: number;
  direction: 1 | -1;
  prevTime: number | null;
  rafId: number;
  /** Accumulates rAF-delta intervals — one entry per rAF tick where a frame was actually submitted. */
  frameTimes: number[];
  /** rAF timestamp of the last tick where a frame was actually submitted to the GPU. */
  lastRenderedRafTime: number | null;
};

let playbackState: PlaybackState | null = null;
let lastBroadcastTime = 0;
const BROADCAST_INTERVAL_MS = 33;
/** GPU backpressure — skip frames when the GPU can't keep up. */
let gpuFence: WebGLSync | null = null;
/** When true, a render was skipped due to GPU backpressure and should be retried. */
let renderPending = false;
let deferredRenderTimer: ReturnType<typeof setTimeout> | null = null;

/** Check if the GPU is done with the previous frame. */
const isGpuReady = (): boolean => {
  if (!gpuFence || !state) return true;
  const gl = state.mainView.gl;
  const status = gl.clientWaitSync(gpuFence, 0, 0);
  if (status !== gl.TIMEOUT_EXPIRED) {
    gl.deleteSync(gpuFence);
    gpuFence = null;
    return true;
  }
  return false;
};

/** Submit a render if the GPU is ready; otherwise mark it pending for later.
 *  Returns true if a frame was actually submitted this call. */
const throttledRenderMainView = (): boolean => {
  if (!isGpuReady()) {
    renderPending = true;
    // Poll for GPU completion to flush the deferred render.
    // During playback, let the rAF loop drive rendering so that
    // playbackTick sees isGpuReady()=true and can record frame times.
    if (!deferredRenderTimer) {
      deferredRenderTimer = setTimeout(function poll() {
        deferredRenderTimer = null;
        if (renderPending && state && !playbackState) throttledRenderMainView();
      }, 2);
    }
    return false;
  }
  renderPending = false;
  renderMainView();
  if (state) {
    const gl = state.mainView.gl;
    gpuFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
  }
  return true;
};

const playbackTick = (time: number): void => {
  if (!playbackState || !state?.tour) return;

  const frameMs = playbackState.prevTime !== null ? time - playbackState.prevTime : undefined;

  if (frameMs !== undefined) {
    const dt = frameMs / 1000;
    const delta = (dt * playbackState.speed * playbackState.direction) / 20;
    let next = state.tour.position + delta;
    next = next - Math.floor(next);
    state.tour.position = next;
  }
  playbackState.prevTime = time;

  state.directBasis = null;
  const didRender = throttledRenderMainView();

  // Record rAF-delta frame time: bounded by display refresh rate, so heavy datasets show lower FPS.
  if (didRender) {
    if (playbackState.lastRenderedRafTime !== null) {
      playbackState.frameTimes.push(time - playbackState.lastRenderedRafTime);
    }
    playbackState.lastRenderedRafTime = time;
  }

  if (time - lastBroadcastTime >= BROADCAST_INTERVAL_MS) {
    postMain({ type: 'playbackTick', position: state.tour.position, frameMs });
    lastBroadcastTime = time;
  }

  playbackState.rafId = requestAnimationFrame(playbackTick);
};

const startPlayback = (speed: number, direction: 1 | -1): void => {
  if (playbackState) {
    playbackState.speed = speed;
    playbackState.direction = direction;
  } else {
    gpuFence = null;
    renderPending = false;
    playbackState = {
      speed,
      direction,
      prevTime: null,
      rafId: 0,
      frameTimes: [],
      lastRenderedRafTime: null,
    };
    playbackState.rafId = requestAnimationFrame(playbackTick);
  }
};

const stopPlayback = (): void => {
  if (playbackState) {
    cancelAnimationFrame(playbackState.rafId);
    if (state?.tour) {
      postMain({ type: 'playbackTick', position: state.tour.position });
    }
    // Return all accumulated frame times for benchmark collection
    const ft = Float64Array.from(playbackState.frameTimes);
    postMain({ type: 'playbackStopped', frameTimes: ft }, [ft.buffer]);
    playbackState = null;
  }
};

// ─── Render helpers ───────────────────────────────────────────────────────

const renderView = (
  basis: Float32Array,
  canvasWidth: number,
  canvasHeight: number,
  hdrSlot: number,
  camera: CameraState,
  tonemapTarget?: WebGLFramebuffer,
): void => {
  if (!state || !state.dataTexture || !state.normMins || !state.normRanges) return;

  const { numPoints, numDims, pointProgram, pointLocs, tonemapProgram, tonemapLocs } = state;
  const gl = state.mainView.gl;

  const { weights, biasX, biasY } = computeAdjustedBasis(
    basis,
    state.normMins,
    state.normRanges,
    numDims,
  );

  const resolved = resolveStyleForCanvas(canvasWidth, canvasHeight);

  // Select blend mode + tonemap mode
  const bg = state.backgroundColor;
  const bgLuminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2];
  const useNormal = state.colorMode > 0;
  const useSubtractive = !useNormal && bgLuminance > 0.5;

  let tonemapMode: number;
  if (useNormal) {
    tonemapMode = 1;
  } else if (useSubtractive) {
    tonemapMode = 1;
  } else {
    tonemapMode = 0;
  }

  const aspect = canvasWidth / canvasHeight || 1;

  // Ensure HDR FBO
  ensureHdrFbo(hdrSlot, canvasWidth, canvasHeight);

  // ── Pass 1: Render points to HDR FBO ──
  gl.bindFramebuffer(gl.FRAMEBUFFER, state.hdrFbos[hdrSlot]!);
  gl.viewport(0, 0, canvasWidth, canvasHeight);
  gl.clearColor(bg[0], bg[1], bg[2], 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);

  if (numPoints > 0) {
    gl.useProgram(pointProgram);

    // Set blend mode
    gl.enable(gl.BLEND);
    if (useNormal) {
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    } else if (useSubtractive) {
      gl.blendEquationSeparate(gl.FUNC_REVERSE_SUBTRACT, gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
    } else {
      gl.blendEquation(gl.FUNC_ADD);
      gl.blendFunc(gl.ONE, gl.ONE);
    }

    // Bind data texture to unit 0
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, state.dataTexture);
    gl.uniform1i(pointLocs.u_data, 0);

    // Bind color LUT texture to unit 1
    gl.activeTexture(gl.TEXTURE1);
    if (state.colorLutTexture) {
      gl.bindTexture(gl.TEXTURE_2D, state.colorLutTexture);
    }
    gl.uniform1i(pointLocs.u_colorLutTex, 1);

    // Bind selection texture to unit 2
    gl.activeTexture(gl.TEXTURE2);
    if (state.selectionTexture) {
      gl.bindTexture(gl.TEXTURE_2D, state.selectionTexture);
    }
    gl.uniform1i(pointLocs.u_selectionTex, 2);

    // Bind categorical index texture to unit 3
    gl.activeTexture(gl.TEXTURE3);
    if (state.colorMode === 2 && state.activeCatColumnName) {
      const catTex = state.catIndexTextures.get(state.activeCatColumnName);
      if (catTex) gl.bindTexture(gl.TEXTURE_2D, catTex);
    }
    gl.uniform1i(pointLocs.u_catIndexTex, 3);

    // Upload adjusted basis weights as uniform arrays
    const basisX = new Float32Array(128);
    const basisY = new Float32Array(128);
    for (let d = 0; d < numDims && d < 128; d++) {
      basisX[d] = weights[d]!;
      basisY[d] = weights[numDims + d]!;
    }
    gl.uniform1fv(pointLocs.u_adjBasisX, basisX);
    gl.uniform1fv(pointLocs.u_adjBasisY, basisY);

    // Upload uniforms
    gl.uniform2f(pointLocs.u_bias, biasX, biasY);
    gl.uniform1f(pointLocs.u_pointSize, resolved.pointSize);
    gl.uniform1f(pointLocs.u_opacity, resolved.opacity);
    gl.uniform4f(pointLocs.u_color, resolved.color[0], resolved.color[1], resolved.color[2], 1.0);
    gl.uniform1f(pointLocs.u_useSubtractive, useSubtractive ? 1.0 : 0.0);
    gl.uniform2f(pointLocs.u_pan, camera.panX, camera.panY);
    gl.uniform1f(pointLocs.u_zoom, camera.zoom);
    gl.uniform1f(pointLocs.u_aspect, aspect);
    gl.uniform1f(pointLocs.u_viewportHeight, canvasHeight);
    gl.uniform1f(pointLocs.u_insetOffsetY, camera.insetOffsetY);
    gl.uniform1f(pointLocs.u_insetZoom, camera.insetZoom);
    gl.uniform1i(pointLocs.u_numPoints, numPoints);
    gl.uniform1i(pointLocs.u_numDims, numDims);
    gl.uniform1i(pointLocs.u_maxPoints, state.maxPoints);
    gl.uniform1i(pointLocs.u_texWidth, state.texWidth);
    gl.uniform1i(pointLocs.u_colorMode, state.colorMode);
    gl.uniform1i(pointLocs.u_colorColumnIndex, state.colorColumnIndex);
    gl.uniform1f(pointLocs.u_colorMin, state.colorMin);
    gl.uniform1f(pointLocs.u_colorRange, state.colorRange);
    gl.uniform1i(pointLocs.u_colorNumStops, state.colorNumStops);
    gl.uniform1f(pointLocs.u_useSelectionMask, state.styleFlags.useSelectionMask ? 1.0 : 0.0);

    gl.drawArrays(gl.POINTS, 0, numPoints);
    gl.disable(gl.BLEND);
  }

  // ── Pass 2: Tonemap to canvas (or blit FBO for previews) ──
  gl.bindFramebuffer(gl.FRAMEBUFFER, tonemapTarget ?? null);
  gl.viewport(0, 0, canvasWidth, canvasHeight);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  gl.useProgram(tonemapProgram);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, state.hdrTextures[hdrSlot]!);
  gl.uniform1i(tonemapLocs.u_hdrTexture, 0);
  gl.uniform1f(tonemapLocs.u_mode, tonemapMode);

  gl.drawArrays(gl.TRIANGLES, 0, 3);
};

// ─── Deferred preview rendering ───────────────────────────────────────────

let previewRenderPending = false;

// Reusable buffers for preview readback — avoids per-preview heap allocations
let previewPixelBuf: Uint8Array | null = null;
let previewFlipTemp: Uint8Array | null = null;

const renderOnePreview = (id: number, basis: Float32Array): void => {
  if (!state?.dataTexture) return;

  const entry = state.previewCanvases.get(id);
  if (!entry) {
    postMain({ type: 'rendered', viewIndex: id });
    return;
  }

  const w = entry.canvas.width;
  const h = entry.canvas.height;
  if (w === 0 || h === 0) {
    postMain({ type: 'rendered', viewIndex: id });
    return;
  }

  const gl = state.mainView.gl;

  // Ensure RGBA8 blit FBO for readback
  ensureBlitFbo(w, h);

  // Render points → HDR FBO (slot 1, reused for all previews), tonemap → blit FBO
  renderView(
    basis,
    w,
    h,
    1, // preview HDR slot
    { panX: 0, panY: 0, zoom: 1, aspect: 1, viewportHeight: h, insetOffsetY: 0, insetZoom: 1 },
    state.blitFbo!,
  );

  // Read pixels from blit FBO (reuse buffers across previews)
  const pixelBytes = w * h * 4;
  const rowSize = w * 4;
  if (!previewPixelBuf || previewPixelBuf.length < pixelBytes) {
    previewPixelBuf = new Uint8Array(pixelBytes);
  }
  if (!previewFlipTemp || previewFlipTemp.length < rowSize) {
    previewFlipTemp = new Uint8Array(rowSize);
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, state.blitFbo!);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, previewPixelBuf);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // Flip vertically (GL bottom-up → Canvas top-down)
  for (let row = 0; row < h >> 1; row++) {
    const topOff = row * rowSize;
    const botOff = (h - 1 - row) * rowSize;
    previewFlipTemp.set(previewPixelBuf.subarray(topOff, topOff + rowSize));
    previewPixelBuf.copyWithin(topOff, botOff, botOff + rowSize);
    previewPixelBuf.set(previewFlipTemp, botOff);
  }

  // Blit to preview canvas via 2D context
  const imageData = new ImageData(
    new Uint8ClampedArray(previewPixelBuf.buffer as ArrayBuffer, 0, pixelBytes),
    w,
    h,
  );
  entry.ctx2d.putImageData(imageData, 0, 0);

  postMain({ type: 'rendered', viewIndex: id });
};

const renderPreviewViews = (): void => {
  if (!state?.dataTexture || !state.tour) return;
  const { tour } = state;
  for (const [id] of state.previewCanvases) {
    if (id >= tour.bases.length) continue;
    renderOnePreview(id, tour.bases[id]!);
  }
};

const schedulePreviewRender = (): void => {
  if (previewRenderPending) return;
  previewRenderPending = true;
  setTimeout(() => {
    previewRenderPending = false;
    renderPreviewViews();
  }, 0);
};

const renderAllViews = (): void => {
  if (!state?.dataTexture) return;
  renderMainView();
  schedulePreviewRender();
};

const renderMainView = (): void => {
  if (!state?.dataTexture) return;
  const { canvas } = state.mainView;

  if (state.directBasis) {
    renderView(state.directBasis, canvas.width, canvas.height, 0, state.camera);
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
  renderView(tour.interpolatedBasis, canvas.width, canvas.height, 0, state.camera);
  postMain({ type: 'rendered', viewIndex: 0 });
};

// ─── Color/selection encoding (CPU) ──────────────────────────────────────

const applyColorUpdate = (): void => {
  if (!state) return;
  if ((state.tour || state.directBasis) && state.dataTexture) {
    renderAllViews();
  }
};

const applySelectionUpdate = (): void => {
  if (!state) return;
  state.styleFlags.useSelectionMask = true;
  if ((state.tour || state.directBasis) && state.dataTexture) {
    renderAllViews();
  }
};

// ─── Data from Data Worker ─────────────────────────────────────────────────

let currentDataVersion = 0;

const onDataMessage = (event: MessageEvent<DataToGpu>): void => {
  if (!state) return;
  const gl = state.mainView.gl;

  // ── Continuous color (LUT upload) ──
  if (event.data.type === 'setColorContinuous') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { columnIndex, min, range, colormap } = event.data;

    // Upload tiny LUT as R32UI texture (width=numStops, height=1)
    if (state.colorLutTexture) gl.deleteTexture(state.colorLutTexture);
    state.colorLutTexture = uploadLutTexture(gl, colormap);
    state.colorMode = 1;
    state.colorColumnIndex = columnIndex;
    state.colorMin = min;
    state.colorRange = range;
    state.colorNumStops = colormap.length;
    state.activeCatColumnName = null;

    applyColorUpdate();
    return;
  }

  // ── Categorical color (LUT upload) ──
  if (event.data.type === 'setColorCategorical') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { catColumnName, palette } = event.data;
    if (!state.catIndexTextures.has(catColumnName)) return;

    // Upload palette as tiny R32UI texture (width=numStops, height=1)
    if (state.colorLutTexture) gl.deleteTexture(state.colorLutTexture);
    state.colorLutTexture = uploadLutTexture(gl, palette);
    state.colorMode = 2;
    state.colorNumStops = palette.length;
    state.activeCatColumnName = catColumnName;

    applyColorUpdate();
    return;
  }

  // ── Continuous selection (CPU) ──
  if (event.data.type === 'selectContinuous') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { columnIndex, ranges } = event.data;
    const { numPoints, dataBuffers } = state;
    const colData = dataBuffers[columnIndex];
    if (!colData) return;

    const mask = new Uint32Array(Math.ceil(numPoints / 32));
    const numRanges = ranges.length / 2;

    for (let i = 0; i < numPoints; i++) {
      const val = colData[i]!;
      let selected = false;
      for (let r = 0; r < numRanges; r++) {
        if (val >= ranges[r * 2]! && val <= ranges[r * 2 + 1]!) {
          selected = true;
          break;
        }
      }
      if (selected) {
        mask[i >> 5] = mask[i >> 5]! | 0 | (1 << (i & 31));
      }
    }

    uploadSelectionMask(gl, mask);
    applySelectionUpdate();
    return;
  }

  // ── Categorical selection (CPU) ──
  if (event.data.type === 'selectCategorical') {
    if (event.data.dataVersion !== currentDataVersion) return;

    const { catColumnName, selectedLabels } = event.data;
    const indices = state.categoricalBuffers.get(catColumnName);
    if (!indices) return;

    const { numPoints } = state;
    const mask = new Uint32Array(Math.ceil(numPoints / 32));

    for (let i = 0; i < numPoints; i++) {
      const labelIdx = indices[i]!;
      if (labelIdx < selectedLabels.length && selectedLabels[labelIdx]! > 0) {
        mask[i >> 5] = mask[i >> 5]! | 0 | (1 << (i & 31));
      }
    }

    uploadSelectionMask(gl, mask);
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
  if (state.dataTexture) {
    gl.deleteTexture(state.dataTexture);
    state.dataTexture = null;
  }
  if (state.colorLutTexture) {
    gl.deleteTexture(state.colorLutTexture);
    state.colorLutTexture = null;
  }
  for (const tex of state.catIndexTextures.values()) {
    gl.deleteTexture(tex);
  }
  state.catIndexTextures.clear();
  if (state.selectionTexture) {
    gl.deleteTexture(state.selectionTexture);
    state.selectionTexture = null;
  }

  state.styleFlags = { useSelectionMask: false };
  state.colorMode = 0;
  state.colorColumnIndex = 0;
  state.colorMin = 0;
  state.colorRange = 1;
  state.colorNumStops = 0;
  state.activeCatColumnName = null;
  state.categoricalBuffers.clear();

  // Store raw data on CPU (for color/selection/lasso/PCA)
  state.dataBuffers = buffers;
  state.normMins = new Float32Array(mins);
  state.normRanges = new Float32Array(ranges);
  state.numPoints = rows;
  state.numDims = dims;

  // Upload data as R32F texture with tiling for large point counts.
  // Layout: width=texWidth, height=numDims*tileRows.
  // For point idx: col = idx % texWidth, row = d * tileRows + idx / texWidth.
  const maxTexSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  const texWidth = Math.min(rows, maxTexSize);
  const tileRows = Math.ceil(rows / texWidth);
  state.texWidth = texWidth;

  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);

  const texHeight = dims * tileRows;
  const texData = new Float32Array(texWidth * texHeight);
  for (let d = 0; d < dims; d++) {
    const col = buffers[d]!;
    const baseRow = d * tileRows;
    for (let i = 0; i < rows; i++) {
      const tx = i % texWidth;
      const ty = baseRow + Math.floor(i / texWidth);
      texData[ty * texWidth + tx] = col[i]!;
    }
  }
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, texWidth, texHeight, 0, gl.RED, gl.FLOAT, texData);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  state.dataTexture = tex;

  // Store categorical indices on CPU + upload as tiled R32UI textures
  for (const cat of categoricalColumns) {
    state.categoricalBuffers.set(cat.name, cat.indices);
    state.catIndexTextures.set(cat.name, uploadCatIndexTexture(gl, cat.indices, rows, texWidth));
  }

  // Render
  if (state.tour && state.tour.dims === dims) {
    renderAllViews();
  } else {
    const count = Math.max(1, state.previewCanvases.size);
    const defaultBases: Float32Array[] = [];
    for (let k = 0; k < count; k++) {
      const basis = new Float32Array(dims * 2);
      const d = Math.floor((k / count) * dims);
      basis[d] = 1;
      basis[dims + ((d + 1) % dims)] = 1;
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

// ─── Color LUT texture upload (tiny, 1D) ─────────────────────────────────────

const uploadLutTexture = (gl: WebGL2RenderingContext, lut: Uint32Array): WebGLTexture => {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, lut.length, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, lut);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
};

// ─── Categorical index texture upload (tiled R32UI) ──────────────────────────

const uploadCatIndexTexture = (
  gl: WebGL2RenderingContext,
  indices: Uint32Array,
  numPoints: number,
  texWidth: number,
): WebGLTexture => {
  const tileRows = Math.ceil(numPoints / texWidth);
  let tiledData: Uint32Array;
  if (tileRows === 1) {
    tiledData = indices;
  } else {
    tiledData = new Uint32Array(texWidth * tileRows);
    for (let i = 0; i < numPoints; i++) {
      const tx = i % texWidth;
      const ty = Math.floor(i / texWidth);
      tiledData[ty * texWidth + tx] = indices[i]!;
    }
  }

  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R32UI,
    texWidth,
    tileRows,
    0,
    gl.RED_INTEGER,
    gl.UNSIGNED_INT,
    tiledData,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
};

// ─── Selection mask upload ──────────────────────────────────────────────────

const uploadSelectionMask = (gl: WebGL2RenderingContext, mask: Uint32Array): void => {
  if (!state) return;
  if (state.selectionTexture) gl.deleteTexture(state.selectionTexture);

  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
    gl.TEXTURE_2D,
    0,
    gl.R32UI,
    mask.length,
    1,
    0,
    gl.RED_INTEGER,
    gl.UNSIGNED_INT,
    mask,
  );
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  state.selectionTexture = tex;
};

// ─── Lasso (CPU) ────────────────────────────────────────────────────────────

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

const handleLassoSelect = (polygon: Float32Array): void => {
  if (!state || !state.normMins || !state.normRanges || state.numPoints === 0) return;

  const { numPoints, numDims, dataBuffers, camera } = state;
  const numVertices = polygon.length / 2;
  if (numVertices < 3) return;

  // Get current basis
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

  // Compute adjusted basis for projection
  const { weights, biasX, biasY } = computeAdjustedBasis(
    currentBasis,
    state.normMins,
    state.normRanges,
    numDims,
  );

  // Transform polygon from NDC to projection space
  const canvas = state.mainView.canvas;
  const aspect = canvas.width / canvas.height || 1;
  const iz = camera.insetZoom;
  const zoomIz = camera.zoom * iz;
  const projPolygon = new Float32Array(polygon.length);
  for (let v = 0; v < numVertices; v++) {
    projPolygon[v * 2] = (polygon[v * 2]! * aspect) / zoomIz - camera.panX;
    projPolygon[v * 2 + 1] = (polygon[v * 2 + 1]! - camera.insetOffsetY) / zoomIz - camera.panY;
  }

  // Project all points on CPU and do point-in-polygon
  const mask = new Uint32Array(Math.ceil(numPoints / 32));
  for (let i = 0; i < numPoints; i++) {
    let px = biasX;
    let py = biasY;
    for (let d = 0; d < numDims; d++) {
      const raw = dataBuffers[d]![i]!;
      px += raw * weights[d]!;
      py += raw * weights[numDims + d]!;
    }

    if (pointInPolygon(px, py, projPolygon, numVertices)) {
      mask[i >> 5] = mask[i >> 5]! | 0 | (1 << (i & 31));
    }
  }

  const gl = state.mainView.gl;
  uploadSelectionMask(gl, mask);
  applySelectionUpdate();

  // Send mask back so the host can read which points were selected
  const maskCopy = new Uint32Array(mask);
  postMain({ type: 'selectionResult', mask: maskCopy }, [maskCopy.buffer]);
};

// ─── Metrics ─────────────────────────────────────────────────────────────

/** Compute estimated GPU memory usage from known texture sizes. */
const computeGpuMemoryBytes = (): number => {
  if (!state) return 0;
  let bytes = 0;

  // Data texture (R32F = 4 bytes/pixel)
  if (state.dataTexture) {
    const tileRows = Math.ceil(state.numPoints / state.texWidth);
    bytes += state.texWidth * state.numDims * tileRows * 4;
  }

  // HDR textures (RGBA32F = 16 bytes/pixel)
  for (let i = 0; i < state.hdrSizes.length; i++) {
    const size = state.hdrSizes[i];
    if (size) bytes += size[0] * size[1] * 16;
  }

  // Blit texture (RGBA8 = 4 bytes/pixel)
  if (state.blitSize) bytes += state.blitSize[0] * state.blitSize[1] * 4;

  // Color LUT texture (R32UI = 4 bytes per stop)
  if (state.colorLutTexture) bytes += state.colorNumStops * 4;

  // Categorical index textures (R32UI, tiled)
  for (const buf of state.categoricalBuffers.values()) bytes += buf.byteLength;

  // Selection texture (R32UI, ceil(N/32))
  if (state.selectionTexture) bytes += Math.ceil(state.numPoints / 32) * 4;

  return bytes;
};

const handleGetMetrics = (): void => {
  if (!state) return;

  // ArrayBuffer backing stores live outside V8 heap — sum them explicitly.
  let externalBytes = 0;
  for (const buf of state.dataBuffers) externalBytes += buf.byteLength;
  for (const buf of state.categoricalBuffers.values()) externalBytes += buf.byteLength;
  if (state.normMins) externalBytes += state.normMins.byteLength;
  if (state.normRanges) externalBytes += state.normRanges.byteLength;
  if (state.tour) {
    for (const b of state.tour.bases) externalBytes += b.byteLength;
    externalBytes += state.tour.interpolatedBasis.byteLength;
  }

  postMain({
    type: 'metricsResult',
    gpuMemoryBytes: computeGpuMemoryBytes(),
    numPoints: state.numPoints,
    numDims: state.numDims,
    workerJsHeapBytes: externalBytes,
  });
};

// ─── Benchmark ───────────────────────────────────────────────────────────

const handleBenchmark = (numFrames: number): void => {
  if (!state || !state.tour || !state.dataTexture || state.numPoints === 0) {
    postMain({ type: 'error', message: 'No data/tour loaded for benchmark' });
    return;
  }
  const gl = state.mainView.gl;
  const { numPoints, numDims, tour } = state;

  // Save and restore tour position after benchmark
  const savedPosition = tour.position;
  state.directBasis = null;

  const syncPixel = new Uint8Array(4);
  const gpuSync = (): void => {
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, syncPixel);
  };

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
    renderView(
      tour.interpolatedBasis,
      state.mainView.canvas.width,
      state.mainView.canvas.height,
      0,
      state.camera,
    );
    gpuSync();
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
    renderView(
      tour.interpolatedBasis,
      state.mainView.canvas.width,
      state.mainView.canvas.height,
      0,
      state.camera,
    );
    gpuSync();
    frameTimes[i] = performance.now() - t0;
  }

  // Restore original position
  tour.position = savedPosition;

  postMain({ type: 'benchmarkResult', frameTimes, numPoints, numDims }, [frameTimes.buffer]);
};

// ─── Main thread messages ──────────────────────────────────────────────────

const handleMessage = (msg: MainToGpu): void => {
  if (!state) return;

  if (msg.type === 'setBases') {
    const { bases } = msg;
    if (bases.length === 0) return;
    const dims = bases[0]!.length / 2;

    state.tour = {
      bases,
      arcLengths: computeArcLengths(bases, dims),
      dims,
      position: state.tour?.position ?? 0,
      interpolatedBasis: new Float32Array(dims * 2),
    };

    if (state.dataTexture) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'addPreviewCanvas') {
    const ctx2d = msg.canvas.getContext('2d');
    if (!ctx2d) return;
    state.previewCanvases.set(msg.id, { canvas: msg.canvas, ctx2d });
    // Render immediately if data + tour exist
    if (state.dataTexture && state.tour && msg.id < state.tour.bases.length) {
      renderOnePreview(msg.id, state.tour.bases[msg.id]!);
    }
    return;
  }

  if (msg.type === 'removePreviewCanvas') {
    state.previewCanvases.delete(msg.id);
    return;
  }

  if (msg.type === 'setTourPosition') {
    if (!state.tour) return;
    state.tour.position = msg.position;
    state.directBasis = null;
    throttledRenderMainView();
    return;
  }

  if (msg.type === 'setStyle') {
    state.style = { pointSize: msg.pointSize, opacity: msg.opacity, color: msg.color };
    if ((state.tour || state.directBasis) && state.dataTexture) {
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
    };
    renderMainView();
    return;
  }

  if (msg.type === 'resize') {
    if (msg.dpr !== undefined) {
      state.dpr = msg.dpr;
    }
    if (msg.viewIndex === 0) {
      state.mainView.canvas.width = msg.width;
      state.mainView.canvas.height = msg.height;
      if ((state.tour || state.directBasis) && state.dataTexture) {
        renderMainView();
      }
    }
    return;
  }

  if (msg.type === 'render') {
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setDirectBasis') {
    if (!state.dataTexture) return;
    state.directBasis = msg.basis;
    renderView(
      msg.basis,
      state.mainView.canvas.width,
      state.mainView.canvas.height,
      0,
      state.camera,
    );
    postMain({ type: 'rendered', viewIndex: 0 });
    return;
  }

  if (msg.type === 'clearColors') {
    if (state.colorLutTexture) {
      state.mainView.gl.deleteTexture(state.colorLutTexture);
      state.colorLutTexture = null;
    }
    state.colorMode = 0;
    state.colorColumnIndex = 0;
    state.colorMin = 0;
    state.colorRange = 1;
    state.colorNumStops = 0;
    state.activeCatColumnName = null;
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setBackgroundColor') {
    state.backgroundColor = msg.color;
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setMaxPoints') {
    state.maxPoints = msg.maxPoints;
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setSelectionMask') {
    const gl = state.mainView.gl;
    uploadSelectionMask(gl, msg.mask);
    state.styleFlags.useSelectionMask = true;
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    // Emit mask so onPointSelectionChange subscribers see click selections
    const maskCopy = new Uint32Array(msg.mask);
    postMain({ type: 'selectionResult', mask: maskCopy }, [maskCopy.buffer]);
    return;
  }

  if (msg.type === 'clearSelectionMask') {
    if (state.selectionTexture) {
      state.mainView.gl.deleteTexture(state.selectionTexture);
      state.selectionTexture = null;
    }
    state.styleFlags.useSelectionMask = false;
    if ((state.tour || state.directBasis) && state.dataTexture) {
      renderAllViews();
    }
    postMain({ type: 'selectionResult', mask: new Uint32Array(0) });
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
    // TODO: CPU PCA via jacobi.ts — requires importing and calling on dataBuffers
    postMain({ type: 'error', message: 'PCA not yet implemented in WebGL backend' });
    return;
  }

  if (msg.type === 'lassoSelect') {
    handleLassoSelect(msg.polygon);
    return;
  }

  if (msg.type === 'getProjectedPositions') {
    if (!state || !state.normMins || !state.normRanges || state.numPoints === 0) {
      postMain({ type: 'projectedPositions', positions: new Float32Array(0) }, []);
      return;
    }

    const { numPoints, numDims, dataBuffers } = state;

    // Get current basis
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
    if (!currentBasis) {
      postMain({ type: 'projectedPositions', positions: new Float32Array(0) }, []);
      return;
    }

    const { weights, biasX, biasY } = computeAdjustedBasis(
      currentBasis,
      state.normMins,
      state.normRanges,
      numDims,
    );

    // Project all points on CPU
    const positions = new Float32Array(numPoints * 2);
    for (let i = 0; i < numPoints; i++) {
      let px = biasX;
      let py = biasY;
      for (let d = 0; d < numDims; d++) {
        const raw = dataBuffers[d]![i]!;
        px += raw * weights[d]!;
        py += raw * weights[numDims + d]!;
      }
      positions[i * 2] = px;
      positions[i * 2 + 1] = py;
    }

    postMain({ type: 'projectedPositions', positions }, [positions.buffer]);
    return;
  }

  if (msg.type === 'getPointData') {
    if (!state || state.numPoints === 0) {
      postMain({
        type: 'pointData',
        pointIndex: msg.pointIndex,
        numericValues: {},
        categoricalValues: {},
      });
      return;
    }

    const { numDims, dataBuffers } = state;
    const idx = msg.pointIndex;
    if (idx < 0 || idx >= state.numPoints) {
      postMain({
        type: 'pointData',
        pointIndex: idx,
        numericValues: {},
        categoricalValues: {},
      });
      return;
    }

    const numericValues: Record<string, number> = {};
    for (let d = 0; d < numDims; d++) {
      numericValues[String(d)] = dataBuffers[d]![idx]!;
    }

    const categoricalValues: Record<string, number> = {};
    for (const [name, indices] of state.categoricalBuffers) {
      categoricalValues[name] = indices[idx]!;
    }

    postMain({ type: 'pointData', pointIndex: idx, numericValues, categoricalValues });
    return;
  }

  if (msg.type === 'benchmark') {
    handleBenchmark(msg.numFrames);
    return;
  }

  if (msg.type === 'getMetrics') {
    handleGetMetrics();
    return;
  }
};

// ─── Init + entry point ──────────────────────────────────────────────────

self.onmessage = (event: MessageEvent<MainToGpu>): void => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      const mainCanvas = msg.canvas;
      const mainGl = mainCanvas.getContext('webgl2', {
        alpha: true,
        premultipliedAlpha: true,
        antialias: false,
        preserveDrawingBuffer: false,
      });
      if (!mainGl) throw new Error('WebGL2 not supported');

      const floatExt = mainGl.getExtension('EXT_color_buffer_float');
      if (!floatExt) {
        console.warn('EXT_color_buffer_float not available — HDR rendering may fail');
      }
      mainGl.getExtension('OES_texture_float_linear');

      const gl = mainGl;

      const pointProgram = linkProgram(gl, pointVert, pointFrag);
      const pointLocs: PointLocations = {
        u_data: getUniform(gl, pointProgram, 'u_data'),
        u_adjBasisX: getUniform(gl, pointProgram, 'u_adjBasisX'),
        u_adjBasisY: getUniform(gl, pointProgram, 'u_adjBasisY'),
        u_bias: getUniform(gl, pointProgram, 'u_bias'),
        u_pointSize: getUniform(gl, pointProgram, 'u_pointSize'),
        u_opacity: getUniform(gl, pointProgram, 'u_opacity'),
        u_color: getUniform(gl, pointProgram, 'u_color'),
        u_useSubtractive: getUniform(gl, pointProgram, 'u_useSubtractive'),
        u_pan: getUniform(gl, pointProgram, 'u_pan'),
        u_zoom: getUniform(gl, pointProgram, 'u_zoom'),
        u_aspect: getUniform(gl, pointProgram, 'u_aspect'),
        u_viewportHeight: getUniform(gl, pointProgram, 'u_viewportHeight'),
        u_insetOffsetY: getUniform(gl, pointProgram, 'u_insetOffsetY'),
        u_insetZoom: getUniform(gl, pointProgram, 'u_insetZoom'),
        u_numPoints: getUniform(gl, pointProgram, 'u_numPoints'),
        u_numDims: getUniform(gl, pointProgram, 'u_numDims'),
        u_maxPoints: getUniform(gl, pointProgram, 'u_maxPoints'),
        u_texWidth: getUniform(gl, pointProgram, 'u_texWidth'),
        u_colorMode: getUniform(gl, pointProgram, 'u_colorMode'),
        u_colorColumnIndex: getUniform(gl, pointProgram, 'u_colorColumnIndex'),
        u_colorMin: getUniform(gl, pointProgram, 'u_colorMin'),
        u_colorRange: getUniform(gl, pointProgram, 'u_colorRange'),
        u_colorNumStops: getUniform(gl, pointProgram, 'u_colorNumStops'),
        u_colorLutTex: getUniform(gl, pointProgram, 'u_colorLutTex'),
        u_catIndexTex: getUniform(gl, pointProgram, 'u_catIndexTex'),
        u_useSelectionMask: getUniform(gl, pointProgram, 'u_useSelectionMask'),
        u_selectionTex: getUniform(gl, pointProgram, 'u_selectionTex'),
      };

      const tonemapProgram = linkProgram(gl, tonemapVert, tonemapFrag);
      const tonemapLocs: TonemapLocations = {
        u_hdrTexture: getUniform(gl, tonemapProgram, 'u_hdrTexture'),
        u_mode: getUniform(gl, tonemapProgram, 'u_mode'),
      };

      state = {
        mainView: { canvas: mainCanvas, gl: mainGl },
        previewCanvases: new Map(),
        pointProgram,
        pointLocs,
        tonemapProgram,
        tonemapLocs,
        hdrFbos: [],
        hdrTextures: [],
        hdrSizes: [],
        blitFbo: null,
        blitTexture: null,
        blitSize: null,
        dataTexture: null,
        colorLutTexture: null,
        catIndexTextures: new Map(),
        selectionTexture: null,
        numPoints: 0,
        numDims: 0,
        dataBuffers: [],
        normMins: null,
        normRanges: null,
        categoricalBuffers: new Map(),
        tour: null,
        style: { pointSize: 'auto', opacity: 'auto', color: [0.25, 0.5, 0.9] },
        styleFlags: { useSelectionMask: false },
        colorMode: 0,
        colorColumnIndex: 0,
        colorMin: 0,
        colorRange: 1,
        colorNumStops: 0,
        activeCatColumnName: null,
        camera: {
          panX: 0,
          panY: 0,
          zoom: msg.zoom,
          aspect: 1,
          viewportHeight: 1,
          insetOffsetY: 0,
          insetZoom: 1,
        },
        directBasis: null,
        backgroundColor: [0, 0, 0],
        dpr: msg.dpr,
        maxPoints: 0,
        texWidth: 0,
      };

      msg.dataPort.onmessage = onDataMessage;

      // Replay buffered messages
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

  if (pendingMessages) {
    pendingMessages.push(msg);
    return;
  }

  handleMessage(msg);
};
