/// <reference lib="webworker" />

import type { DataToGpu } from '../data/messages.ts';
import { type CanvasView, configureCanvas, renderPoints } from '../renderer.ts';
import { computeArcLengths, interpolateAtPosition } from '../tour/arc-length.ts';
import { initDevice } from './device.ts';
import type { GpuToMain, MainToGpu } from './messages.ts';
import {
  type CameraState,
  type PointStyle,
  type StyleFlags,
  createPointBindGroup,
  createPointPipeline,
  writeCamera,
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
  // Compute pipeline
  projectionPipeline: ReturnType<typeof createProjectionPipeline>;
  projectionResources: ProjectionResources | null;
  projectionBindGroup: GPUBindGroup | null;
  // Data
  numPoints: number;
  numDims: number;
  // Tour
  tour: TourState | null;
  // Style
  style: PointStyle;
  styleFlags: StyleFlags;
  camera: CameraState;
  // Per-point color + selection
  colorBuffer: GPUBuffer | null;
  selectionBuffer: GPUBuffer | null;
  // Sticky direct basis — when set, renderMainView/renderAllViews use this
  // for the main view instead of tour interpolation. Cleared by setTourPosition.
  directBasis: Float32Array | null;
};

let state: GpuState | null = null;

const postMain = (msg: GpuToMain): void => {
  self.postMessage(msg);
};

// ─── Projection + Render helpers ──────────────────────────────────────────

/** Write a basis to the GPU basis buffer and dispatch compute + render for one view. */
const projectAndRender = (
  basis: Float32Array,
  viewIndex: number,
  camera: CameraState,
  projBindGroup: GPUBindGroup,
  renBindGroup: GPUBindGroup,
): void => {
  if (!state || !state.projectionResources) return;

  const { device, projectionPipeline, projectionResources, numPoints } = state;

  // Upload basis
  device.queue.writeBuffer(projectionResources.basisBuffer, 0, basis as Float32Array<ArrayBuffer>);

  // Write the caller-supplied camera (with aspect + viewport height from the target canvas)
  const canvas = state.views[viewIndex]!.canvas;
  const aspect = canvas.width / canvas.height || 1;
  writeCamera(device, state.pointPipeline.cameraBuffer, {
    ...camera,
    aspect,
    viewportHeight: canvas.height,
  });

  // Dispatch compute then render — single submit for efficiency
  const computeCmd = dispatchProjection(device, projectionPipeline, projBindGroup, numPoints);
  const renderCmd = renderPoints(
    device,
    state.views[viewIndex]!,
    state.pointPipeline,
    renBindGroup,
    numPoints,
  );

  device.queue.submit([computeCmd, renderCmd]);
};

// ─── Deferred preview rendering ───────────────────────────────────────────
// Preview rendering is coalesced via setTimeout(0) so rapid-fire state
// updates (setBases, setStyle, render, …) only produce a single
// getCurrentTexture + submit per preview canvas, avoiding expired-texture
// issues that can silently drop frames on some WebGPU implementations.

let previewRenderPending = false;

/** Render a single preview view with an ephemeral projected buffer. */
const renderOnePreview = (viewIndex: number, basis: Float32Array): void => {
  if (!state?.projectionResources) return;

  const { device, projectionPipeline, pointPipeline, projectionResources, numPoints } = state;
  const colorBuf = state.colorBuffer ?? pointPipeline.defaultColorBuffer;
  const selBuf = state.selectionBuffer ?? pointPipeline.defaultSelectionBuffer;

  // Ephemeral projected buffer — destroyed after submit
  const ephemeralProjBuffer = device.createBuffer({
    label: `preview-proj-${viewIndex}`,
    size: numPoints * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const ephemeralProjBindGroup = createProjectionBindGroup(
    device,
    projectionPipeline.bindGroupLayout,
    projectionPipeline.paramsBuffer,
    { ...projectionResources, projectedBuffer: ephemeralProjBuffer },
  );

  const ephemeralRenderBindGroup = createPointBindGroup(
    device,
    pointPipeline.bindGroupLayout,
    pointPipeline.uniformBuffer,
    ephemeralProjBuffer,
    pointPipeline.cameraBuffer,
    colorBuf,
    selBuf,
  );

  projectAndRender(
    basis,
    viewIndex,
    { panX: 0, panY: 0, zoom: 1, aspect: 1, viewportHeight: 1, insetOffsetY: 0, insetZoom: 1 },
    ephemeralProjBindGroup,
    ephemeralRenderBindGroup,
  );

  // GPU retains backing memory until submitted work completes
  ephemeralProjBuffer.destroy();
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
  if (!state?.projectionResources || !state.projectionBindGroup || !state.renderBindGroup) return;

  if (state.directBasis) {
    projectAndRender(
      state.directBasis,
      0,
      state.camera,
      state.projectionBindGroup,
      state.renderBindGroup,
    );
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
  projectAndRender(
    tour.interpolatedBasis,
    0,
    state.camera,
    state.projectionBindGroup,
    state.renderBindGroup,
  );
  postMain({ type: 'rendered', viewIndex: 0 });
};

// ─── Build bind groups after data + pipeline are ready ────────────────────

const rebuildBindGroups = (): void => {
  if (!state || !state.projectionResources) return;

  const { device, projectionPipeline, pointPipeline, projectionResources } = state;
  const colorBuf = state.colorBuffer ?? pointPipeline.defaultColorBuffer;
  const selBuf = state.selectionBuffer ?? pointPipeline.defaultSelectionBuffer;

  // Main view bind groups (projected buffer used for lasso readback)
  state.projectionBindGroup = createProjectionBindGroup(
    device,
    projectionPipeline.bindGroupLayout,
    projectionPipeline.paramsBuffer,
    projectionResources,
  );

  state.renderBindGroup = createPointBindGroup(
    device,
    pointPipeline.bindGroupLayout,
    pointPipeline.uniformBuffer,
    projectionResources.projectedBuffer,
    pointPipeline.cameraBuffer,
    colorBuf,
    selBuf,
  );
};

// ─── Data from Data Worker ─────────────────────────────────────────────────

const onDataMessage = (event: MessageEvent<DataToGpu>): void => {
  if (!state) return;

  if (event.data.type === 'colors') {
    // Per-point color data from data worker
    const { colors } = event.data;
    const { device } = state;

    // Destroy old color buffer if it's not the default
    if (state.colorBuffer) {
      state.colorBuffer.destroy();
    }

    state.colorBuffer = device.createBuffer({
      label: 'point-colors',
      size: colors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(state.colorBuffer, 0, colors as Uint32Array<ArrayBuffer>);

    state.styleFlags.usePerPointColor = true;
    writeUniforms(device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (event.data.type !== 'data') return;

  const { dims, rows, buffers, mins, ranges } = event.data;

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
  // Clean up previous color/selection buffers
  if (state.colorBuffer) {
    state.colorBuffer.destroy();
    state.colorBuffer = null;
  }
  if (state.selectionBuffer) {
    state.selectionBuffer.destroy();
    state.selectionBuffer = null;
  }
  state.styleFlags = { usePerPointColor: false, useSelectionMask: false };
  writeUniforms(state.device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);

  const { device, projectionPipeline } = state;

  // Create new projection resources
  const res = createProjectionResources(device, projectionPipeline, rows, dims);

  // Upload each column contiguously into the concatenated data buffer
  for (let d = 0; d < dims; d++) {
    device.queue.writeBuffer(
      res.dataBuffer,
      d * rows * 4,
      buffers[d]! as Float32Array<ArrayBuffer>,
    );
  }

  // Upload norm params: [min, range] pairs as vec2f array
  const normData = new Float32Array(dims * 2);
  for (let d = 0; d < dims; d++) {
    normData[d * 2] = mins[d]!;
    normData[d * 2 + 1] = ranges[d]!;
  }
  device.queue.writeBuffer(res.normParamsBuffer, 0, normData);

  // Write projection params
  writeProjectionParams(device, projectionPipeline.paramsBuffer, rows, dims, 2.0);

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

self.onmessage = async (event: MessageEvent<MainToGpu>): Promise<void> => {
  const msg = event.data;

  if (msg.type === 'init') {
    try {
      const { device } = await initDevice();

      const views = msg.canvases.map((canvas) => configureCanvas(canvas, device));
      const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

      const pointPipeline = createPointPipeline(device, canvasFormat);
      const projectionPipeline = createProjectionPipeline(device);

      state = {
        device,
        views,
        pointPipeline,
        renderBindGroup: null,
        projectionPipeline,
        projectionResources: null,
        projectionBindGroup: null,
        numPoints: 0,
        numDims: 0,
        tour: null,
        style: { pointSize: 0.012, opacity: 0.7, color: [0.25, 0.5, 0.9] },
        styleFlags: { usePerPointColor: false, useSelectionMask: false },
        camera: { panX: 0, panY: 0, zoom: 1, aspect: 1, viewportHeight: 1, insetOffsetY: 0, insetZoom: 1 },
        colorBuffer: null,
        selectionBuffer: null,
        directBasis: null,
      };

      msg.dataPort.onmessage = onDataMessage;

      postMain({ type: 'ready' });
    } catch (err) {
      postMain({
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      });
    }
    return;
  }

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
    renderMainView();
    return;
  }

  if (msg.type === 'setStyle') {
    state.style = { pointSize: msg.pointSize, opacity: msg.opacity, color: msg.color };
    writeUniforms(state.device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
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
    };
    renderMainView();
    return;
  }

  if (msg.type === 'resize') {
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
    if (!state.projectionResources || !state.projectionBindGroup || !state.renderBindGroup) return;
    state.directBasis = msg.basis;
    projectAndRender(msg.basis, 0, state.camera, state.projectionBindGroup, state.renderBindGroup);
    postMain({ type: 'rendered', viewIndex: 0 });
    return;
  }

  if (msg.type === 'setColors') {
    const { colors } = msg;
    const { device } = state;

    if (state.colorBuffer) {
      state.colorBuffer.destroy();
    }

    state.colorBuffer = device.createBuffer({
      label: 'point-colors',
      size: colors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(state.colorBuffer, 0, colors as Uint32Array<ArrayBuffer>);

    state.styleFlags.usePerPointColor = true;
    writeUniforms(device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'clearColors') {
    if (state.colorBuffer) {
      state.colorBuffer.destroy();
      state.colorBuffer = null;
    }
    state.styleFlags.usePerPointColor = false;
    writeUniforms(state.device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'setSelectionMask') {
    const { mask } = msg;
    const { device } = state;

    if (state.selectionBuffer) {
      state.selectionBuffer.destroy();
    }

    state.selectionBuffer = device.createBuffer({
      label: 'selection-mask',
      size: mask.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(state.selectionBuffer, 0, mask as Uint32Array<ArrayBuffer>);

    state.styleFlags.useSelectionMask = true;
    writeUniforms(device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
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
    writeUniforms(state.device, state.pointPipeline.uniformBuffer, state.style, state.styleFlags);
    rebuildBindGroups();

    if ((state.tour || state.directBasis) && state.projectionResources) {
      renderAllViews();
    }
    return;
  }

  if (msg.type === 'lassoSelect') {
    if (!state.projectionResources || state.numPoints === 0) return;

    const { device, numPoints, projectionResources } = state;
    const { polygon } = msg;
    const numVertices = polygon.length / 2;
    if (numVertices < 3) return;

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

      // CPU point-in-polygon (ray casting) for each point
      const mask = new Uint32Array(numPoints);
      for (let i = 0; i < numPoints; i++) {
        const px = projected[i * 2]!;
        const py = projected[i * 2 + 1]!;
        mask[i] = pointInPolygon(px, py, polygon, numVertices) ? 1 : 0;
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
      writeUniforms(device, state!.pointPipeline.uniformBuffer, state!.style, state!.styleFlags);
      rebuildBindGroups();

      if ((state!.tour || state!.directBasis) && state!.projectionResources) {
        renderAllViews();
      }
    });
    return;
  }
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
