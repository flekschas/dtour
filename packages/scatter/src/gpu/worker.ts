/// <reference lib="webworker" />

import type { DataToGpu } from '../data/messages.ts';
import { type CanvasView, configureCanvas, renderPoints } from '../renderer.ts';
import { computeArcLengths, interpolateAtPosition } from '../tour/arc-length.ts';
import { initDevice } from './device.ts';
import type { GpuToMain, MainToGpu } from './messages.ts';
import {
  type CameraState,
  type PointStyle,
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
  camera: CameraState;
};

let state: GpuState | null = null;

const postMain = (msg: GpuToMain): void => {
  self.postMessage(msg);
};

// ─── Projection + Render helpers ──────────────────────────────────────────

/** Write a basis to the GPU basis buffer and dispatch compute + render for one view. */
const projectAndRender = (basis: Float32Array, viewIndex: number, camera: CameraState): void => {
  if (!state || !state.projectionResources || !state.projectionBindGroup || !state.renderBindGroup)
    return;

  const { device, projectionPipeline, projectionResources, projectionBindGroup, numPoints } = state;

  // Upload basis
  device.queue.writeBuffer(projectionResources.basisBuffer, 0, basis as Float32Array<ArrayBuffer>);

  // Write the caller-supplied camera (with aspect computed from the target canvas)
  const canvas = state.views[viewIndex]!.canvas;
  const aspect = canvas.width / canvas.height || 1;
  writeCamera(device, state.pointPipeline.cameraBuffer, { ...camera, aspect });

  // Dispatch compute then render — single submit for efficiency
  const computeCmd = dispatchProjection(device, projectionPipeline, projectionBindGroup, numPoints);
  const renderCmd = renderPoints(
    device,
    state.views[viewIndex]!,
    state.pointPipeline,
    state.renderBindGroup,
    numPoints,
  );

  device.queue.submit([computeCmd, renderCmd]);
};

/** Render all views with their keyframe bases. */
const renderAllViews = (): void => {
  if (!state?.tour || !state.projectionResources) return;

  const { tour, views } = state;

  const identityCamera: CameraState = { panX: 0, panY: 0, zoom: 1, aspect: 1 };

  // Render main view (index 0) with interpolated basis at current position
  interpolateAtPosition(
    tour.interpolatedBasis,
    tour.bases,
    tour.arcLengths,
    tour.dims,
    tour.position,
  );
  projectAndRender(tour.interpolatedBasis, 0, state.camera);
  postMain({ type: 'rendered', viewIndex: 0 });

  // Render preview views (index 1+) with their keyframe bases and identity camera
  for (let i = 1; i < views.length && i - 1 < tour.bases.length; i++) {
    try {
      projectAndRender(tour.bases[i - 1]!, i, identityCamera);
    } catch (err) {
      postMain({
        type: 'error',
        message: `Preview ${i} render failed: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
    postMain({ type: 'rendered', viewIndex: i });
  }
};

/** Re-render only the main view at the current tour position. */
const renderMainView = (): void => {
  if (!state?.tour || !state.projectionResources) return;

  const { tour } = state;
  interpolateAtPosition(
    tour.interpolatedBasis,
    tour.bases,
    tour.arcLengths,
    tour.dims,
    tour.position,
  );
  projectAndRender(tour.interpolatedBasis, 0, state.camera);
  postMain({ type: 'rendered', viewIndex: 0 });
};

// ─── Build bind groups after data + pipeline are ready ────────────────────

const rebuildBindGroups = (): void => {
  if (!state || !state.projectionResources) return;

  state.projectionBindGroup = createProjectionBindGroup(
    state.device,
    state.projectionPipeline.bindGroupLayout,
    state.projectionPipeline.paramsBuffer,
    state.projectionResources,
  );

  state.renderBindGroup = createPointBindGroup(
    state.device,
    state.pointPipeline.bindGroupLayout,
    state.pointPipeline.uniformBuffer,
    state.projectionResources.projectedBuffer,
    state.pointPipeline.cameraBuffer,
  );
};

// ─── Data from Data Worker ─────────────────────────────────────────────────

const onDataMessage = (event: MessageEvent<DataToGpu>): void => {
  if (event.data.type !== 'data' || !state) return;

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
    const numPreviews = state.views.length - 1;
    const count = Math.max(1, numPreviews);
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
        camera: { panX: 0, panY: 0, zoom: 1, aspect: 1 },
      };

      msg.dataPort.onmessage = onDataMessage;

      postMain({
        type: 'ready',
        limits: {
          maxBufferSize: device.limits.maxBufferSize,
          maxTextureDimension2D: device.limits.maxTextureDimension2D,
        },
      });
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
    const { bases, dims } = msg;
    if (bases.length === 0) return;

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
    renderMainView();
    return;
  }

  if (msg.type === 'setStyle') {
    state.style = { pointSize: msg.pointSize, opacity: msg.opacity, color: msg.color };
    writeUniforms(state.device, state.pointPipeline.uniformBuffer, state.style);
    if (state.tour && state.projectionResources) {
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
    };
    renderMainView();
    return;
  }

  if (msg.type === 'resize') {
    const view = state.views[msg.viewIndex];
    if (view) {
      view.canvas.width = msg.width;
      view.canvas.height = msg.height;
      if (state.tour && state.projectionResources) {
        renderAllViews();
      }
    }
    return;
  }

  if (msg.type === 'render') {
    if (state.tour && state.projectionResources) {
      renderAllViews();
    }
    return;
  }
};
