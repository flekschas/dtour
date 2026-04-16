import DataWorkerFactory from '../data/worker.ts?worker&inline';
// ?worker&inline tells Vite to bundle each worker + all its imports and embed the
// result as a base64 data URL in the library output. Consumers of @dtour/scatter
// get a single self-contained JS file — no separate worker files to host.
import GpuWorkerFactory from './worker.ts?worker&inline';

import type { DataToMain, MainToData } from '../data/messages.ts';
import type { Metadata } from '../data/types.ts';
import type { GpuToMain, MainToGpu } from './messages.ts';

export type ScatterOptions = {
  /** The main canvas to render into. */
  canvas: HTMLCanvasElement;
  /** Initial camera zoom level. Default 1. */
  zoom?: number;
  /** Device pixel ratio. Default `window.devicePixelRatio ?? 1`. */
  dpr?: number;
};

export type ScatterStatus =
  | { type: 'ready' }
  | { type: 'rendered'; viewIndex: number }
  | { type: 'metadata'; metadata: Metadata }
  | { type: 'error'; message: string }
  | {
      type: 'pcaResult';
      eigenvectors: Float32Array[];
      eigenvalues: Float32Array;
      numDims: number;
    }
  | { type: 'playbackTick'; position: number; frameMs?: number }
  | { type: 'playbackStopped'; frameTimes: Float64Array }
  | {
      type: 'benchmarkResult';
      frameTimes: Float64Array;
      numPoints: number;
      numDims: number;
      stats: {
        avgMs: number;
        fps: number;
        p50Ms: number;
        p95Ms: number;
        minMs: number;
        maxMs: number;
      };
    }
  | {
      type: 'metricsResult';
      gpuMemoryBytes: number;
      numPoints: number;
      numDims: number;
      workerJsHeapBytes: number | null;
    }
  | { type: 'residualPC'; residualPC: Float32Array }
  | { type: 'selectionResult'; mask: Uint32Array }
  | { type: 'projectedPositions'; positions: Float32Array }
  | {
      type: 'pointData';
      pointIndex: number;
      numericValues: Record<string, number>;
      categoricalValues: Record<string, number>;
    };

export type ScatterInstance = {
  /** Transfer an Arrow IPC or Parquet ArrayBuffer for loading. Ownership is transferred. */
  loadData: (buffer: ArrayBuffer) => void;
  /**
   * Set tour basis matrices. Each basis is a p×2 column-major Float32Array:
   * [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
   * where x_i is the x-projection weight for dimension i.
   * dims (p) is inferred as bases[0].length / 2.
   *
   * @param bases - array of basis matrices, one per tour keyframe
   */
  setBases: (bases: Float32Array[]) => void;
  /** Set tour position along the arc-length parameterized path [0, 1]. */
  setTourPosition: (position: number) => void;
  /** Update point rendering style. Use 'auto' for density-adaptive sizing. */
  setStyle: (options: {
    pointSize?: number | 'auto';
    opacity?: number | 'auto';
    color?: [number, number, number];
  }) => void;
  /** Set 2D camera (pan, zoom, and optional viewport inset for toolbar offset). */
  setCamera: (options: {
    pan?: [number, number];
    zoom?: number;
    /** NDC-space Y offset — shifts content to center below toolbar. */
    insetOffsetY?: number;
    /** Zoom multiplier — scales content to fit visible area below toolbar. */
    insetZoom?: number;
  }) => void;
  /** Resize a canvas to the given pixel dimensions (use for DPI-aware sizing). */
  resize: (viewIndex: number, width: number, height: number, dpr?: number) => void;
  /** Request a render of all views. */
  render: () => void;
  /** Set a single basis directly for manual/zen modes. Renders main view only. */
  setDirectBasis: (basis: Float32Array) => void;
  /** Encode a column as per-point colors. Column can be categorical or numeric. */
  encodeColor: (
    column: string,
    palette?: string,
    theme?: 'light' | 'dark',
    colorMap?: Record<string, [number, number, number]>,
  ) => void;
  /** Set the background clear color (RGB 0–1). */
  setBackgroundColor: (color: [number, number, number]) => void;
  /** Clear per-point colors and revert to uniform color. */
  clearColor: () => void;
  /** Select points by column value. Mask is built in the data worker. */
  selectByColumn: (
    column: string,
    opts: { labelIndices?: number[]; valueRanges?: Float32Array },
  ) => void;
  /** Set a bit-packed selection mask (1 bit per point, 32 per u32). Length: ceil(numPoints / 32). */
  setSelectionMask: (mask: Uint32Array) => void;
  /** Lasso select: send NDC polygon, GPU does point-in-polygon test. */
  lassoSelect: (polygon: Float32Array) => void;
  /** Clear selection mask — all points visible. */
  clearSelection: () => void;
  /** Request GPU-accelerated PCA computation. Results arrive via subscribe as 'pcaResult'. */
  computePCA: () => void;
  /** Start worker-driven playback. Worker runs its own rAF loop and posts position updates. */
  startPlayback: (speed: number, direction: 1 | -1) => void;
  /** Stop worker-driven playback. */
  stopPlayback: () => void;
  /** Set max rendered points. 0 = disabled (render all). Uses deterministic hash decimation. */
  setMaxPoints: (n: number) => void;
  /** Enable 3D camera rotation mode. Computes residual PC for the 3rd projected axis. */
  enable3d: () => void;
  /** Disable 3D camera rotation mode. Reverts to standard 2D projection. */
  disable3d: () => void;
  /** Update the 3×3 camera rotation matrix (column-major, 9 floats). */
  set3dRotation: (matrix: Float32Array) => void;
  /** Run a render benchmark on loaded data. Sweeps through the full tour, timing each frame. */
  benchmark: (numFrames?: number) => Promise<{
    frameTimes: Float64Array;
    numPoints: number;
    numDims: number;
    stats: {
      avgMs: number;
      fps: number;
      p50Ms: number;
      p95Ms: number;
      minMs: number;
      maxMs: number;
    };
  }>;
  /** Get a point-in-time snapshot of GPU memory, dataset dimensions, and JS heap usage. */
  getMetrics: () => Promise<{
    gpuMemoryBytes: number;
    numPoints: number;
    numDims: number;
    /** Main-thread JS heap (V8 heap of the calling context). */
    jsHeapUsedBytes: number | null;
    /** GPU-worker JS heap (V8 heap of the render worker). */
    workerJsHeapUsedBytes: number | null;
  }>;
  /** Request projected 2D positions for building a spatial index. Returns N×2 interleaved Float32Array. */
  getProjectedPositions: () => Promise<Float32Array>;
  /** Get column values for a single point by index. Numeric values keyed by dim index string, categorical by column name. */
  getPointData: (pointIndex: number) => Promise<{
    pointIndex: number;
    numericValues: Record<string, number>;
    categoricalValues: Record<string, number>;
  }>;
  /** Add a preview canvas. Ownership is transferred; the worker blits rendered previews to its 2D context. */
  addPreviewCanvas: (id: number, canvas: HTMLCanvasElement) => void;
  /** Remove a previously added preview canvas. */
  removePreviewCanvas: (id: number) => void;
  /** Resize a preview canvas backing store (physical pixels). */
  resizePreview: (id: number, width: number, height: number) => void;
  /** Subscribe to status events from both workers. Returns an unsubscribe function. */
  subscribe: (handler: (status: ScatterStatus) => void) => () => void;
  /** Terminate both workers and release resources. */
  destroy: () => void;
};

const computeStats = (ft: Float64Array) => {
  const avg = ft.reduce((a: number, b: number) => a + b, 0) / ft.length;
  const sorted = Float64Array.from(ft).sort();
  return {
    avgMs: avg,
    fps: 1000 / avg,
    p50Ms: sorted[Math.floor(ft.length * 0.5)]!,
    p95Ms: sorted[Math.floor(ft.length * 0.95)]!,
    minMs: sorted[0]!,
    maxMs: sorted[ft.length - 1]!,
  };
};

const sendToGpu = (worker: Worker, msg: MainToGpu, transfers?: Transferable[]): void => {
  worker.postMessage(msg, transfers ?? []);
};

const sendToData = (worker: Worker, msg: MainToData, transfers?: Transferable[]): void => {
  worker.postMessage(msg, transfers ?? []);
};

/**
 * Create a scatter renderer instance.
 *
 * Instantiates the GPU Worker and Data Worker, connects them via a
 * MessageChannel (so parsed data flows directly to the GPU worker without
 * passing through the main thread), and transfers OffscreenCanvas control
 * to the GPU worker.
 *
 * @example
 * ```ts
 * const scatter = createScatter({ canvases: [mainCanvas, ...previewCanvases] });
 * scatter.subscribe(console.log);
 * scatter.loadData(arrowBuffer);
 * scatter.setBases(bases);
 * scatter.setTourPosition(0.5);
 * ```
 */
export const createScatter = (options: ScatterOptions): ScatterInstance => {
  const {
    canvas,
    zoom: initialZoom = 1,
    dpr = typeof self !== 'undefined' && 'devicePixelRatio' in self ? self.devicePixelRatio : 1,
  } = options;

  const gpuWorker = new GpuWorkerFactory();
  const dataWorker = new DataWorkerFactory();

  // Direct MessageChannel between data worker and GPU worker.
  const channel = new MessageChannel();

  const subscribers = new Set<(status: ScatterStatus) => void>();

  const emit = (status: ScatterStatus): void => {
    for (const handler of subscribers) {
      handler(status);
    }
  };

  gpuWorker.onmessage = (event: MessageEvent<GpuToMain>): void => {
    const msg = event.data;
    // Enrich benchmark results with computed stats before emitting
    if (msg.type === 'benchmarkResult') {
      const ft = msg.frameTimes;
      emit({ ...msg, stats: computeStats(ft) });
      return;
    }
    emit(msg as ScatterStatus);
  };

  gpuWorker.onerror = (err): void => {
    emit({ type: 'error', message: err.message });
  };

  dataWorker.onmessage = (event: MessageEvent<DataToMain>): void => {
    if (event.data.type === 'metadata') {
      emit({ type: 'metadata', metadata: event.data.metadata });
    } else {
      emit(event.data);
    }
  };

  dataWorker.onerror = (err): void => {
    emit({ type: 'error', message: err.message });
  };

  // Transfer OffscreenCanvas control to GPU worker (main canvas only)
  const offscreen = canvas.transferControlToOffscreen();

  sendToData(dataWorker, { type: 'init', gpuPort: channel.port1 }, [channel.port1]);
  sendToGpu(
    gpuWorker,
    { type: 'init', canvas: offscreen, dataPort: channel.port2, zoom: initialZoom, dpr },
    [offscreen, channel.port2],
  );

  // Track current style/camera for merging partial updates
  let currentStyle: {
    pointSize: number | 'auto';
    opacity: number | 'auto';
    color: [number, number, number];
  } = {
    pointSize: 'auto',
    opacity: 'auto',
    color: [0.25, 0.5, 0.9],
  };
  let currentCamera = {
    pan: [0, 0] as [number, number],
    zoom: initialZoom,
    insetOffsetY: 0,
    insetZoom: 1,
  };

  const loadData = (buffer: ArrayBuffer): void => {
    sendToData(dataWorker, { type: 'load', buffer }, [buffer]);
  };

  const setBases = (bases: Float32Array[]): void => {
    // Transfer ownership of basis buffers for zero-copy
    const transfers = bases.map((b) => b.buffer);
    sendToGpu(gpuWorker, { type: 'setBases', bases }, transfers);
  };

  const setTourPosition = (position: number): void => {
    sendToGpu(gpuWorker, { type: 'setTourPosition', position });
  };

  const setStyle = (opts: {
    pointSize?: number | 'auto';
    opacity?: number | 'auto';
    color?: [number, number, number];
  }): void => {
    currentStyle = { ...currentStyle, ...opts };
    sendToGpu(gpuWorker, {
      type: 'setStyle',
      pointSize: currentStyle.pointSize,
      opacity: currentStyle.opacity,
      color: currentStyle.color,
    });
  };

  const setCamera = (opts: {
    pan?: [number, number];
    zoom?: number;
    insetOffsetY?: number;
    insetZoom?: number;
  }): void => {
    currentCamera = { ...currentCamera, ...opts };
    sendToGpu(gpuWorker, {
      type: 'setCamera',
      pan: currentCamera.pan,
      zoom: currentCamera.zoom,
      insetOffsetY: currentCamera.insetOffsetY,
      insetZoom: currentCamera.insetZoom,
    });
  };

  const resize = (viewIndex: number, width: number, height: number, dpr?: number): void => {
    sendToGpu(gpuWorker, { type: 'resize', viewIndex, width, height, dpr });
  };

  const render = (): void => {
    sendToGpu(gpuWorker, { type: 'render' });
  };

  const setDirectBasis = (basis: Float32Array): void => {
    sendToGpu(gpuWorker, { type: 'setDirectBasis', basis }, [basis.buffer]);
  };

  const encodeColor = (
    column: string,
    palette?: string,
    theme?: 'light' | 'dark',
    colorMap?: Record<string, [number, number, number]>,
  ): void => {
    const msg: MainToData = {
      type: 'encodeColor',
      column,
      ...(palette ? { palette } : {}),
      ...(theme ? { theme } : {}),
      ...(colorMap ? { colorMap } : {}),
    };
    sendToData(dataWorker, msg);
  };

  const setBackgroundColor = (color: [number, number, number]): void => {
    sendToGpu(gpuWorker, { type: 'setBackgroundColor', color });
  };

  const clearColor = (): void => {
    sendToGpu(gpuWorker, { type: 'clearColors' });
  };

  const selectByColumn = (
    column: string,
    opts: { labelIndices?: number[]; valueRanges?: Float32Array },
  ): void => {
    // Clone valueRanges before transferring so the caller's buffer isn't detached
    const ranges = opts.valueRanges ? new Float32Array(opts.valueRanges) : undefined;
    const transfers: Transferable[] = [];
    if (ranges) transfers.push(ranges.buffer);
    sendToData(
      dataWorker,
      { type: 'selectByColumn', column, labelIndices: opts.labelIndices, valueRanges: ranges },
      transfers,
    );
  };

  const setSelectionMask = (mask: Uint32Array): void => {
    sendToGpu(gpuWorker, { type: 'setSelectionMask', mask }, [mask.buffer]);
  };

  const lassoSelect = (polygon: Float32Array): void => {
    sendToGpu(gpuWorker, { type: 'lassoSelect', polygon }, [polygon.buffer]);
  };

  const clearSelection = (): void => {
    sendToGpu(gpuWorker, { type: 'clearSelectionMask' });
  };

  const computePCA = (): void => {
    sendToGpu(gpuWorker, { type: 'computePCA' });
  };

  const startPlayback = (speed: number, direction: 1 | -1): void => {
    sendToGpu(gpuWorker, { type: 'startPlayback', speed, direction });
  };

  const stopPlayback = (): void => {
    sendToGpu(gpuWorker, { type: 'stopPlayback' });
  };

  const setMaxPoints = (n: number): void => {
    sendToGpu(gpuWorker, { type: 'setMaxPoints', maxPoints: n });
  };

  const enable3d = (): void => {
    sendToGpu(gpuWorker, { type: 'enable3d' });
  };

  const disable3d = (): void => {
    sendToGpu(gpuWorker, { type: 'disable3d' });
  };

  const set3dRotation = (matrix: Float32Array): void => {
    sendToGpu(gpuWorker, { type: 'set3dRotation', matrix });
  };

  const benchmark = (numFrames = 120) => {
    return new Promise<{
      frameTimes: Float64Array;
      numPoints: number;
      numDims: number;
      stats: {
        avgMs: number;
        fps: number;
        p50Ms: number;
        p95Ms: number;
        minMs: number;
        maxMs: number;
      };
    }>((resolve) => {
      const unsub = subscribe((s: ScatterStatus) => {
        if (s.type !== 'benchmarkResult') return;
        unsub();
        resolve(s);
      });
      sendToGpu(gpuWorker, { type: 'benchmark', numFrames });
    });
  };

  const getMetrics = () => {
    return new Promise<{
      gpuMemoryBytes: number;
      numPoints: number;
      numDims: number;
      jsHeapUsedBytes: number | null;
      workerJsHeapUsedBytes: number | null;
    }>((resolve) => {
      const unsub = subscribe((s: ScatterStatus) => {
        if (s.type !== 'metricsResult') return;
        unsub();
        const mainHeapSize =
          typeof performance !== 'undefined' && 'memory' in performance
            ? (performance as unknown as { memory: { usedJSHeapSize: number } }).memory
                .usedJSHeapSize
            : undefined;
        const jsHeapUsedBytes = Number.isFinite(mainHeapSize) ? (mainHeapSize as number) : null;
        const workerJsHeapUsedBytes = Number.isFinite(s.workerJsHeapBytes as number)
          ? s.workerJsHeapBytes
          : null;
        resolve({ ...s, jsHeapUsedBytes, workerJsHeapUsedBytes });
      });
      sendToGpu(gpuWorker, { type: 'getMetrics' });
    });
  };

  const subscribe = (handler: (status: ScatterStatus) => void): (() => void) => {
    subscribers.add(handler);
    return () => subscribers.delete(handler);
  };

  // Latest-wins: only one getProjectedPositions request can be in-flight at a time
  let pendingPositions: { unsub: () => void } | undefined;

  const getProjectedPositions = () => {
    if (pendingPositions) pendingPositions.unsub();
    return new Promise<Float32Array>((resolve) => {
      const unsub = subscribe((s: ScatterStatus) => {
        if (s.type !== 'projectedPositions') return;
        unsub();
        pendingPositions = undefined;
        resolve(s.positions);
      });
      pendingPositions = { unsub };
      sendToGpu(gpuWorker, { type: 'getProjectedPositions' });
    });
  };

  const getPointData = (pointIndex: number) => {
    return new Promise<{
      pointIndex: number;
      numericValues: Record<string, number>;
      categoricalValues: Record<string, number>;
    }>((resolve) => {
      const unsub = subscribe((s: ScatterStatus) => {
        if (s.type !== 'pointData' || s.pointIndex !== pointIndex) return;
        unsub();
        resolve(s);
      });
      sendToGpu(gpuWorker, { type: 'getPointData', pointIndex });
    });
  };

  const addPreviewCanvas = (id: number, canvas: HTMLCanvasElement): void => {
    const offscreen = canvas.transferControlToOffscreen();
    sendToGpu(gpuWorker, { type: 'addPreviewCanvas', id, canvas: offscreen }, [offscreen]);
  };

  const removePreviewCanvas = (id: number): void => {
    sendToGpu(gpuWorker, { type: 'removePreviewCanvas', id });
  };

  const resizePreview = (id: number, width: number, height: number): void => {
    sendToGpu(gpuWorker, { type: 'resizePreview', id, width, height });
  };

  const destroy = (): void => {
    gpuWorker.terminate();
    dataWorker.terminate();
    subscribers.clear();
  };

  return {
    loadData,
    setBases,
    setTourPosition,
    setStyle,
    setCamera,
    resize,
    render,
    setDirectBasis,
    encodeColor,
    setBackgroundColor,
    clearColor,
    selectByColumn,
    setSelectionMask,
    lassoSelect,
    clearSelection,
    getProjectedPositions,
    getPointData,
    computePCA,
    startPlayback,
    stopPlayback,
    setMaxPoints,
    enable3d,
    disable3d,
    set3dRotation,
    benchmark,
    getMetrics,
    addPreviewCanvas,
    removePreviewCanvas,
    resizePreview,
    subscribe,
    destroy,
  };
};
