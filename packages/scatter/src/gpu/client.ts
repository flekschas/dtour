import DataWorkerFactory from '../data/worker.ts?worker&inline';
// ?worker&inline tells Vite to bundle each worker + all its imports and embed the
// result as a base64 data URL in the library output. Consumers of @dtour/scatter
// get a single self-contained JS file — no separate worker files to host.
import GpuWorkerFactory from './worker.ts?worker&inline';

import type { DataToMain, MainToData } from '../data/messages.ts';
import type { Metadata } from '../data/types.ts';
import type { GpuToMain, MainToGpu } from './messages.ts';

export type ScatterOptions = {
  /** All canvases to render into. Index 0 = main view, 1+ = previews. */
  canvases: HTMLCanvasElement[];
};

export type ScatterStatus =
  | { type: 'ready'; limits: { maxBufferSize: number; maxTextureDimension2D: number } }
  | { type: 'rendered'; viewIndex: number }
  | { type: 'metadata'; metadata: Metadata }
  | { type: 'error'; message: string };

export type ScatterInstance = {
  /** Transfer an Arrow IPC or Parquet ArrayBuffer for loading. Ownership is transferred. */
  loadData: (buffer: ArrayBuffer) => void;
  /**
   * Set tour basis matrices. Each basis is a p×2 column-major Float32Array:
   * [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
   * where x_i is the x-projection weight for dimension i.
   *
   * @param bases - array of basis matrices, one per tour keyframe
   * @param dims  - number of dimensions (p)
   */
  setBases: (bases: Float32Array[], dims: number) => void;
  /** Set tour position along the arc-length parameterized path [0, 1]. */
  setTourPosition: (position: number) => void;
  /** Update point rendering style. */
  setStyle: (options: {
    pointSize?: number;
    opacity?: number;
    color?: [number, number, number];
  }) => void;
  /** Set 2D camera (pan in world space, zoom factor). */
  setCamera: (options: { pan?: [number, number]; zoom?: number }) => void;
  /** Resize a canvas to the given pixel dimensions (use for DPI-aware sizing). */
  resize: (viewIndex: number, width: number, height: number) => void;
  /** Request a render of all views. */
  render: () => void;
  /** Subscribe to status events from both workers. Returns an unsubscribe function. */
  subscribe: (handler: (status: ScatterStatus) => void) => () => void;
  /** Terminate both workers and release resources. */
  destroy: () => void;
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
 * scatter.setBases(bases, dims);
 * scatter.setTourPosition(0.5);
 * ```
 */
export const createScatter = (options: ScatterOptions): ScatterInstance => {
  const { canvases } = options;

  if (canvases.length === 0) {
    throw new Error('createScatter requires at least one canvas');
  }

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
    emit(event.data);
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

  // Transfer OffscreenCanvas control to GPU worker
  const offscreens = canvases.map((c) => c.transferControlToOffscreen());

  sendToData(dataWorker, { type: 'init', gpuPort: channel.port1 }, [channel.port1]);
  sendToGpu(gpuWorker, { type: 'init', canvases: offscreens, dataPort: channel.port2 }, [
    ...offscreens,
    channel.port2,
  ]);

  // Track current style/camera for merging partial updates
  let currentStyle = {
    pointSize: 0.012,
    opacity: 0.7,
    color: [0.25, 0.5, 0.9] as [number, number, number],
  };
  let currentCamera = { pan: [0, 0] as [number, number], zoom: 1 };

  const loadData = (buffer: ArrayBuffer): void => {
    sendToData(dataWorker, { type: 'load', buffer }, [buffer]);
  };

  const setBases = (bases: Float32Array[], dims: number): void => {
    // Transfer ownership of basis buffers for zero-copy
    const transfers = bases.map((b) => b.buffer);
    sendToGpu(gpuWorker, { type: 'setBases', bases, dims }, transfers);
  };

  const setTourPosition = (position: number): void => {
    sendToGpu(gpuWorker, { type: 'setTourPosition', position });
  };

  const setStyle = (opts: {
    pointSize?: number;
    opacity?: number;
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

  const setCamera = (opts: { pan?: [number, number]; zoom?: number }): void => {
    currentCamera = { ...currentCamera, ...opts };
    sendToGpu(gpuWorker, {
      type: 'setCamera',
      pan: currentCamera.pan,
      zoom: currentCamera.zoom,
    });
  };

  const resize = (viewIndex: number, width: number, height: number): void => {
    sendToGpu(gpuWorker, { type: 'resize', viewIndex, width, height });
  };

  const render = (): void => {
    sendToGpu(gpuWorker, { type: 'render' });
  };

  const subscribe = (handler: (status: ScatterStatus) => void): (() => void) => {
    subscribers.add(handler);
    return () => subscribers.delete(handler);
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
    subscribe,
    destroy,
  };
};
