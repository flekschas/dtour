import DataWorkerFactory from '../data/worker.ts?worker&inline';
import WebGLWorkerFactory from './worker.ts?worker&inline';

import type { DataToMain, MainToData } from '../data/messages.ts';
import type { Metadata } from '../data/types.ts';
import type { ScatterInstance, ScatterOptions, ScatterStatus } from '../gpu/client.ts';
import type { GpuToMain, MainToGpu } from '../gpu/messages.ts';

const sendToGpu = (worker: Worker, msg: MainToGpu, transfers?: Transferable[]): void => {
  worker.postMessage(msg, transfers ?? []);
};

const sendToData = (worker: Worker, msg: MainToData, transfers?: Transferable[]): void => {
  worker.postMessage(msg, transfers ?? []);
};

/**
 * Create a scatter renderer instance using the WebGL2 backend.
 *
 * Same API as `createScatter` (WebGPU) — instantiates a WebGL Worker
 * and Data Worker, connects them via a MessageChannel, and transfers
 * OffscreenCanvas control to the WebGL worker.
 */
export const createScatterWebGL = (options: ScatterOptions): ScatterInstance => {
  const {
    canvases,
    zoom: initialZoom = 1,
    dpr = typeof self !== 'undefined' && 'devicePixelRatio' in self ? self.devicePixelRatio : 1,
  } = options;

  if (canvases.length === 0) {
    throw new Error('createScatterWebGL requires at least one canvas');
  }

  const gpuWorker = new WebGLWorkerFactory();
  const dataWorker = new DataWorkerFactory();

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

  const offscreens = canvases.map((c) => c.transferControlToOffscreen());

  sendToData(dataWorker, { type: 'init', gpuPort: channel.port1 }, [channel.port1]);
  sendToGpu(
    gpuWorker,
    { type: 'init', canvases: offscreens, dataPort: channel.port2, zoom: initialZoom, dpr },
    [...offscreens, channel.port2],
  );

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

  const benchmark = (
    numPoints?: number,
  ): Promise<{ frameTimes: Float64Array; numPoints: number; avgMs: number; fps: number }> => {
    return new Promise((resolve) => {
      const unsub = subscribe((s: ScatterStatus) => {
        if (s.type !== 'benchmarkResult') return;
        unsub();
        const ft = s.frameTimes;
        const avg = ft.reduce((a: number, b: number) => a + b, 0) / ft.length;
        const sorted = Float64Array.from(ft).sort();
        const p50 = sorted[Math.floor(ft.length * 0.5)]!;
        const p95 = sorted[Math.floor(ft.length * 0.95)]!;
        const min = sorted[0]!;
        const max = sorted[ft.length - 1]!;
        console.log(
          `Benchmark: ${s.numPoints.toLocaleString()} points\n` +
            `  ${ft.length} frames, avg ${avg.toFixed(1)}ms (${(1000 / avg).toFixed(0)} fps)\n` +
            `  min ${min.toFixed(1)}ms, p50 ${p50.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms, max ${max.toFixed(1)}ms`,
        );
        resolve({ frameTimes: ft, numPoints: s.numPoints, avgMs: avg, fps: 1000 / avg });
      });
      if (numPoints) {
        sendToGpu(gpuWorker, { type: 'benchmark', numPoints, numDims: 4, numFrames: 120 });
      } else {
        sendToGpu(gpuWorker, { type: 'benchmarkExisting', numFrames: 120 });
      }
    });
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
    setDirectBasis,
    encodeColor,
    setBackgroundColor,
    clearColor,
    selectByColumn,
    setSelectionMask,
    lassoSelect,
    clearSelection,
    computePCA,
    startPlayback,
    stopPlayback,
    setMaxPoints,
    benchmark,
    subscribe,
    destroy,
  };
};
