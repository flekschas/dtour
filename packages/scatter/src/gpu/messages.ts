// Main thread → GPU Worker
export type MainToGpu =
  | {
      type: 'init';
      canvas: OffscreenCanvas;
      dataPort: MessagePort;
      zoom: number;
      dpr: number;
    }
  | { type: 'addPreviewCanvas'; id: number; canvas: OffscreenCanvas }
  | { type: 'removePreviewCanvas'; id: number }
  | {
      // Array of p×2 column-major basis matrices, one per view.
      // Column-major: [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
      // dims (p) is inferred as bases[0].length / 2.
      type: 'setBases';
      bases: Float32Array[];
    }
  | {
      // Normalized position along the tour arc [0, 1].
      // GPU worker interpolates the basis and re-renders the main view.
      type: 'setTourPosition';
      position: number;
    }
  | {
      type: 'setStyle';
      pointSize: number | 'auto';
      opacity: number | 'auto';
      color: [number, number, number];
    }
  | {
      type: 'setCamera';
      pan: [number, number];
      zoom: number;
      insetOffsetY: number;
      insetZoom: number;
    }
  | { type: 'resize'; viewIndex: number; width: number; height: number; dpr?: number }
  | { type: 'render' }
  | {
      // Single basis for manual/zen modes — renders main view only, no arc-length.
      type: 'setDirectBasis';
      basis: Float32Array;
    }
  | { type: 'clearColors' }
  | { type: 'setBackgroundColor'; color: [number, number, number] }
  | { type: 'setSelectionMask'; mask: Uint32Array }
  | { type: 'clearSelectionMask' }
  | {
      // NDC polygon for lasso selection (flat [x0,y0, x1,y1, ...]).
      // GPU worker runs point-in-polygon against projected positions.
      type: 'lassoSelect';
      polygon: Float32Array;
    }
  | {
      // Trigger GPU-accelerated PCA computation on the loaded data.
      // Result arrives as a GpuToMain 'pcaResult' message.
      type: 'computePCA';
    }
  | {
      // Start worker-driven playback via requestAnimationFrame.
      // Worker advances tour position, renders, and posts playbackTick
      // events back at ~30fps for UI sync.
      type: 'startPlayback';
      speed: number;
      direction: 1 | -1;
    }
  | { type: 'stopPlayback' }
  | { type: 'setMaxPoints'; maxPoints: number }
  | {
      // Enable 3D camera rotation mode. Computes residual PC for the 3rd axis.
      type: 'enable3d';
    }
  | {
      // Disable 3D camera rotation mode. Reverts to standard 2D projection.
      type: 'disable3d';
    }
  | {
      // Update the 3×3 camera rotation matrix (column-major, 9 floats).
      type: 'set3dRotation';
      matrix: Float32Array;
    }
  | { type: 'benchmark'; numFrames: number }
  | { type: 'getMetrics' };

// GPU Worker → Main thread
export type GpuToMain =
  | { type: 'ready' }
  | { type: 'rendered'; viewIndex: number }
  | { type: 'error'; message: string }
  | {
      type: 'pcaResult';
      eigenvectors: Float32Array[];
      eigenvalues: Float32Array;
      numDims: number;
    }
  | {
      // Periodic position update during worker-driven playback (~30fps).
      type: 'playbackTick';
      position: number;
      /** Milliseconds since the previous frame (rAF delta). */
      frameMs?: number;
    }
  | { type: 'playbackStopped'; frameTimes: Float64Array }
  | { type: 'benchmarkResult'; frameTimes: Float64Array; numPoints: number; numDims: number }
  | {
      type: 'metricsResult';
      gpuMemoryBytes: number;
      numPoints: number;
      numDims: number;
      /** JS heap measured from inside the GPU worker (its own V8 context). */
      workerJsHeapBytes: number | null;
    }
  | { type: 'residualPC'; residualPC: Float32Array };
