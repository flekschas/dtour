// Main thread → GPU Worker
export type MainToGpu =
  | {
      type: 'init';
      canvases: OffscreenCanvas[];
      dataPort: MessagePort;
      zoom: number;
      dpr: number;
    }
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
  | { type: 'setStyle'; pointSize: number | 'auto'; opacity: number | 'auto'; color: [number, number, number] }
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
    };

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
    };
