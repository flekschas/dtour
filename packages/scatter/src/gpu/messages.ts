// Main thread → GPU Worker
export type MainToGpu =
  | {
      type: 'init';
      canvases: OffscreenCanvas[];
      dataPort: MessagePort;
    }
  | {
      // Array of p×2 column-major basis matrices, one per view.
      // Column-major: [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
      type: 'setBases';
      bases: Float32Array[];
      dims: number;
    }
  | {
      // Normalized position along the tour arc [0, 1].
      // GPU worker interpolates the basis and re-renders the main view.
      type: 'setTourPosition';
      position: number;
    }
  | { type: 'setStyle'; pointSize: number; opacity: number; color: [number, number, number] }
  | { type: 'setCamera'; pan: [number, number]; zoom: number }
  | { type: 'resize'; viewIndex: number; width: number; height: number }
  | { type: 'render' };

// GPU Worker → Main thread
export type GpuToMain =
  | {
      type: 'ready';
      limits: { maxBufferSize: number; maxTextureDimension2D: number };
    }
  | { type: 'rendered'; viewIndex: number }
  | { type: 'error'; message: string };
