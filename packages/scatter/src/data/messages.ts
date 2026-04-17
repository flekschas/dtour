import type { Metadata } from './types.ts';

// Main thread → Data Worker
export type MainToData =
  | { type: 'init'; gpuPort: MessagePort }
  | { type: 'load'; buffer: ArrayBuffer }
  | {
      type: 'encodeColor';
      column: string;
      palette?: string;
      theme?: 'light' | 'dark';
      colorMap?: Record<string, [number, number, number]>;
    }
  | {
      type: 'encodeColor2D';
      columnX: string;
      columnY: string;
      colormap: string; // e.g. 'schumann', 'bremm', 'oklab_polar', ...
    }
  | { type: 'selectByColumn'; column: string; labelIndices?: number[]; valueRanges?: Float32Array };

// Data Worker → Main thread
export type DataToMain =
  | { type: 'metadata'; metadata: Metadata }
  | { type: 'error'; message: string };

// Data Worker → GPU Worker (via MessageChannel port)
// Every message carries a dataVersion so the GPU worker can discard stale
// color/selection updates that arrive after a newer dataset has been loaded.
export type DataToGpu =
  | {
      type: 'data';
      dataVersion: number;
      dims: number;
      rows: number;
      buffers: Float32Array[];
      mins: number[];
      ranges: number[];
      /** Categorical column indices, transferred alongside numeric buffers. */
      categoricalColumns: { name: string; indices: Uint32Array }[];
    }
  // Color mapping — resolved by data worker, applied as shader LUT
  | {
      type: 'setColorContinuous';
      dataVersion: number;
      columnIndex: number;
      min: number;
      range: number;
      /** Packed RGBA u32 colormap LUT (e.g. 25 stops for viridis). */
      colormap: Uint32Array;
    }
  | {
      type: 'setColorCategorical';
      dataVersion: number;
      catColumnName: string;
      /** Packed RGBA u32 per label. */
      palette: Uint32Array;
    }
  // 2D color mapping — two numeric columns → procedural 2D colormap
  | {
      type: 'setColor2D';
      dataVersion: number;
      columnIndexX: number;
      columnIndexY: number;
      minX: number;
      rangeX: number;
      minY: number;
      rangeY: number;
      /** SVD rank-1 LUT curves packed as f32: R_X[16], R_Y[16], G_X[16], G_Y[16], B_X[16], B_Y[16], B_X2[16], B_Y2[16]. null for procedural colormaps (oklab_polar). */
      lut: Float32Array | null;
      /** Colormap index: 0=schumann, 1=bremm, 2=steiger, 3=ziegler, 4=teulingfig2, 5=cubediagonal, 6=oklab_polar */
      mapIndex: number;
    }
  // Selection — resolved by data worker, computed on GPU
  | {
      type: 'selectContinuous';
      dataVersion: number;
      columnIndex: number;
      /** Flat [lo0, hi0, lo1, hi1, ...] pairs. */
      ranges: Float32Array;
    }
  | {
      type: 'selectCategorical';
      dataVersion: number;
      catColumnName: string;
      /** Per-label mask: selectedLabels[labelIdx] = 1 or 0. */
      selectedLabels: Uint32Array;
    };
