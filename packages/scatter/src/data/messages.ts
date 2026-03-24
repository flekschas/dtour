import type { Metadata } from './types.ts';

// Main thread → Data Worker
export type MainToData =
  | { type: 'init'; gpuPort: MessagePort }
  | { type: 'load'; buffer: ArrayBuffer }
  | { type: 'encodeColor'; column: string; palette?: string; theme?: 'light' | 'dark'; colorMap?: Record<string, [number, number, number]> }
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
  // Color encoding — resolved by data worker, computed on GPU
  | {
      type: 'encodeColorContinuous';
      dataVersion: number;
      columnIndex: number;
      min: number;
      range: number;
      /** Packed RGBA u32 colormap LUT (e.g. 25 stops for viridis). */
      colormap: Uint32Array;
    }
  | {
      type: 'encodeColorCategorical';
      dataVersion: number;
      catColumnName: string;
      /** Packed RGBA u32 per label. */
      palette: Uint32Array;
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
