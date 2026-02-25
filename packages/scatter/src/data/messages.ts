import type { Metadata } from './types.ts';

// Main thread → Data Worker
export type MainToData =
  | { type: 'init'; gpuPort: MessagePort }
  | { type: 'load'; buffer: ArrayBuffer }
  | { type: 'encodeColor'; column: string; palette?: string };

// Data Worker → Main thread
export type DataToMain =
  | { type: 'metadata'; metadata: Metadata }
  | { type: 'error'; message: string };

// Data Worker → GPU Worker (via MessageChannel port)
// Buffers are transferable Float32Arrays, one per dimension (raw, unnormalized).
// mins/ranges are per-dimension stats so the GPU can normalize in the shader.
export type DataToGpu =
  | {
      type: 'data';
      dims: number;
      rows: number;
      buffers: Float32Array[];
      mins: number[];
      ranges: number[];
    }
  | { type: 'colors'; colors: Uint32Array };
