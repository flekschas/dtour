/// <reference lib="webworker" />

import { encodeCategoricalColors, encodeContinuousColors } from './color.ts';
import type { DataToGpu, DataToMain, MainToData } from './messages.ts';
import { GLASBEY_DARK, OKABE_ITO, VIRIDIS_20 } from './palettes.ts';
import { parseBuffer } from './parse.ts';
import type { CategoricalColumn } from './types.ts';

let gpuPort: MessagePort | null = null;

// Retained column data for color encoding (copies kept after transfer)
let numericData = new Map<string, Float32Array>();
let numericMins = new Map<string, number>();
let numericRanges = new Map<string, number>();
let categoricalData = new Map<string, CategoricalColumn>();

self.onmessage = async (event: MessageEvent<MainToData>) => {
  const msg = event.data;

  if (msg.type === 'init') {
    gpuPort = msg.gpuPort;
    return;
  }

  if (msg.type === 'load') {
    if (!gpuPort) {
      const err: DataToMain = { type: 'error', message: 'GPU port not initialized before load' };
      self.postMessage(err);
      return;
    }

    try {
      const parsed = await parseBuffer(msg.buffer);

      // Retain copies of column data for later color encoding
      numericData = new Map();
      numericMins = new Map();
      numericRanges = new Map();
      for (const col of parsed.columns) {
        numericData.set(col.name, col.values.slice());
        numericMins.set(col.name, col.min);
        numericRanges.set(col.name, col.range);
      }
      categoricalData = new Map();
      for (const cat of parsed.categoricalColumns) {
        categoricalData.set(cat.name, cat);
      }

      // Build categorical labels map
      const categoricalLabels: Record<string, string[]> = {};
      for (const cat of parsed.categoricalColumns) {
        categoricalLabels[cat.name] = cat.labels;
      }

      // Send column metadata back to main thread (small JSON, structured clone)
      const metadata: DataToMain = {
        type: 'metadata',
        metadata: {
          columnNames: parsed.columns.map((c) => c.name),
          categoricalColumnNames: parsed.categoricalColumns.map((c) => c.name),
          categoricalLabels,
          rowCount: parsed.rowCount,
          dimCount: parsed.columns.length,
          mins: parsed.columns.map((c) => c.min),
          maxes: parsed.columns.map((c) => c.max),
          ranges: parsed.columns.map((c) => c.range),
        },
      };
      self.postMessage(metadata);

      // Send raw column buffers directly to GPU worker (zero-copy transfer).
      // Normalization happens on the GPU via per-dim mins/ranges.
      const buffers = parsed.columns.map((c) => c.values);
      const transferables = buffers.map((b) => b.buffer);

      const dataMsg: DataToGpu = {
        type: 'data',
        dims: parsed.columns.length,
        rows: parsed.rowCount,
        buffers,
        mins: parsed.columns.map((c) => c.min),
        ranges: parsed.columns.map((c) => c.range),
      };
      gpuPort.postMessage(dataMsg, transferables);
    } catch (err) {
      const errMsg: DataToMain = {
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      };
      self.postMessage(errMsg);
    }
  }

  if (msg.type === 'encodeColor') {
    if (!gpuPort) return;

    const { column, palette } = msg;

    // Check categorical columns first
    const cat = categoricalData.get(column);
    if (cat) {
      const pal = palette === 'glasbey_dark' ? GLASBEY_DARK : OKABE_ITO;
      const colors = encodeCategoricalColors(cat.indices, pal);
      gpuPort.postMessage({ type: 'colors', colors } as DataToGpu, [colors.buffer]);
      return;
    }

    // Check numeric columns for continuous color
    const numCol = numericData.get(column);
    if (numCol) {
      const min = numericMins.get(column) ?? 0;
      const range = numericRanges.get(column) ?? 1;
      const colors = encodeContinuousColors(numCol, min, range, VIRIDIS_20);
      gpuPort.postMessage({ type: 'colors', colors } as DataToGpu, [colors.buffer]);
    }
  }
};
