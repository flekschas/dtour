/// <reference lib="webworker" />

import { COLORMAP_2D_INDEX, packColormap2DLut } from './colormaps2d.ts';
import type { Colormap2DName } from './colormaps2d.ts';
import type { DataToGpu, DataToMain, MainToData } from './messages.ts';
import { GLASBEY_DARK, GLASBEY_LIGHT, MAGMA_25, OKABE_ITO, VIRIDIS_25 } from './palettes.ts';
import { parseBuffer } from './parse.ts';

/** Pack an RGB palette into Uint32Array of 0xAABBGGRR. */
const packPalette = (palette: [number, number, number][]): Uint32Array => {
  const packed = new Uint32Array(palette.length);
  for (let i = 0; i < palette.length; i++) {
    const [r, g, b] = palette[i]!;
    packed[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return packed;
};

let gpuPort: MessagePort | null = null;

// Monotonically increasing dataset version — incremented on each load.
// Included in every DataToGpu message so the GPU worker can discard stale updates.
let dataVersion = 0;

// Lightweight metadata retained for resolving column names → GPU parameters.
// Raw per-row data is NOT kept — it lives only on the GPU.
let numericColumnNames: string[] = [];
let numericMins = new Map<string, number>();
let numericRanges = new Map<string, number>();
let categoricalLabels = new Map<string, string[]>();
let rowCount = 0;

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

      // Bump version so in-flight color/selection messages from the previous
      // dataset will be discarded by the GPU worker.
      dataVersion++;

      // Retain only lightweight metadata — raw values are transferred to GPU
      rowCount = parsed.rowCount;
      numericColumnNames = parsed.columns.map((c) => c.name);
      numericMins = new Map();
      numericRanges = new Map();
      for (const col of parsed.columns) {
        numericMins.set(col.name, col.min);
        numericRanges.set(col.name, col.range);
      }
      categoricalLabels = new Map();
      for (const cat of parsed.categoricalColumns) {
        categoricalLabels.set(cat.name, cat.labels);
      }

      // Build categorical labels record for main thread metadata
      const catLabelsRecord: Record<string, string[]> = {};
      for (const cat of parsed.categoricalColumns) {
        catLabelsRecord[cat.name] = cat.labels;
      }

      // Send column metadata back to main thread (small JSON, structured clone)
      const metadata: DataToMain = {
        type: 'metadata',
        metadata: {
          columnNames: parsed.columns.map((c) => c.name),
          categoricalColumnNames: parsed.categoricalColumns.map((c) => c.name),
          categoricalLabels: catLabelsRecord,
          rowCount: parsed.rowCount,
          dimCount: parsed.columns.length,
          mins: parsed.columns.map((c) => c.min),
          maxes: parsed.columns.map((c) => c.max),
          ranges: parsed.columns.map((c) => c.range),
          embeddedConfig: parsed.embeddedConfig,
        },
      };
      self.postMessage(metadata);

      // Transfer numeric + categorical buffers to GPU worker (zero-copy).
      const numericBuffers = parsed.columns.map((c) => c.values);
      const catColumns = parsed.categoricalColumns.map((c) => ({
        name: c.name,
        indices: c.indices,
      }));
      const transferables = [
        ...new Set(numericBuffers.map((b) => b.buffer)),
        ...catColumns.map((c) => c.indices.buffer),
      ];

      const dataMsg: DataToGpu = {
        type: 'data',
        dataVersion,
        dims: parsed.columns.length,
        rows: parsed.rowCount,
        buffers: numericBuffers,
        mins: parsed.columns.map((c) => c.min),
        ranges: parsed.columns.map((c) => c.range),
        categoricalColumns: catColumns,
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

    const { column, palette, theme, colorMap } = msg;
    const isLight = theme === 'light';

    // Check categorical columns first
    const labels = categoricalLabels.get(column);
    if (labels) {
      let pal: [number, number, number][];
      if (colorMap) {
        pal = labels.map((label) => colorMap[label] ?? [128, 128, 128]);
      } else {
        const glasbey = isLight ? GLASBEY_LIGHT : GLASBEY_DARK;
        pal =
          labels.length <= OKABE_ITO.length
            ? OKABE_ITO
            : ([
                ...OKABE_ITO,
                ...Array(Math.ceil((labels.length - OKABE_ITO.length) / glasbey.length))
                  .fill(undefined)
                  .flatMap(() => glasbey),
              ] as [number, number, number][]);
      }
      const packed = packPalette(pal);
      const gpuMsg: DataToGpu = {
        type: 'setColorCategorical',
        dataVersion,
        catColumnName: column,
        palette: packed,
      };
      gpuPort.postMessage(gpuMsg, [packed.buffer]);
      return;
    }

    // Check numeric columns for continuous color
    const colIdx = numericColumnNames.indexOf(column);
    if (colIdx >= 0) {
      const min = numericMins.get(column) ?? 0;
      const range = numericRanges.get(column) ?? 1;
      const baseCmap = palette === 'magma' ? MAGMA_25 : VIRIDIS_25;
      const cmap = isLight ? ([...baseCmap].reverse() as [number, number, number][]) : baseCmap;
      const packed = packPalette(cmap);
      const gpuMsg: DataToGpu = {
        type: 'setColorContinuous',
        dataVersion,
        columnIndex: colIdx,
        min,
        range,
        colormap: packed,
      };
      gpuPort.postMessage(gpuMsg, [packed.buffer]);
    }
  }

  if (msg.type === 'encodeColor2D') {
    if (!gpuPort) return;

    const { columnX, columnY, colormap } = msg;
    const colIdxX = numericColumnNames.indexOf(columnX);
    const colIdxY = numericColumnNames.indexOf(columnY);
    if (colIdxX < 0 || colIdxY < 0) return;

    const mapIndex = COLORMAP_2D_INDEX[colormap as Colormap2DName] ?? 0;
    const lut = packColormap2DLut(mapIndex);
    const transfers: Transferable[] = [];
    if (lut) transfers.push(lut.buffer);

    const gpuMsg: DataToGpu = {
      type: 'setColor2D',
      dataVersion,
      columnIndexX: colIdxX,
      columnIndexY: colIdxY,
      minX: numericMins.get(columnX) ?? 0,
      rangeX: numericRanges.get(columnX) ?? 1,
      minY: numericMins.get(columnY) ?? 0,
      rangeY: numericRanges.get(columnY) ?? 1,
      lut,
      mapIndex,
    };
    gpuPort.postMessage(gpuMsg, transfers);
  }

  if (msg.type === 'selectByColumn') {
    if (!gpuPort || rowCount === 0) return;

    const { column, labelIndices, valueRanges } = msg;

    // Categorical selection
    const labels = categoricalLabels.get(column);
    if (labels && labelIndices && labelIndices.length > 0) {
      const selectedLabels = new Uint32Array(labels.length);
      for (const idx of labelIndices) {
        if (idx < selectedLabels.length) selectedLabels[idx] = 1;
      }
      const gpuMsg: DataToGpu = {
        type: 'selectCategorical',
        dataVersion,
        catColumnName: column,
        selectedLabels,
      };
      gpuPort.postMessage(gpuMsg, [selectedLabels.buffer]);
      return;
    }

    // Continuous selection
    const colIdx = numericColumnNames.indexOf(column);
    if (colIdx >= 0 && valueRanges && valueRanges.length >= 2) {
      const gpuMsg: DataToGpu = {
        type: 'selectContinuous',
        dataVersion,
        columnIndex: colIdx,
        ranges: valueRanges,
      };
      gpuPort.postMessage(gpuMsg, [valueRanges.buffer]);
      return;
    }

    // Unknown column or missing params — no-op rather than dimming everything
  }
};
