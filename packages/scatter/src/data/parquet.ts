import { parquetReadObjects } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';
import type { CategoricalColumn } from './types.ts';

export type ParquetResult = {
  numeric: Record<string, Float32Array>;
  categorical: CategoricalColumn[];
};

// hyparquet expects an AsyncBuffer wrapping arbitrary byte sources
type AsyncBuffer = {
  byteLength: number;
  slice(start: number, end?: number): Promise<ArrayBuffer>;
};

const toAsyncBuffer = (buffer: ArrayBuffer): AsyncBuffer => ({
  byteLength: buffer.byteLength,
  slice: (start, end) => Promise.resolve(buffer.slice(start, end)),
});

/**
 * Load a Parquet file from an ArrayBuffer and extract all numeric columns
 * as Float32Arrays plus categorical columns. Runs inside the Data Worker.
 */
export const loadParquet = async (buffer: ArrayBuffer): Promise<ParquetResult> => {
  const asyncBuffer = toAsyncBuffer(buffer);

  // parquetReadObjects returns Promise<Record<string, unknown>[]> (row-oriented, object format)
  const rows = await parquetReadObjects({ file: asyncBuffer, compressors });

  // Identify numeric vs string columns from first row,
  // skipping pandas index columns (__index_level_N__)
  const numericKeys: string[] = [];
  const stringKeys: string[] = [];
  if (rows.length > 0) {
    for (const [key, value] of Object.entries(rows[0]!)) {
      if (/^__index_level_\d+__$/.test(key)) continue;
      if (typeof value === 'number') {
        numericKeys.push(key);
      } else if (typeof value === 'string') {
        stringKeys.push(key);
      }
    }
  }

  const rowCount = rows.length;

  // Extract numeric columns — non-finite values become 0
  const numeric: Record<string, Float32Array> = {};
  for (const key of numericKeys) {
    numeric[key] = new Float32Array(rowCount);
  }

  // Build categorical label sets
  const labelSets = new Map<string, Map<string, number>>();
  const catIndices = new Map<string, Uint32Array>();
  for (const key of stringKeys) {
    labelSets.set(key, new Map());
    catIndices.set(key, new Uint32Array(rowCount));
  }

  for (let i = 0; i < rowCount; i++) {
    const row = rows[i]!;
    for (const key of numericKeys) {
      const value = row[key];
      numeric[key]![i] = typeof value === 'number' && Number.isFinite(value) ? value : 0;
    }
    for (const key of stringKeys) {
      const val = String(row[key] ?? '');
      const labels = labelSets.get(key)!;
      let idx = labels.get(val);
      if (idx === undefined) {
        idx = labels.size;
        labels.set(val, idx);
      }
      catIndices.get(key)![i] = idx;
    }
  }

  const categorical: CategoricalColumn[] = stringKeys.map((key) => ({
    name: key,
    indices: catIndices.get(key)!,
    labels: [...labelSets.get(key)!.keys()],
  }));

  return { numeric, categorical };
};
