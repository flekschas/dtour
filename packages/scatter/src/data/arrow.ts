import { tableFromIPC } from '@uwdata/flechette';
import type { CategoricalColumn } from './types.ts';

export type ArrowResult = {
  numeric: Record<string, Float32Array>;
  categorical: CategoricalColumn[];
};

const NUMERIC_TYPED_ARRAYS = new Set([
  'Float32Array',
  'Float64Array',
  'Int8Array',
  'Int16Array',
  'Int32Array',
  'Uint8Array',
  'Uint16Array',
  'Uint32Array',
]);

/**
 * Load Arrow IPC (file or stream format) and extract all numeric columns
 * as Float32Arrays plus categorical columns. Runs inside the Data Worker.
 */
export const loadArrow = (buffer: ArrayBuffer): ArrowResult => {
  const table = tableFromIPC(new Uint8Array(buffer));
  // toColumns() returns { [name]: TypedArray | unknown[] }
  const columns = table.toColumns() as Record<string, unknown>;
  const numeric: Record<string, Float32Array> = {};
  const categorical: CategoricalColumn[] = [];

  for (const [name, arr] of Object.entries(columns)) {
    // Numeric typed arrays → Float32Array
    if (ArrayBuffer.isView(arr)) {
      const tag = Object.prototype.toString.call(arr).slice(8, -1);
      if (NUMERIC_TYPED_ARRAYS.has(tag)) {
        numeric[name] =
          arr instanceof Float32Array ? arr : new Float32Array(arr as unknown as ArrayLike<number>);
      }
      continue;
    }

    // Plain arrays (string/dictionary columns) → categorical
    if (Array.isArray(arr) && arr.length > 0 && typeof arr[0] === 'string') {
      const labelSet = new Map<string, number>();
      const indices = new Uint32Array(arr.length);
      for (let i = 0; i < arr.length; i++) {
        const val = String(arr[i] ?? '');
        let idx = labelSet.get(val);
        if (idx === undefined) {
          idx = labelSet.size;
          labelSet.set(val, idx);
        }
        indices[i] = idx;
      }
      categorical.push({ name, indices, labels: [...labelSet.keys()] });
    }
  }

  return { numeric, categorical };
};
