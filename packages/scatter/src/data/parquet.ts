import { parquetMetadata, parquetReadObjects } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';
import type { CategoricalColumn } from './types.ts';

export type ParquetResult = {
  numeric: Record<string, Float32Array>;
  categorical: CategoricalColumn[];
  /** Raw JSON string from the Parquet "dtour" key_value_metadata entry. */
  embeddedConfig?: string;
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
 * Extract the "dtour" metadata value from the ARROW:schema FlatBuffer entry.
 * arro3's Parquet writer embeds schema metadata inside the ARROW:schema key
 * (base64-encoded IPC Message containing a Schema FlatBuffer) rather than as
 * separate Parquet key_value_metadata entries.
 */
const extractFromArrowSchema = (
  kvMetadata: { key: string; value?: string }[] | undefined,
): string | undefined => {
  const entry = kvMetadata?.find((kv) => kv.key === 'ARROW:schema');
  if (!entry?.value) return undefined;

  try {
    const binary = atob(entry.value);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return findDtourInArrowIpc(bytes);
  } catch {
    return undefined;
  }
};

/** Parse the IPC Message → Schema → custom_metadata to find the "dtour" key. */
const findDtourInArrowIpc = (bytes: Uint8Array): string | undefined => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 16) return undefined;

  // IPC preamble: 0xFFFFFFFF continuation marker + int32 message length
  // The FlatBuffer Message starts at offset 8.
  const marker = view.getInt32(0, true);
  if (marker !== -1) return undefined; // not an IPC message
  const fbStart = 8;

  // Helper: read int32/uint16/string relative to FlatBuffer start
  const i32 = (off: number) => view.getInt32(fbStart + off, true);
  const u16 = (off: number) => view.getUint16(fbStart + off, true);
  const str = (off: number) => {
    const len = i32(off);
    return new TextDecoder().decode(bytes.subarray(fbStart + off + 4, fbStart + off + 4 + len));
  };

  // Message root table
  const msgOff = i32(0);
  const msgVtRel = i32(msgOff);
  const msgVtOff = msgOff - msgVtRel;
  const msgVtSize = u16(msgVtOff);
  const msgFields = (msgVtSize - 4) >> 1;
  if (msgFields < 3) return undefined;

  // Message field 2 = header (union value → Schema table offset)
  const headerFieldOff = u16(msgVtOff + 4 + 2 * 2);
  if (headerFieldOff === 0) return undefined;
  const schemaRel = i32(msgOff + headerFieldOff);
  const schemaOff = msgOff + headerFieldOff + schemaRel;

  // Schema table → custom_metadata (field 2)
  const sVtRel = i32(schemaOff);
  const sVtOff = schemaOff - sVtRel;
  const sVtSize = u16(sVtOff);
  const sFields = (sVtSize - 4) >> 1;
  if (sFields < 3) return undefined;

  const cmFieldOff = u16(sVtOff + 4 + 2 * 2);
  if (cmFieldOff === 0) return undefined;

  // Vector of KeyValue tables
  const vecRel = i32(schemaOff + cmFieldOff);
  const vecOff = schemaOff + cmFieldOff + vecRel;
  const numEntries = i32(vecOff);

  for (let e = 0; e < numEntries; e++) {
    const eRel = i32(vecOff + 4 + e * 4);
    const eOff = vecOff + 4 + e * 4 + eRel;

    // KeyValue vtable
    const kvVtRel = i32(eOff);
    const kvVtOff = eOff - kvVtRel;
    const kvFields = (u16(kvVtOff) - 4) >> 1;
    if (kvFields < 2) continue;

    const kFO = u16(kvVtOff + 4);
    const vFO = u16(kvVtOff + 6);
    if (kFO === 0 || vFO === 0) continue;

    const key = str(eOff + kFO + i32(eOff + kFO));
    if (key !== 'dtour') continue;
    return str(eOff + vFO + i32(eOff + vFO));
  }

  return undefined;
};

/**
 * Load a Parquet file from an ArrayBuffer and extract all numeric columns
 * as Float32Arrays plus categorical columns. Runs inside the Data Worker.
 */
export const loadParquet = async (buffer: ArrayBuffer): Promise<ParquetResult> => {
  const asyncBuffer = toAsyncBuffer(buffer);

  // Extract embedded dtour config from Parquet footer metadata (synchronous, fast).
  // Check direct kv entry first (pyarrow), then fall back to ARROW:schema (arro3).
  const fileMeta = parquetMetadata(buffer);
  const dtourEntry = fileMeta.key_value_metadata?.find((kv) => kv.key === 'dtour');
  const embeddedConfig = dtourEntry?.value ?? extractFromArrowSchema(fileMeta.key_value_metadata);

  // parquetReadObjects returns Promise<Record<string, unknown>[]> (row-oriented, object format)
  const rows = await parquetReadObjects({ file: asyncBuffer, compressors });

  // Identify numeric vs string columns by scanning up to the first 100 rows.
  // First-row-only inference silently drops columns when the first value is null.
  // Skip pandas index columns (__index_level_N__).
  const numericKeys: string[] = [];
  const stringKeys: string[] = [];
  if (rows.length > 0) {
    const allKeys = Object.keys(rows[0]!).filter((k) => !/^__index_level_\d+__$/.test(k));
    const scanLimit = Math.min(rows.length, 100);
    for (const key of allKeys) {
      let inferred: 'number' | 'string' | null = null;
      for (let i = 0; i < scanLimit; i++) {
        const value = rows[i]![key];
        if (value == null) continue;
        if (typeof value === 'number') {
          inferred = 'number';
          break;
        }
        if (typeof value === 'string') {
          inferred = 'string';
          break;
        }
      }
      if (inferred === 'number') numericKeys.push(key);
      else if (inferred === 'string') stringKeys.push(key);
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

  // Sort labels alphabetically for deterministic ordering
  const categorical: CategoricalColumn[] = stringKeys.map((key) => {
    const unsortedLabels = [...labelSets.get(key)!.keys()];
    const sortedLabels = [...unsortedLabels].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }),
    );
    const newIdx = new Map<string, number>();
    for (let j = 0; j < sortedLabels.length; j++) {
      newIdx.set(sortedLabels[j]!, j);
    }
    const oldToNew = unsortedLabels.map((l) => newIdx.get(l)!);
    const indices = catIndices.get(key)!;
    for (let i = 0; i < indices.length; i++) {
      indices[i] = oldToNew[indices[i]!]!;
    }
    return { name: key, indices, labels: sortedLabels };
  });

  return { numeric, categorical, embeddedConfig };
};
