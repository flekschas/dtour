import { parquetMetadata, parquetRead } from 'hyparquet';
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
 *
 * Uses parquetRead with onChunk for columnar streaming — avoids allocating
 * a row-object per row, which is critical for large datasets (≥1M rows).
 */
export const loadParquet = async (buffer: ArrayBuffer): Promise<ParquetResult> => {
  const asyncBuffer = toAsyncBuffer(buffer);

  // Extract embedded dtour config from Parquet footer metadata (synchronous, fast).
  // Check direct kv entry first (pyarrow), then fall back to ARROW:schema (arro3).
  const fileMeta = parquetMetadata(buffer);
  const dtourEntry = fileMeta.key_value_metadata?.find((kv) => kv.key === 'dtour');
  const embeddedConfig = dtourEntry?.value ?? extractFromArrowSchema(fileMeta.key_value_metadata);

  // Classify columns from Parquet schema — avoids scanning rows for type inference.
  // Leaf nodes have a `type` field; group nodes (lists, maps, structs) have `num_children`
  // and no `type`. FIXED_LEN_BYTE_ARRAY is skipped: it's typically decimals or UUIDs, not
  // strings. Parquet internal names from nested groups (element, list, key, value, …) are
  // also skipped to avoid silently importing garbage categorical columns from nested schemas.
  const PARQUET_INTERNAL_NAMES = new Set([
    'list',
    'element',
    'key',
    'value',
    'bag',
    'array',
    'item',
    'entries',
    'entry',
  ]);
  const numericColNames: string[] = [];
  const stringColNames: string[] = [];
  for (const field of fileMeta.schema) {
    if (!('type' in field) || field.type == null) continue; // group node
    if (/^__index_level_\d+__$/.test(field.name)) continue; // pandas index
    if (PARQUET_INTERNAL_NAMES.has(field.name)) continue; // nested internals
    const t = field.type as string;
    if (
      t === 'FLOAT' ||
      t === 'DOUBLE' ||
      t === 'INT32' ||
      t === 'INT64' ||
      t === 'INT96' ||
      t === 'BOOLEAN'
    ) {
      numericColNames.push(field.name);
    } else if (t === 'BYTE_ARRAY') {
      // BYTE_ARRAY is the standard Parquet string type.
      // FIXED_LEN_BYTE_ARRAY is decimal/UUID — unsupported, silently skipped.
      stringColNames.push(field.name);
    }
  }

  const rowCount = Number(fileMeta.num_rows);

  // Pre-allocate output buffers before streaming begins.
  const numeric: Record<string, Float32Array> = {};
  for (const col of numericColNames) numeric[col] = new Float32Array(rowCount);

  const labelSets = new Map<string, Map<string, number>>();
  const catIndices = new Map<string, Uint32Array>();
  for (const col of stringColNames) {
    labelSets.set(col, new Map());
    catIndices.set(col, new Uint32Array(rowCount));
  }

  const numericSet = new Set(numericColNames);
  const stringSet = new Set(stringColNames);

  // Stream row-group chunks column by column directly into pre-allocated arrays.
  // onChunk delivers ArrayLike<unknown> per (column, row-group) — no row objects.
  await parquetRead({
    file: asyncBuffer,
    compressors,
    columns: [...numericColNames, ...stringColNames],
    onChunk({
      columnName,
      columnData,
      rowStart,
    }: {
      columnName: string;
      columnData: ArrayLike<unknown>;
      rowStart: number;
      rowEnd: number;
    }) {
      const len = columnData.length;

      if (numericSet.has(columnName)) {
        const out = numeric[columnName]!;
        if (columnData instanceof Float32Array) {
          // Zero-copy fast path for native float32 chunks.
          out.set(columnData, rowStart);
        } else {
          for (let i = 0; i < len; i++) {
            const v = columnData[i];
            out[rowStart + i] =
              typeof v === 'bigint'
                ? Number.isFinite(Number(v))
                  ? Number(v)
                  : 0
                : typeof v === 'number' && Number.isFinite(v)
                  ? v
                  : 0;
          }
        }
        return;
      }

      if (stringSet.has(columnName)) {
        const catIdx = catIndices.get(columnName)!;
        const labels = labelSets.get(columnName)!;
        for (let i = 0; i < len; i++) {
          const val = String(columnData[i] ?? '');
          let idx = labels.get(val);
          if (idx === undefined) {
            idx = labels.size;
            labels.set(val, idx);
          }
          catIdx[rowStart + i] = idx;
        }
      }
    },
  });

  // Sort labels alphabetically for deterministic ordering.
  const categorical: CategoricalColumn[] = stringColNames.map((key) => {
    const unsortedLabels = [...labelSets.get(key)!.keys()];
    const sortedLabels = [...unsortedLabels].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }),
    );
    const newIdx = new Map<string, number>();
    for (let j = 0; j < sortedLabels.length; j++) newIdx.set(sortedLabels[j]!, j);
    const oldToNew = unsortedLabels.map((l) => newIdx.get(l)!);
    const indices = catIndices.get(key)!;
    for (let i = 0; i < indices.length; i++) indices[i] = oldToNew[indices[i]!]!;
    return { name: key, indices, labels: sortedLabels };
  });

  return { numeric, categorical, embeddedConfig };
};
