/**
 * Inline Web Worker that parses a CSV ArrayBuffer and returns the result
 * as an Arrow IPC ArrayBuffer, following the same pattern as lorenz.worker.ts.
 *
 * Handles quoted fields (RFC 4180), BOM, CRLF/LF, and auto-detects
 * numeric vs categorical columns.
 */

/// <reference lib="webworker" />
import { tableFromArrays, tableToIPC } from '@uwdata/flechette';

// ---------------------------------------------------------------------------
// CSV parser — handles quoted fields, escaped quotes, mixed line endings
// ---------------------------------------------------------------------------

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  const len = text.length;
  let i = 0;

  // Skip BOM
  if (len > 0 && text.charCodeAt(0) === 0xfeff) i = 1;

  while (i < len) {
    const row: string[] = [];
    let more = true;

    while (more && i < len && text[i] !== '\n' && text[i] !== '\r') {
      if (text[i] === '"') {
        // Quoted field
        i++; // skip opening quote
        let field = '';
        let start = i;
        while (i < len) {
          if (text[i] === '"') {
            field += text.slice(start, i);
            i++;
            if (i < len && text[i] === '"') {
              // Escaped quote
              field += '"';
              i++;
              start = i;
            } else {
              break;
            }
          } else {
            i++;
          }
        }
        row.push(field);
      } else {
        // Unquoted field
        const start = i;
        while (i < len && text[i] !== ',' && text[i] !== '\n' && text[i] !== '\r') {
          i++;
        }
        row.push(text.slice(start, i));
      }

      if (i < len && text[i] === ',') {
        i++; // consume comma
        // Trailing comma → empty final field
        if (i >= len || text[i] === '\n' || text[i] === '\r') {
          row.push('');
          more = false;
        }
      } else {
        more = false;
      }
    }

    // Skip line ending
    if (i < len && text[i] === '\r') i++;
    if (i < len && text[i] === '\n') i++;

    // Skip blank lines
    if (row.length > 0 && !(row.length === 1 && row[0] === '')) {
      rows.push(row);
    }
  }

  return rows;
}

// ---------------------------------------------------------------------------
// Worker entry point
// ---------------------------------------------------------------------------

self.onmessage = (e: MessageEvent<ArrayBuffer>) => {
  try {
    const text = new TextDecoder().decode(e.data);
    const rows = parseCsv(text);

    if (rows.length < 2) {
      self.postMessage({ error: 'CSV must have a header row and at least one data row' });
      return;
    }

    const headers = rows[0]!;
    const dataRows = rows.slice(1);
    const numCols = headers.length;

    // Build column arrays and detect types
    const columns: Record<string, Float32Array | string[]> = {};
    const usedNames = new Set<string>();

    for (let col = 0; col < numCols; col++) {
      let name = headers[col]!.trim() || `column_${col}`;
      if (usedNames.has(name)) {
        let suffix = 2;
        while (usedNames.has(`${name}_${suffix}`)) suffix++;
        name = `${name}_${suffix}`;
      }
      usedNames.add(name);

      // Collect values, padding short rows with empty strings
      const values: string[] = new Array(dataRows.length);
      for (let r = 0; r < dataRows.length; r++) {
        values[r] = dataRows[r]![col] ?? '';
      }

      // Detect numeric: all non-blank trimmed values must parse as finite numbers
      let allNumeric = true;
      let hasNumeric = false;
      for (let r = 0; r < values.length; r++) {
        const trimmed = values[r]!.trim();
        if (trimmed === '') continue;
        if (!Number.isFinite(Number(trimmed))) {
          allNumeric = false;
          break;
        }
        hasNumeric = true;
      }

      if (allNumeric && hasNumeric) {
        const arr = new Float32Array(values.length);
        for (let r = 0; r < values.length; r++) {
          const trimmed = values[r]!.trim();
          arr[r] = trimmed === '' ? Number.NaN : Number(trimmed);
        }
        columns[name] = arr;
      } else {
        columns[name] = values;
      }
    }

    const table = tableFromArrays(columns);
    const bytes = tableToIPC(table);
    const buffer = bytes.buffer as ArrayBuffer;
    self.postMessage(buffer, [buffer]);
  } catch (err) {
    self.postMessage({ error: err instanceof Error ? err.message : String(err) });
  }
};
