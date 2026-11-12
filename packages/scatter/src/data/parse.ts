import { analyzeColumn } from './analyze.ts';
import { loadArrow } from './arrow.ts';
import { detectFormat } from './detect.ts';
import { MAX_COLOR_CATEGORIES } from './palettes.ts';
import { loadParquet } from './parquet.ts';
import type { AnalyzedData, CategoricalColumn } from './types.ts';

// `localeCompare` with options creates a collator per call, which is ~20x slower on large columns.
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

/** Sort a column's labels in place and remap its indices to the new order. */
const sortLabels = (column: CategoricalColumn): void => {
  const sorted = [...column.labels].sort(collator.compare);
  const newIdx = new Map(sorted.map((label, i) => [label, i]));
  const oldToNew = column.labels.map((label) => newIdx.get(label)!);
  const { indices } = column;
  for (let i = 0; i < indices.length; i++) indices[i] = oldToNew[indices[i]!]!;
  column.labels = sorted;
};

export type ParseResult = AnalyzedData & {
  /** Raw JSON from Parquet "dtour" key_value_metadata. Undefined for Arrow. */
  embeddedConfig?: string;
};

/**
 * Detect format, load, and extract numeric columns with per-column stats.
 * Runs inside the Data Worker.
 */
export const parseBuffer = async (buffer: ArrayBuffer): Promise<ParseResult> => {
  const format = detectFormat(buffer);

  const parquetResult = format === 'parquet' ? await loadParquet(buffer) : null;
  const result = parquetResult ?? loadArrow(buffer);

  const columnNames = Object.keys(result.numeric);
  if (columnNames.length === 0) {
    throw new Error('No numeric columns found in dataset');
  }

  const firstCol = result.numeric[columnNames[0]!]!;
  const rowCount = firstCol.length;

  if (rowCount === 0) {
    throw new Error('Dataset contains no rows');
  }

  for (const name of columnNames) {
    const col = result.numeric[name]!;
    if (col.length !== rowCount) {
      throw new Error(
        `Column "${name}" has ${col.length} rows, expected ${rowCount}. All columns must have equal length.`,
      );
    }
  }

  const columns = columnNames.map((name) => {
    const values = result.numeric[name]!;
    const { min, max, range, mean } = analyzeColumn(values);
    return { name, values, min, max, range, mean };
  });

  // Only columns that can color points need a stable, sorted label order (e.g., for the legend).
  for (const column of result.categorical) {
    if (column.labels.length <= MAX_COLOR_CATEGORIES) sortLabels(column);
  }

  return {
    columns,
    categoricalColumns: result.categorical,
    rowCount,
    embeddedConfig: parquetResult?.embeddedConfig,
  };
};
