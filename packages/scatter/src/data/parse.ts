import { analyzeColumn } from './analyze.ts';
import { loadArrow } from './arrow.ts';
import { detectFormat } from './detect.ts';
import { loadParquet } from './parquet.ts';
import type { AnalyzedData } from './types.ts';

export type ParseResult = AnalyzedData & {
  /** Raw JSON from Parquet "dtour" key_value_metadata. Undefined for Arrow. */
  embeddedConfig?: string;
};

/**
 * Detect format, load, extract numeric columns, normalize to [-1, 1].
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
    const { min, max, range } = analyzeColumn(values);
    return { name, values, min, max, range };
  });

  return {
    columns,
    categoricalColumns: result.categorical,
    rowCount,
    embeddedConfig: parquetResult?.embeddedConfig,
  };
};
