export type ColumnData = {
  name: string;
  values: Float32Array;
  min: number;
  max: number;
  range: number;
  mean: number;
};

export type CategoricalColumn = {
  name: string;
  indices: Uint32Array;
  labels: string[];
};

export type AnalyzedData = {
  columns: ColumnData[];
  categoricalColumns: CategoricalColumn[];
  rowCount: number;
};

/** The most rows {@link Metadata.sample} holds. */
export const SAMPLE_ROWS = 1024;

export type Metadata = {
  columnNames: string[];
  categoricalColumnNames: string[];
  categoricalLabels: Record<string, string[]>;
  rowCount: number;
  dimCount: number;
  mins: number[];
  maxes: number[];
  ranges: number[];
  means: number[];
  /**
   * Evenly spaced rows of the numeric columns, at most {@link SAMPLE_ROWS},
   * row-major (`sample[row * dimCount + dim]`), e.g., to estimate how far
   * projections spread.
   */
  sample: Float32Array;
  /** Raw JSON string from the Parquet "dtour" key_value_metadata entry. Undefined for Arrow files. */
  embeddedConfig?: string;
};
