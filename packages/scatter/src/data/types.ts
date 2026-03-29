export type ColumnData = {
  name: string;
  values: Float32Array;
  min: number;
  max: number;
  range: number;
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

export type Metadata = {
  columnNames: string[];
  categoricalColumnNames: string[];
  categoricalLabels: Record<string, string[]>;
  rowCount: number;
  dimCount: number;
  mins: number[];
  maxes: number[];
  ranges: number[];
  /** Raw JSON string from the Parquet "dtour" key_value_metadata entry. Undefined for Arrow files. */
  embeddedConfig?: string;
};
