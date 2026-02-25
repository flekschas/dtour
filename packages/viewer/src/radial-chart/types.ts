export type RadialTrackConfig = {
  /** Column name in the Arrow IPC table. */
  metric: string;
  /** Track height in px. Default 16. */
  height?: number;
  /** Bar fill color. Default auto-assigned from palette. */
  color?: string;
  /** 'full' = span the segment, number = fixed px width. Default 'full'. */
  barWidth?: 'full' | number;
  /** Explicit [min, max] domain. Default: per-track normalize. */
  domain?: [number, number];
  /** Tooltip label override. Default: column name. */
  label?: string;
};

export type ParsedTrack = {
  label: string;
  rawValues: Float64Array | Float32Array;
  normalizedValues: number[];
  height: number;
  color: string;
  barWidth: 'full' | number;
};
