import { tableFromIPC } from '@uwdata/flechette';
import type { ParsedTrack, RadialTrackConfig } from './types.ts';

/** Default color palette for auto-assigned track colors. */
const PALETTE = [
  '#4080e8', // accent blue
  '#e8a040', // warm orange
  '#50c878', // green
  '#e05070', // rose
  '#9070e0', // purple
  '#40b8d0', // teal
  '#d0a040', // gold
  '#70a0c0', // steel blue
];

const DEFAULT_HEIGHT = 16;
const DEFAULT_BAR_WIDTH: 'full' | number = 'full';

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
 * Parse an Arrow IPC buffer into per-track normalized data ready for rendering.
 *
 * Each column in the Arrow table represents a metric; rows are per-view values.
 * When `configs` is provided, only the listed metrics are shown in that order.
 * When omitted, all numeric columns are shown with auto-assigned colors.
 */
export const parseMetrics = (buffer: ArrayBuffer, configs?: RadialTrackConfig[]): ParsedTrack[] => {
  const table = tableFromIPC(new Uint8Array(buffer));
  const columns = table.toColumns() as Record<string, unknown>;

  // Collect numeric columns
  const numericColumns = new Map<string, Float32Array | Float64Array>();
  for (const [name, arr] of Object.entries(columns)) {
    if (!ArrayBuffer.isView(arr)) continue;
    const tag = Object.prototype.toString.call(arr).slice(8, -1);
    if (!NUMERIC_TYPED_ARRAYS.has(tag)) continue;
    numericColumns.set(
      name,
      arr instanceof Float32Array || arr instanceof Float64Array
        ? arr
        : new Float32Array(arr as unknown as ArrayLike<number>),
    );
  }

  // Determine which tracks to show
  const trackDefs: { metric: string; config?: RadialTrackConfig }[] = configs
    ? configs
        .filter((c) => numericColumns.has(c.metric))
        .map((c) => ({ metric: c.metric, config: c }))
    : [...numericColumns.keys()].map((m) => ({ metric: m }));

  return trackDefs.map(({ metric, config }, i): ParsedTrack => {
    const raw = numericColumns.get(metric)!;

    // Normalize
    let min: number;
    let max: number;
    if (config?.domain) {
      [min, max] = config.domain;
    } else {
      min = Number.POSITIVE_INFINITY;
      max = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < raw.length; j++) {
        const v = raw[j] as number;
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const range = max - min || 1;
    const normalizedValues = Array.from(raw, (v) => Math.max(0, Math.min(1, (v - min) / range)));

    return {
      label: config?.label ?? metric,
      rawValues: raw,
      normalizedValues,
      height: config?.height ?? DEFAULT_HEIGHT,
      color: config?.color ?? (PALETTE[i % PALETTE.length] as string),
      barWidth: config?.barWidth ?? DEFAULT_BAR_WIDTH,
    };
  });
};
