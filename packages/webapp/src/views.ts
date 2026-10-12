import type { DtourSpec } from '@dtour/viewer';
import { parseSpecFields } from './url-spec.ts';

/** A view's settings and, optionally, the URL of its own data. Views without `data` show the page's data. */
export type ViewSpec = DtourSpec & { data?: string };

/** View indices by row: `[0, 1]` is one row, `[[0, 1], [2, 3]]` a 2×2 grid. Rows can differ in length. */
export type Layout = number[] | number[][];

function parseJson(text: string | null): unknown {
  if (text === null) return undefined;
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

/** Read the `views` URL parameter: a JSON array of view settings. */
export function readUrlViews(params: URLSearchParams): ViewSpec[] | null {
  const value = parseJson(params.get('views'));
  if (value === undefined) return null;
  if (!Array.isArray(value) || value.length === 0) {
    console.warn('[dtour] Ignoring the views URL parameter, which must be a non-empty JSON array');
    return null;
  }
  return value.map((entry) => {
    const fields = typeof entry === 'object' && entry !== null ? entry : {};
    const spec: ViewSpec = parseSpecFields(fields, 'views');
    if (typeof fields.data === 'string') spec.data = fields.data;
    return spec;
  });
}

/** Read the `layout` URL parameter. */
export function readUrlLayout(params: URLSearchParams): Layout | undefined {
  const value = parseJson(params.get('layout'));
  return value === undefined ? undefined : (value as Layout);
}

/** The rows of view indices. A layout lists every view once; an invalid layout shows all views in one row. */
export function layoutRows(layout: Layout | undefined, viewCount: number): number[][] {
  const all = Array.from({ length: viewCount }, (_, i) => i);
  if (layout === undefined) return [all];
  const rows = Array.isArray(layout) && layout.every(Array.isArray) ? layout : [layout];
  const indices = rows.flat();
  const isValid =
    rows.length > 0 &&
    rows.every((row) => Array.isArray(row) && row.length > 0) &&
    indices.every((i) => Number.isInteger(i) && i >= 0 && i < viewCount) &&
    new Set(indices).size === viewCount &&
    indices.length === viewCount;
  if (isValid) return rows as number[][];
  console.warn(`[dtour] Ignoring the invalid layout ${JSON.stringify(layout)}`);
  return [all];
}
