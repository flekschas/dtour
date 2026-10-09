import type { DtourSpec } from '@dtour/viewer';
import { dtourSpecSchema } from '@dtour/viewer';

export const URL_SPEC_KEYS = Object.keys(dtourSpecSchema.shape) as (keyof DtourSpec)[];

function parseJson(text: string): { value: unknown } | null {
  try {
    return { value: JSON.parse(text) };
  } catch {
    return null;
  }
}

/**
 * Read spec fields from URL parameters named like the spec fields. A value is
 * JSON, e.g. `0.5`, `true`, or `["x","y"]`, or plain text for strings.
 * Invalid fields are dropped with a warning.
 */
export function readUrlSpec(params: URLSearchParams): DtourSpec {
  const spec: Record<string, unknown> = {};
  for (const key of URL_SPEC_KEYS) {
    const text = params.get(key);
    if (text === null) continue;
    const field = dtourSpecSchema.shape[key];
    const json = parseJson(text);
    const result = json ? field.safeParse(json.value) : null;
    const parsed = result?.success ? result : field.safeParse(text);
    if (parsed.success && parsed.data !== undefined) {
      spec[key] = parsed.data;
    } else {
      console.warn(`[dtour] Ignoring invalid URL parameter ${key}=${text}`);
    }
  }
  return spec as DtourSpec;
}

function formatParam(value: unknown): string {
  // Plain text unless the string would read back as JSON, e.g. a column named "1"
  if (typeof value === 'string' && !parseJson(value)) return value;
  return JSON.stringify(value);
}

/**
 * Write the spec fields that differ from `baseline` to `params` and remove the
 * others. Fields `spec` leaves undefined keep their parameters.
 */
export function writeUrlSpec(params: URLSearchParams, spec: DtourSpec, baseline: DtourSpec): void {
  for (const key of URL_SPEC_KEYS) {
    const value = spec[key];
    if (value === undefined) continue;
    if (JSON.stringify(value) === JSON.stringify(baseline[key])) {
      params.delete(key);
    } else {
      params.set(key, formatParam(value));
    }
  }
}
