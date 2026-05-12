import { z } from 'zod';

export type PreviewCount = 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16;

const previewCountSchema = z.union([
  z.literal(2),
  z.literal(3),
  z.literal(4),
  z.literal(5),
  z.literal(6),
  z.literal(7),
  z.literal(8),
  z.literal(9),
  z.literal(10),
  z.literal(11),
  z.literal(12),
  z.literal(13),
  z.literal(14),
  z.literal(15),
  z.literal(16),
]);

/**
 * JSON-serializable spec for the Dtour component.
 * All fields optional — omitted fields use defaults.
 * The Zod schema is the single source of truth; the TS type is inferred.
 */
export const dtourSpecSchema = z.object({
  tourTraversal: z.enum(['guided', 'manual', 'grand']).optional(),
  tourBy: z.enum(['dimensions', 'pca', 'parameter']).optional(),
  tourPosition: z.number().min(0).max(1).optional(),
  tourPlaying: z.boolean().optional(),
  tourSpeed: z.number().min(0.1).max(5).optional(),
  tourDirection: z.enum(['forward', 'backward']).optional(),
  tourSliderSpacing: z.enum(['equal', 'geodesic']).optional(),
  tourSliderVisibility: z.enum(['visible', 'subtle', 'hidden']).optional(),
  previewCount: previewCountSchema.optional(),
  previewScale: z.union([z.literal(1), z.literal(0.75), z.literal(0.5)]).optional(),
  previewPadding: z.number().nonnegative().optional(),
  pointSize: z.union([z.number().positive(), z.literal('auto')]).optional(),
  pointOpacity: z.union([z.number().min(0).max(1), z.literal('auto')]).optional(),
  minPointSize: z.number().min(1).max(20).optional(),
  pointColor: z.tuple([z.number(), z.number(), z.number()]).optional(),
  pointColorBy: z.string().nullable().optional(),
  pointColorMap: z.record(z.string(), z.string()).optional(),
  cameraPanX: z.number().optional(),
  cameraPanY: z.number().optional(),
  cameraZoom: z.number().positive().optional(),
  showLegend: z.boolean().optional(),
  showAxes: z.boolean().optional(),
  showKeyframeNumbers: z.boolean().optional(),
  showKeyframeLoadings: z.boolean().optional(),
  showTourDescription: z.boolean().nullable().optional(),
  themeMode: z.enum(['light', 'dark', 'system']).optional(),
  centering: z.enum(['midrange', 'mean']).optional(),
});

export type DtourSpec = z.infer<typeof dtourSpecSchema>;

/** Per-keyframe feature loading: primary and secondary feature with their correlation coefficients. */
export interface KeyframeLoading {
  primary: [string, number];
  secondary: [string, number];
}

/** Parsed contents of the Parquet "dtour" key_value_metadata entry. */
export type EmbeddedConfig = {
  spec: DtourSpec;
  tour?: {
    /** Tour family: hyperdimensional (one high-D space) or sequential (multiple 2D embeddings). */
    family: 'hyperdimensional' | 'sequential';
    /** Projection matrices — one per keyframe. Each is a column-major Float32Array of size nDims*2. */
    keyframes: Float32Array[];
    /** Numeric column names that participate in the tour. */
    dimensions: string[];
    /** Human-readable description of the tour. When absent, no description is shown. */
    description?: string;
    /** Per-keyframe descriptions: a string[] of literals, or a single template string
     *  with {primary}, {secondary}, {relation} placeholders (requires keyframeLoadings).
     *  When absent, no per-keyframe descriptions are shown. */
    keyframeDescriptions?: string | string[];
    /** Per-keyframe feature loadings for the loading pills UI.
     *  When absent, no loading pills are shown. */
    keyframeLoadings?: KeyframeLoading[];
  };
};

const SPEC_SHAPE_KEYS = Object.keys(dtourSpecSchema.shape) as (keyof DtourSpec)[];

/**
 * Parse the raw JSON "dtour" value from Parquet key_value_metadata.
 * Returns null if the string is falsy or unparseable.
 * Invalid spec fields are silently dropped.
 */
export function parseEmbeddedConfig(raw: string | undefined): EmbeddedConfig | null {
  if (!raw) return null;

  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof obj !== 'object' || obj === null) return null;

  // Validate each spec field individually — invalid fields are dropped
  // without affecting valid ones.
  const spec: Record<string, unknown> = {};
  for (const key of SPEC_SHAPE_KEYS) {
    if (!(key in obj)) continue;
    const fieldSchema = dtourSpecSchema.shape[key];
    const result = fieldSchema.safeParse(obj[key]);
    if (result.success) spec[key] = result.data;
  }

  // Extract tour data
  let tour: EmbeddedConfig['tour'] | undefined;
  if (obj.tour && typeof obj.tour === 'object') {
    const t = obj.tour as Record<string, unknown>;
    // nDims and nViews are only needed for decoding the base64 blob
    const nDims = typeof t.nDims === 'number' ? t.nDims : 0;
    const nKeyframes = typeof t.nViews === 'number' ? t.nViews : 0;
    const viewsB64 = typeof t.views === 'string' ? t.views : '';
    if (nDims >= 2 && nKeyframes >= 2 && viewsB64) {
      try {
        const binary = atob(viewsB64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const floats = new Float32Array(bytes.buffer);
        const stride = nDims * 2;
        // Validate dimensions (required) and keyframe count
        const rawDims = Array.isArray(t.dimensions) ? (t.dimensions as unknown[]) : null;
        const dimensions = rawDims?.every((s) => typeof s === 'string')
          ? (rawDims as string[])
          : null;

        const family =
          t.family === 'sequential'
            ? 'sequential'
            : t.family === 'hyperdimensional'
              ? 'hyperdimensional'
              : null;

        if (floats.length === nKeyframes * stride && dimensions && family) {
          const keyframes: Float32Array[] = [];
          for (let v = 0; v < nKeyframes; v++) {
            keyframes.push(floats.slice(v * stride, (v + 1) * stride));
          }
          tour = { family, keyframes, dimensions };

          if (typeof t.description === 'string') {
            tour.description = t.description;
          }

          if (Array.isArray(t.keyframeDescriptions)) {
            const kd = t.keyframeDescriptions as unknown[];
            if (kd.every((s) => typeof s === 'string')) {
              tour.keyframeDescriptions = kd as string[];
            }
          } else if (typeof t.keyframeDescriptions === 'string') {
            tour.keyframeDescriptions = t.keyframeDescriptions;
          }

          if (Array.isArray(t.keyframeLoadings)) {
            const kl: KeyframeLoading[] = [];
            let valid = true;
            for (const entry of t.keyframeLoadings as unknown[]) {
              if (
                entry &&
                typeof entry === 'object' &&
                'primary' in entry &&
                'secondary' in entry
              ) {
                const e = entry as { primary: unknown; secondary: unknown };
                if (
                  Array.isArray(e.primary) &&
                  e.primary.length === 2 &&
                  typeof e.primary[0] === 'string' &&
                  typeof e.primary[1] === 'number' &&
                  Array.isArray(e.secondary) &&
                  e.secondary.length === 2 &&
                  typeof e.secondary[0] === 'string' &&
                  typeof e.secondary[1] === 'number'
                ) {
                  kl.push({
                    primary: [e.primary[0], e.primary[1]],
                    secondary: [e.secondary[0], e.secondary[1]],
                  });
                } else {
                  valid = false;
                  break;
                }
              } else {
                valid = false;
                break;
              }
            }
            if (valid && kl.length > 0) tour.keyframeLoadings = kl;
          }
        }
      } catch {
        // Invalid base64 — skip tour
      }
    }
  }

  const result = { spec: spec as DtourSpec, tour };
  if (import.meta.env.DEV) {
    console.log('[dtour] parseEmbeddedConfig:', {
      spec,
      tour: tour
        ? {
            dimensions: tour.dimensions,
            keyframes: tour.keyframes.length,
            family: tour.family,
            description: tour.description,
            keyframeDescriptions: tour.keyframeDescriptions,
            keyframeLoadings: tour.keyframeLoadings?.length ?? null,
          }
        : undefined,
    });
  }
  return result;
}

export const DTOUR_DEFAULTS: Required<DtourSpec> = {
  tourBy: 'dimensions',
  tourPosition: 0,
  tourPlaying: false,
  tourSpeed: 1,
  tourDirection: 'forward',
  previewCount: 4,
  previewScale: 1,
  previewPadding: 12,
  pointSize: 'auto',
  pointOpacity: 'auto',
  minPointSize: 2,
  pointColor: [0.25, 0.5, 0.9],
  pointColorBy: null,
  pointColorMap: {},
  cameraPanX: 0,
  cameraPanY: 0,
  cameraZoom: 1 / 1.5,
  tourTraversal: 'guided',
  showLegend: true,
  showAxes: false,
  showKeyframeNumbers: false,
  showKeyframeLoadings: true,
  showTourDescription: null,
  tourSliderVisibility: 'visible',
  tourSliderSpacing: 'equal',
  themeMode: 'dark',
  centering: 'midrange',
};
