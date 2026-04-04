import { z } from 'zod';

/**
 * JSON-serializable spec for the Dtour component.
 * All fields optional — omitted fields use defaults.
 * The Zod schema is the single source of truth; the TS type is inferred.
 */
export const dtourSpecSchema = z.object({
  tourBy: z.enum(['dimensions', 'pca']).optional(),
  tourPosition: z.number().min(0).max(1).optional(),
  tourPlaying: z.boolean().optional(),
  tourSpeed: z.number().min(0.1).max(5).optional(),
  tourDirection: z.enum(['forward', 'backward']).optional(),
  previewCount: z.union([z.literal(4), z.literal(8), z.literal(12), z.literal(16)]).optional(),
  previewScale: z.union([z.literal(1), z.literal(0.75), z.literal(0.5)]).optional(),
  previewPadding: z.number().nonnegative().optional(),
  pointSize: z.union([z.number().positive(), z.literal('auto')]).optional(),
  pointOpacity: z.union([z.number().min(0).max(1), z.literal('auto')]).optional(),
  pointColor: z.union([z.tuple([z.number(), z.number(), z.number()]), z.string()]).optional(),
  cameraPanX: z.number().optional(),
  cameraPanY: z.number().optional(),
  cameraZoom: z.number().positive().optional(),
  viewMode: z.enum(['guided', 'manual', 'grand']).optional(),
  showLegend: z.boolean().optional(),
  showAxes: z.boolean().optional(),
  showFrameNumbers: z.boolean().optional(),
  showFrameLoadings: z.boolean().optional(),
  showTourDescription: z.boolean().optional(),
  sliderSpacing: z.enum(['equal', 'geodesic']).optional(),
  themeMode: z.enum(['light', 'dark', 'system']).optional(),
});

export type DtourSpec = z.infer<typeof dtourSpecSchema>;

/** Per-frame top-2 feature correlations: [featureName, pearsonR] pairs. */
export type FrameLoading = [string, number];

/** Parsed contents of the Parquet "dtour" key_value_metadata entry. */
export type EmbeddedConfig = {
  spec: DtourSpec;
  colorMap?: Record<string, string>;
  tour?: {
    nDims: number;
    nViews: number;
    views: Float32Array[];
    tourMode?: 'signed' | 'discriminative' | null;
    frameLoadings?: FrameLoading[][];
    /** Human-readable description of the tour (shown in description sub-bar). */
    tourDescription?: string;
    /** Template for per-frame tooltip, with {dim1}, {dim2}, {relation} placeholders. */
    tourFrameDescription?: string;
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

  // Extract colorMap (label → hex string)
  let colorMap: Record<string, string> | undefined;
  if (obj.colorMap && typeof obj.colorMap === 'object' && !Array.isArray(obj.colorMap)) {
    const cm = obj.colorMap as Record<string, unknown>;
    const valid: Record<string, string> = {};
    let hasEntries = false;
    for (const [k, v] of Object.entries(cm)) {
      if (typeof v === 'string') {
        valid[k] = v;
        hasEntries = true;
      }
    }
    if (hasEntries) colorMap = valid;
  }

  // Extract tour views (base64 float32 column-major)
  let tour: EmbeddedConfig['tour'] | undefined;
  if (obj.tour && typeof obj.tour === 'object') {
    const t = obj.tour as Record<string, unknown>;
    const nDims = typeof t.nDims === 'number' ? t.nDims : 0;
    const nViews = typeof t.nViews === 'number' ? t.nViews : 0;
    const viewsB64 = typeof t.views === 'string' ? t.views : '';
    if (nDims >= 2 && nViews >= 2 && viewsB64) {
      try {
        const binary = atob(viewsB64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
          bytes[i] = binary.charCodeAt(i);
        }
        const floats = new Float32Array(bytes.buffer);
        const stride = nDims * 2;
        if (floats.length === nViews * stride) {
          const views: Float32Array[] = [];
          for (let v = 0; v < nViews; v++) {
            views.push(floats.slice(v * stride, (v + 1) * stride));
          }
          tour = { nDims, nViews, views };

          // Parse tourMode
          if (t.tourMode === 'signed' || t.tourMode === 'discriminative') {
            tour.tourMode = t.tourMode;
          }

          // Parse tourDescription and tourFrameDescription
          if (typeof t.tourDescription === 'string') {
            tour.tourDescription = t.tourDescription;
          }
          if (typeof t.tourFrameDescription === 'string') {
            tour.tourFrameDescription = t.tourFrameDescription;
          }

          // Parse frameLoadings: array of [[name, coeff], [name, coeff]] per view
          if (Array.isArray(t.frameLoadings)) {
            const fl: FrameLoading[][] = [];
            let valid = true;
            for (const frame of t.frameLoadings as unknown[][]) {
              if (!Array.isArray(frame)) {
                valid = false;
                break;
              }
              const pairs: FrameLoading[] = [];
              for (const pair of frame) {
                if (
                  Array.isArray(pair) &&
                  pair.length === 2 &&
                  typeof pair[0] === 'string' &&
                  typeof pair[1] === 'number'
                ) {
                  pairs.push([pair[0] as string, pair[1] as number]);
                }
              }
              fl.push(pairs);
            }
            if (valid && fl.length > 0) tour.frameLoadings = fl;
          }
        }
      } catch {
        // Invalid base64 — skip tour
      }
    }
  }

  return { spec: spec as DtourSpec, colorMap, tour };
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
  pointColor: [0.25, 0.5, 0.9],
  cameraPanX: 0,
  cameraPanY: 0,
  cameraZoom: 1 / 1.5,
  viewMode: 'guided',
  showLegend: true,
  showAxes: false,
  showFrameNumbers: false,
  showFrameLoadings: true,
  showTourDescription: false,
  sliderSpacing: 'equal',
  themeMode: 'dark',
};
