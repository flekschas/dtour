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
  themeMode: z.enum(['light', 'dark', 'system']).optional(),
});

export type DtourSpec = z.infer<typeof dtourSpecSchema>;

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
  themeMode: 'dark',
};
