import type { Colormap2DName, Metadata } from '@dtour/scatter';
import { atom } from 'jotai';
import { selectAtom } from 'jotai/utils';
import { fitTour, MIN_RANGE } from '../keyframes.ts';
import {
  fitPreviewCount,
  MAX_PREVIEW_COUNT,
  PREVIEW_SIZE_SCALE,
  PREVIEW_SPACING,
  type PreviewSize,
  type PreviewSizeSetting,
  resolvePreviewSize,
} from '../layout/gallery-positions.ts';
import { selectPreviewKeyframes } from '../layout/preview-keyframes.ts';
import { DTOUR_DEFAULTS, type EmbeddedConfig, type KeyframeLoading } from '../spec.ts';

// ---------------------------------------------------------------------------
// Tour state — controls position and playback along the tour path
// ---------------------------------------------------------------------------

/** Controls how tour keyframes are derived: raw dimension pairs or PCA eigenvectors. */
export const tourByAtom = atom<'dimensions' | 'pca' | 'parameter'>('dimensions');

export const tourPositionAtom = atom(0);
export const tourPlayingAtom = atom(false);
export const tourSpeedAtom = atom(1);
export const tourDirectionAtom = atom<1 | -1>(1);

/** Slider spacing mode: 'equal' = uniform tick spacing, 'geodesic' = arc-length proportional. */
export const tourSliderSpacingAtom = atom<'equal' | 'geodesic'>('equal');

/** Cumulative arc-lengths for the current tour bases. null when no tour is loaded. */
export const arcLengthsAtom = atom<Float32Array | null>(null);

// ---------------------------------------------------------------------------
// View state — controls preview layout and keyframe selection
// ---------------------------------------------------------------------------

export const previewCountAtom = atom(4);
/** User setting for preview size: explicit small/medium/large or viewport-derived 'auto'. */
export const previewSizeAtom = atom<PreviewSizeSetting>('auto');
export const previewPaddingAtom = atom(12);

/**
 * Resolved preview size. Resolves 'auto' from the main canvas's smaller
 * dimension so the Gallery and the circular-selector sizing agree.
 */
export const resolvedPreviewSizeAtom = atom<PreviewSize>((get) => {
  const { width, height } = get(canvasSizeAtom);
  return resolvePreviewSize(get(previewSizeAtom), Math.min(width, height));
});

/** Scale factor of the resolved preview size. */
export const resolvedPreviewScaleAtom = atom(
  (get) => PREVIEW_SIZE_SCALE[get(resolvedPreviewSizeAtom)],
);

/** Keyframe whose preview is hovered, or null. */
export const hoveredKeyframeAtom = atom<number | null>(null);

/**
 * Preview center positions relative to the container center, plus preview
 * size. Indexed by keyframe; keyframes without a preview have no entry.
 */
export const previewCentersAtom = atom<{ x: number; y: number; size: number }[]>([]);

/** Derived: nearest keyframe to the current tour position. */
export const currentKeyframeAtom = atom((get) => {
  const position = get(tourPositionAtom);
  const arcLengths = get(arcLengthsAtom);
  const previewCount = get(previewCountAtom);
  if (!arcLengths || arcLengths.length < 2) {
    return Math.round(position * previewCount) % previewCount;
  }
  const n = arcLengths.length - 1;
  let best = 0;
  let bestDist = 1;
  for (let i = 0; i < n; i++) {
    let dist = Math.abs(position - arcLengths[i]!);
    dist = Math.min(dist, 1 - dist);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
});

/** Derived: true when the tour position is >0.5% (arc-length) from every keyframe. */
export const betweenKeyframesAtom = atom((get) => {
  const position = get(tourPositionAtom);
  const arcLengths = get(arcLengthsAtom);
  if (!arcLengths || arcLengths.length < 2) return false;
  const n = arcLengths.length - 1;
  let best = 1;
  for (let i = 0; i < n; i++) {
    let dist = Math.abs(position - arcLengths[i]!);
    dist = Math.min(dist, 1 - dist);
    if (dist < best) best = dist;
  }
  return best > 0.005;
});

// ---------------------------------------------------------------------------
// Point style — visual appearance of scatter points
// ---------------------------------------------------------------------------

export const pointSizeAtom = atom<number | 'auto'>('auto');
export const pointOpacityAtom = atom<number | 'auto'>('auto');
export const pointColorAtom = atom<[number, number, number]>([0.25, 0.5, 0.9]);
export const pointColorByAtom = atom<string | null>(null);
export const minPointSizeAtom = atom(2);
export const paletteAtom = atom<'viridis' | 'magma'>('viridis');

/** Per-label color overrides. Values are hex strings or theme-aware {light, dark} objects. */
export const colorMapAtom = atom<Record<string, string | { light: string; dark: string }> | null>(
  null,
);

/** Whether 2D coloring mode is enabled. */
export const color2dEnabledAtom = atom(false);
/** Selected columns for 2D coloring (exactly 2 numeric column names). */
export const color2dColumnsAtom = atom<[string, string] | null>(null);
/** Which 2D colormap to use. */
export const color2dMapAtom = atom<Colormap2DName>('schumann');

/**
 * Color encoding as one value: a column name, an [x, y] column pair for 2D
 * coloring, or null. Reads null while a 2D pair is incomplete.
 */
export const colorEncodingAtom = atom(
  (get): string | [string, string] | null => {
    if (!get(color2dEnabledAtom)) return get(pointColorByAtom);
    const columns = get(color2dColumnsAtom);
    return columns?.[1] ? columns : null;
  },
  (_get, set, value: string | [string, string] | null) => {
    const is2d = Array.isArray(value);
    set(color2dEnabledAtom, is2d);
    set(color2dColumnsAtom, is2d ? value : null);
    set(pointColorByAtom, is2d ? null : value);
  },
);

// ---------------------------------------------------------------------------
// Background color — WebGPU clear color (RGB 0–1)
// ---------------------------------------------------------------------------

export const backgroundColorAtom = atom<[number, number, number]>([0, 0, 0]);

// ---------------------------------------------------------------------------
// Projection centering — origin definition for normalization
// ---------------------------------------------------------------------------

/** Centering mode: 'midrange' (default, (min+max)/2) or 'mean' (center of mass). */
export const centeringAtom = atom<'midrange' | 'mean'>('midrange');

/** Id that links this view's selection with other views of the same rows, or null. */
export const linkAtom = atom<string | null>(null);

// ---------------------------------------------------------------------------
// Camera state — 2D pan and zoom
// ---------------------------------------------------------------------------

export const cameraPanXAtom = atom(0);
export const cameraPanYAtom = atom(0);
export const cameraZoomAtom = atom(1 / 1.5);

/** Camera zoom limits of the wheel and the zoom slider. */
export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 4;

/** When true, scroll = zoom and Shift+scroll = tour scrub (inverted from default). */
export const panZoomModeAtom = atom(false);

// ---------------------------------------------------------------------------
// Tour traversal — controls which UI is shown (guided, manual, grand)
// ---------------------------------------------------------------------------

export const tourTraversalAtom = atom<'guided' | 'manual' | 'grand'>('guided');

/**
 * When true, `useScatter` skips `setTourPosition` messages.
 * Set on returning to guided mode from manual/grand so the current
 * projection is preserved until the user clicks the circular slider
 * or presses play.
 */
export const guidedSuspendedAtom = atom(false);

/** True while a basis-blend transition is animating back to the tour projection. */
export const basisTransitioningAtom = atom(false);

/**
 * Callback registered by DtourViewer so sibling components (e.g. DtourToolbar)
 * can trigger a smooth guided-mode resume without holding scatter refs.
 * Wrapped in an object to avoid jotai treating the function as an updater.
 */
export const resumeGuidedAtom = atom<{ fn: (durationMs: number) => void } | null>(null);

/** Target mode after grand ease-out completes. null = not exiting. */
export const grandExitTargetAtom = atom<'guided' | 'manual' | null>(null);

/** True when the 3D camera is rotated away from front-on (manual mode only). */
export const is3dRotatedAtom = atom(false);

/**
 * Tracks the currently-displayed projection basis (p×2 column-major).
 * Updated by tour interpolation, manual axis dragging, and zen animation.
 * Read imperatively (via store.get) on mode switch so the new mode
 * can initialize from the current view without jumping.
 */
export const currentBasisAtom = atom<Float32Array | null>(null);

// ---------------------------------------------------------------------------
// Animation coordination — generation counter for cancellation
// ---------------------------------------------------------------------------

/**
 * Incremented each time a position animation starts or is cancelled.
 * Running animations bail out when their captured generation doesn't
 * match the current value, ensuring only one animation drives the
 * position at a time — even across different components.
 */
export const animationGenAtom = atom(0);

// ---------------------------------------------------------------------------
// Canvas size — tracked for auto opacity/size computation
// ---------------------------------------------------------------------------

export const canvasSizeAtom = atom({ width: 0, height: 0 });

/**
 * Space for the preview gallery: the canvas below the toolbar, before the
 * gallery's insets. null until the canvas has been measured.
 */
export const galleryAreaAtom = atom<{ width: number; height: number } | null>(null);

// ---------------------------------------------------------------------------
// Read-only / derived — not exposed to AI setters
// ---------------------------------------------------------------------------

export const metadataAtom = atom<Metadata | null>(null);

/** Parsed embedded config from Parquet key_value_metadata. Reset on each data load. */
export const embeddedConfigAtom = atom<EmbeddedConfig | null>(null);

// ---------------------------------------------------------------------------
// Column visibility — which numeric dimensions participate in the tour
// ---------------------------------------------------------------------------

/**
 * Set of active dimension indices. `null` means all columns are active
 * (initial state before metadata loads or when all are enabled).
 */
export const activeColumnsAtom = atom<Set<number> | null>(null);

/**
 * Tour columns requested through the spec, applied to {@link activeColumnsAtom}
 * once metadata loads. Each request is a new object, so repeating one applies it again.
 */
export const requestedTourDimensionsAtom = atom<{ names: string[] | null }>({ names: null });

/**
 * Names of the columns an auto-generated tour uses, or null for all, in the
 * order they were requested. Reads the requested columns until metadata loads
 * and while a predefined tour is active.
 */
export const tourDimensionsAtom = atom(
  (get): string[] | null => {
    const meta = get(metadataAtom);
    if (!meta || get(predefinedTourAtom)) return get(requestedTourDimensionsAtom).names;
    const active = get(activeColumnsAtom);
    if (active === null) return null;
    return [...active].map((i) => meta.columnNames[i]!);
  },
  (_get, set, value: string[] | null) => set(requestedTourDimensionsAtom, { names: value }),
);

/**
 * Resolved active dimension indices — never null after metadata loads.
 * Returns sorted array for deterministic iteration in basis generation,
 * grand tour, and manual mode.
 */
export const activeIndicesAtom = atom<number[]>((get) => {
  const active = get(activeColumnsAtom);
  const meta = get(metadataAtom);
  if (!meta) return [];
  if (active === null) return Array.from({ length: meta.dimCount }, (_, i) => i);
  return Array.from(active).sort((a, b) => a - b);
});

/**
 * The [x, y] column indices of the static scatter shown in place of an
 * auto-generated tour that projects two columns, which has nothing to tour.
 * The order of the tour dimensions decides which column is x. A PCA tour
 * projects all numeric columns. null while touring.
 */
export const staticAxesAtom = atom<[number, number] | null>((get) => {
  const meta = get(metadataAtom);
  if (!meta || get(fittedTourAtom)) return null;
  const active = get(tourByAtom) === 'pca' ? null : get(activeColumnsAtom);
  const indices =
    active === null ? Array.from({ length: meta.dimCount }, (_, i) => i) : [...active];
  return indices.length === 2 ? [indices[0]!, indices[1]!] : null;
});

/**
 * Zoom of an untouched camera. Tours zoom out to leave room for rotating
 * projections. A static scatter zooms to fit its points around the centering
 * origin into the canvas, within the zoom limits.
 */
export const defaultCameraZoomAtom = atom((get) => {
  const axes = get(staticAxesAtom);
  const meta = get(metadataAtom);
  if (!axes || !meta) return DTOUR_DEFAULTS.cameraZoom;
  const centering = get(centeringAtom);
  // Coordinates as createStaticKeyframe and the renderer compute them
  const scale = Math.max(...axes.map((d) => Math.max(meta.ranges[d]!, MIN_RANGE)));
  const extent = (d: number) => {
    const center = centering === 'mean' ? meta.means[d]! : meta.mins[d]! + meta.ranges[d]! / 2;
    return (2 * Math.max(meta.maxes[d]! - center, center - meta.mins[d]!)) / scale;
  };
  const { width, height } = get(canvasSizeAtom);
  const aspect = height > 0 ? width / height : 1;
  const zoom = Math.min(aspect / extent(axes[0]), 1 / extent(axes[1]));
  return Number.isFinite(zoom) ? Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom)) : 1;
});

// ---------------------------------------------------------------------------
// Legend — collapsible color legend panel
// ---------------------------------------------------------------------------

/** User preference for showing the legend panel. */
export const showLegendAtom = atom(true);

/** User preference for showing axis biplot in guided mode. */
export const showAxesAtom = atom(false);

/** Keyframe numbers on previews. 'auto' shows them only when some keyframes have no preview. */
export const previewKeyframeNumbersAtom = atom<'auto' | 'visible' | 'hidden'>('auto');

/** Preview label content. 'auto' shows feature loadings when available, else the keyframe description. */
export const previewLabelContentAtom = atom<'auto' | 'description' | 'loadings'>('auto');

/**
 * When preview labels show. 'interactive' shows them over the preview on hover
 * and for the current keyframe. 'auto' is 'visible' up to
 * {@link MAX_PREVIEWS_WITH_VISIBLE_LABELS} previews and 'interactive' above.
 */
export const previewLabelVisibilityAtom = atom<'auto' | 'visible' | 'interactive' | 'hidden'>(
  'auto',
);

/** Up to this many previews, 'auto' labels stay visible below each preview. */
const MAX_PREVIEWS_WITH_VISIBLE_LABELS = 16;

/** Label visibility for a preview count, with 'auto' resolved. */
const labelVisibilityFor = (
  setting: 'auto' | 'visible' | 'interactive' | 'hidden',
  previewCount: number,
) => {
  if (setting !== 'auto') return setting;
  return previewCount > MAX_PREVIEWS_WITH_VISIBLE_LABELS ? 'interactive' : 'visible';
};

/** User preference for showing the tour description sub-bar. null = derive from tourDescription. */
export const showTourDescriptionAtom = atom<boolean | null>(null);

/** Circular slider visibility: 'visible' (full), 'subtle' (translucent + thinner), 'hidden'. */
export const sliderVisibilityAtom = atom<'visible' | 'subtle' | 'hidden'>('visible');

/** Per-keyframe feature loadings from embedded tour config. */
export const keyframeLoadingsAtom = atom<KeyframeLoading[] | null>(null);

/** Tour family: hyperdimensional (one high-D space) or sequential (multiple 2D embeddings). */
export const tourFamilyAtom = atom<'hyperdimensional' | 'sequential'>('hyperdimensional');

/**
 * Whether tour interpolation keeps the bases orthonormal. Sequential tours and
 * static scatters blend their keyframes directly instead.
 */
export const orthonormalizeAtom = atom(
  (get) => get(tourFamilyAtom) !== 'sequential' && !get(staticAxesAtom),
);

// ---------------------------------------------------------------------------
// Predefined tour — locks column selection, preview count, and Dims/PCA toggle
// ---------------------------------------------------------------------------

/**
 * Tour keyframes passed to the viewer and the names of the numeric columns
 * they project (default: the first p numeric columns). null without keyframes.
 */
export const suppliedTourAtom = atom<{
  keyframes: Float32Array[];
  dimensions: string[] | undefined;
} | null>(null);

/**
 * The supplied tour, or else the tour embedded in the data, fitted to the
 * data's numeric columns. null to generate a tour. Names supplied with the
 * keyframes must match the data; an embedded tour falls back to the first columns.
 */
export const fittedTourAtom = atom((get) => {
  const meta = get(metadataAtom);
  if (!meta) return null;
  const supplied = get(suppliedTourAtom);
  if (supplied) return fitTour(supplied.keyframes, meta.columnNames, supplied.dimensions, true);
  const embedded = get(embeddedConfigAtom)?.tour;
  if (embedded && embedded.keyframes.length > 0) {
    return fitTour(embedded.keyframes, meta.columnNames, embedded.dimensions, false);
  }
  return null;
});

/** Info about the active predefined tour, or null for auto-generated tours.
 *  When non-null, column toggles, preview count slider, and Dims/PCA toggle are disabled. */
export const predefinedTourAtom = atom((get) => {
  const fitted = get(fittedTourAtom);
  return fitted ? { dimensions: fitted.dimensions, keyframeCount: fitted.keyframes.length } : null;
});

/** Whether the supplied tour doesn't fit the data, so the viewer shows an auto-generated tour. */
export const tourRejectedAtom = atom((get) => {
  if (!get(metadataAtom) || get(fittedTourAtom)) return false;
  return (
    get(suppliedTourAtom) !== null || (get(embeddedConfigAtom)?.tour?.keyframes.length ?? 0) > 0
  );
});

/**
 * Number of tour keyframes: from the predefined tour, 1 for a static scatter,
 * otherwise {@link previewCountAtom}.
 */
export const keyframeCountAtom = atom((get) => {
  if (get(staticAxesAtom)) return 1;
  return get(predefinedTourAtom)?.keyframeCount ?? get(previewCountAtom);
});

/**
 * Number of previews shown: one per keyframe, up to {@link MAX_PREVIEW_COUNT}
 * and as many as fit the gallery at a readable size. 0 when not even two fit
 * and for a static scatter.
 */
export const resolvedPreviewCountAtom = atom((get) => {
  if (get(staticAxesAtom)) return 0;
  const maxCount = Math.min(get(keyframeCountAtom), MAX_PREVIEW_COUNT);
  const area = get(galleryAreaAtom);
  // Before measuring, assume everything fits so canvases are not rebuilt on startup
  if (!area) return maxCount;
  const hasLabels = get(resolvedPreviewLabelContentAtom) !== null;
  const labelSetting = get(previewLabelVisibilityAtom);
  return fitPreviewCount(
    area.width - 2 * PREVIEW_SPACING,
    area.height - 2 * PREVIEW_SPACING,
    maxCount,
    get(resolvedPreviewScaleAtom),
    (previewCount) => hasLabels && labelVisibilityFor(labelSetting, previewCount) === 'visible',
  );
});

/**
 * Keyframe shown by each preview. Tours with more keyframes than previews show
 * the subset that is most evenly spaced along the tour, always including the
 * first and last keyframe. Keeps its identity while the selection is unchanged,
 * so preview canvases are only rebuilt when they show different keyframes.
 */
export const previewKeyframesAtom = selectAtom(
  atom((get) =>
    selectPreviewKeyframes(
      get(keyframeCountAtom),
      get(resolvedPreviewCountAtom),
      get(arcLengthsAtom),
    ),
  ),
  (keyframes) => keyframes,
  (a, b) => a.length === b.length && a.every((keyframe, i) => keyframe === b[i]),
);

/** Whether previews show keyframe numbers, with 'auto' resolved. */
export const resolvedPreviewKeyframeNumbersAtom = atom((get) => {
  const setting = get(previewKeyframeNumbersAtom);
  if (setting !== 'auto') return setting;
  return get(keyframeCountAtom) > get(resolvedPreviewCountAtom) ? 'visible' : 'hidden';
});

/** Per-keyframe descriptions: string[] of literals, or a template string with
 *  {primary}, {secondary}, {relation} placeholders. */
export const keyframeDescriptionsAtom = atom<string | string[] | null>(null);

/** What preview labels show, with 'auto' resolved. null when there is nothing to show. */
export const resolvedPreviewLabelContentAtom = atom((get) => {
  const setting = get(previewLabelContentAtom);
  const loadings = get(keyframeLoadingsAtom);
  if (setting !== 'description' && loadings && loadings.length > 0) return 'loadings';
  if (setting !== 'loadings' && Array.isArray(get(keyframeDescriptionsAtom))) return 'description';
  return null;
});

/** When preview labels show, with 'auto' resolved. 'hidden' when there is nothing to show. */
export const resolvedPreviewLabelVisibilityAtom = atom((get) => {
  if (get(resolvedPreviewLabelContentAtom) === null) return 'hidden';
  return labelVisibilityFor(get(previewLabelVisibilityAtom), get(resolvedPreviewCountAtom));
});

/** Tour description string from embedded config (shown in description sub-bar). */
export const tourDescriptionAtom = atom<string | null>(null);

/**
 * Derived: legend is visible only when showLegend is true, metadata is loaded,
 * AND points are colored by a known data column (numeric or categorical).
 */
export const legendVisibleAtom = atom((get) => {
  if (!get(showLegendAtom)) return false;
  const meta = get(metadataAtom);
  if (!meta) return false;
  // 2D color mode has its own legend (only when both columns are selected)
  const cols2d = get(color2dColumnsAtom);
  if (get(color2dEnabledAtom) && cols2d?.[1]) return true;
  const colorBy = get(pointColorByAtom);
  if (!colorBy) return false;
  return meta.columnNames.includes(colorBy) || meta.categoricalColumnNames.includes(colorBy);
});

// ---------------------------------------------------------------------------
// Legend selection — which legend entries are actively selected
// ---------------------------------------------------------------------------

/** Which legend entries are selected, or null when no legend selection is active. */
export const legendSelectionAtom = atom<Set<number> | null>(null);

/** Bumped when ColorLegend explicitly deselects — triggers scatter.clearSelection(). */
export const legendClearGenAtom = atom(0);

// ---------------------------------------------------------------------------
// Theme — light/dark mode with system preference support
// ---------------------------------------------------------------------------

/** User preference: explicit light/dark or follow system. */
export const themeModeAtom = atom<'light' | 'dark' | 'system'>('dark');

/** Tracks the OS-level color scheme. Updated by useSystemTheme hook. */
export const systemThemeAtom = atom<'light' | 'dark'>('dark');

/** Resolved theme after applying system preference. */
export const resolvedThemeAtom = atom<'light' | 'dark'>((get) => {
  const mode = get(themeModeAtom);
  return mode === 'system' ? get(systemThemeAtom) : mode;
});
