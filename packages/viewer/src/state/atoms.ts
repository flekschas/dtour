import type { Colormap2DName, Metadata } from '@dtour/scatter';
import { atom } from 'jotai';
import {
  type PreviewScale,
  type PreviewScaleSetting,
  resolvePreviewScale,
} from '../layout/gallery-positions.ts';
import type { EmbeddedConfig, KeyframeLoading, PreviewCount } from '../spec.ts';

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

export const previewCountAtom = atom<PreviewCount>(4);
/** User setting for preview size: explicit S/M/L factor or viewport-derived 'auto'. */
export const previewScaleAtom = atom<PreviewScaleSetting>('auto');
export const previewPaddingAtom = atom(12);

/**
 * Resolved preview-size factor (S/M/L). Resolves 'auto' from the main canvas's
 * smaller dimension so the Gallery and the circular-selector sizing agree.
 */
export const resolvedPreviewScaleAtom = atom<PreviewScale>((get) => {
  const setting = get(previewScaleAtom);
  const { width, height } = get(canvasSizeAtom);
  return resolvePreviewScale(setting, Math.min(width, height));
});
export const selectedKeyframeAtom = atom<number | null>(null);

/** Which gallery preview is currently hovered (index), or null. */
export const hoveredKeyframeAtom = atom<number | null>(null);

/** Preview center positions relative to the container center, plus preview size. */
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

// ---------------------------------------------------------------------------
// Background color — WebGPU clear color (RGB 0–1)
// ---------------------------------------------------------------------------

export const backgroundColorAtom = atom<[number, number, number]>([0, 0, 0]);

// ---------------------------------------------------------------------------
// Projection centering — origin definition for normalization
// ---------------------------------------------------------------------------

/** Centering mode: 'midrange' (default, (min+max)/2) or 'mean' (center of mass). */
export const centeringAtom = atom<'midrange' | 'mean'>('midrange');

// ---------------------------------------------------------------------------
// Camera state — 2D pan and zoom
// ---------------------------------------------------------------------------

export const cameraPanXAtom = atom(0);
export const cameraPanYAtom = atom(0);
export const cameraZoomAtom = atom(1 / 1.5);

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

// ---------------------------------------------------------------------------
// Legend — collapsible color legend panel
// ---------------------------------------------------------------------------

/** User preference for showing the legend panel. */
export const showLegendAtom = atom(true);

/** User preference for showing axis biplot in guided mode. */
export const showAxesAtom = atom(false);

/** User preference for showing keyframe numbers on preview thumbnails. */
export const showKeyframeNumbersAtom = atom(false);

/** User preference for showing feature loading pills on preview thumbnails. */
export const showKeyframeLoadingsAtom = atom(true);

/** User preference for showing the tour description sub-bar. null = derive from tourDescription. */
export const showTourDescriptionAtom = atom<boolean | null>(null);

/** Circular slider visibility: 'visible' (full), 'subtle' (translucent + thinner), 'hidden'. */
export const sliderVisibilityAtom = atom<'visible' | 'subtle' | 'hidden'>('visible');

/** Per-keyframe feature loadings from embedded tour config. */
export const keyframeLoadingsAtom = atom<KeyframeLoading[] | null>(null);

/** Tour family: hyperdimensional (one high-D space) or sequential (multiple 2D embeddings). */
export const tourFamilyAtom = atom<'hyperdimensional' | 'sequential'>('hyperdimensional');

// ---------------------------------------------------------------------------
// Predefined tour — locks column selection, preview count, and Dims/PCA toggle
// ---------------------------------------------------------------------------

/** Info about the active predefined tour, or null for auto-generated tours.
 *  When non-null, column toggles, preview count slider, and Dims/PCA toggle are disabled. */
export const predefinedTourAtom = atom<{
  /** Numeric column names that participate in the tour. */
  dimensions: string[];
  /** Number of keyframes in the tour. */
  keyframeCount: number;
} | null>(null);

/** Per-keyframe descriptions: string[] of literals, or a template string with
 *  {primary}, {secondary}, {relation} placeholders. */
export const keyframeDescriptionsAtom = atom<string | string[] | null>(null);

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
