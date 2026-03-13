import type { Metadata } from '@dtour/scatter';
import { atom } from 'jotai';

// ---------------------------------------------------------------------------
// Tour state — controls position and playback along the tour path
// ---------------------------------------------------------------------------

export const tourPositionAtom = atom(0);
export const tourPlayingAtom = atom(false);
export const tourSpeedAtom = atom(1);
export const tourDirectionAtom = atom<1 | -1>(1);

// ---------------------------------------------------------------------------
// View state — controls preview layout and keyframe selection
// ---------------------------------------------------------------------------

export const previewCountAtom = atom<4 | 8 | 12 | 16>(4);
export const previewScaleAtom = atom<1 | 0.75 | 0.5>(1);
export const previewPaddingAtom = atom(12);
export const selectedKeyframeAtom = atom<number | null>(null);

// ---------------------------------------------------------------------------
// Point style — visual appearance of scatter points
// ---------------------------------------------------------------------------

export const pointSizeAtom = atom<number | 'auto'>('auto');
export const pointOpacityAtom = atom<number | 'auto'>('auto');
export const pointColorAtom = atom<[number, number, number] | string>([0.25, 0.5, 0.9]);
export const paletteAtom = atom<'viridis' | 'magma'>('viridis');

// ---------------------------------------------------------------------------
// Background color — WebGPU clear color (RGB 0–1)
// ---------------------------------------------------------------------------

export const backgroundColorAtom = atom<[number, number, number]>([0, 0, 0]);

// ---------------------------------------------------------------------------
// Camera state — 2D pan and zoom
// ---------------------------------------------------------------------------

export const cameraPanXAtom = atom(0);
export const cameraPanYAtom = atom(0);
export const cameraZoomAtom = atom(1 / 1.5);

// ---------------------------------------------------------------------------
// View mode — controls which UI is shown (guided, manual, grand)
// ---------------------------------------------------------------------------

export const viewModeAtom = atom<'guided' | 'manual' | 'grand'>('guided');

/**
 * When true, `useScatter` skips `setTourPosition` messages.
 * Set on returning to guided mode from manual/grand so the current
 * projection is preserved until the user clicks the circular slider
 * or presses play.
 */
export const guidedSuspendedAtom = atom(false);

/** Target mode after grand ease-out completes. null = not exiting. */
export const grandExitTargetAtom = atom<'guided' | 'manual' | null>(null);

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

/**
 * Derived: legend is visible only when showLegend is true AND points are
 * colored by a data column (string that isn't a hex color).
 */
export const legendVisibleAtom = atom((get) => {
  if (!get(showLegendAtom)) return false;
  const color = get(pointColorAtom);
  return typeof color === 'string' && !/^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(color);
});

// ---------------------------------------------------------------------------
// Legend selection — which legend entries are actively selected
// ---------------------------------------------------------------------------

/** Which legend entries are selected, or null when no legend selection is active. */
export const legendSelectionAtom = atom<Set<number> | null>(null);

/** Bumped when ColorLegend explicitly deselects — triggers scatter.clearSelection(). */
export const legendClearGenAtom = atom(0);

// ---------------------------------------------------------------------------
// Settings persistence — localStorage keyed by data name
// ---------------------------------------------------------------------------

export const dataNameAtom = atom<string | null>(null);
