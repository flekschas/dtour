/** Gap between adjacent previews (CSS px). */
export const GAP = 32;
/** Maximum preview size (CSS px). */
export const MAX_SIZE = 320;
/** Height of the loading bar below/above each preview (CSS px). */
export const LOADING_BAR_HEIGHT = 24;
/**
 * Per-edge-count ratio arrays.
 *   k=1 (4 previews)  → [1]             all same
 *   k=2 (8 previews)  → [1, 0.5]        corners 1, middle 0.5
 *   k=3 (12 previews) → [1, 0.75]       corners 1, mid 0.75
 *   k=4 (16 previews) → [1, 0.75, 0.5]  corners 1, near-corner 0.75, middle 0.5
 */
const RATIOS_BY_K: Record<number, readonly number[]> = {
  1: [1],
  2: [1, 0.5],
  3: [1, 0.75],
  4: [1, 0.75, 0.5],
};

const FALLBACK_RATIOS: readonly number[] = [1, 0.75, 0.5];

/**
 * Size ratio for position `j` on an edge of `k` emitted points.
 * distFromCorner = min(j, k - j), then looked up in the per-k ratio table.
 */
export function sizeRatio(j: number, k: number): number {
  const ratios = RATIOS_BY_K[k] ?? FALLBACK_RATIOS;
  const dist = Math.min(j, k - j);
  return ratios[Math.min(dist, ratios.length - 1)] ?? 1;
}

export type GallerySizes = {
  /** CSS grid-template-columns value (e.g. "320px 256px 320px") */
  gridTemplateColumns: string;
  /** CSS grid-template-rows value */
  gridTemplateRows: string;
  /** Per-preview pixel size (indexed by preview slot) */
  sizes: number[];
  /** Largest preview size in px (for padding/selector math) */
  previewSize: number;
  /** Horizontal padding (px) */
  padX: number;
  /** Vertical padding (px) */
  padY: number;
};

/**
 * Compute ratio-weighted CSS grid templates and preview sizes.
 *
 * Track sizes are pixel values derived from the **shorter** container
 * dimension so previews stay square and identically sized on both axes.
 * `space-between` on the grid container distributes extra space on the
 * longer axis to push tracks to the perimeter edges.
 *
 *   k=1 → "Spx Spx"
 *   k=2 → "Spx 0.8Spx Spx"
 *   k=3 → "Spx 0.8Spx 0.8Spx Spx"
 *   k=4 → "Spx 0.8Spx 0.6Spx 0.8Spx Spx"
 */
export function computeGallerySizes(
  containerWidth: number,
  containerHeight: number,
  previewCount: number,
  scale = 1,
  showLoadings = false,
): GallerySizes {
  const k = Math.max(1, previewCount / 4);
  const numTracks = k + 1;

  // Build ratio array for the grid tracks
  const ratios: number[] = [];
  let ratioSum = 0;
  for (let j = 0; j <= k; j++) {
    const r = sizeRatio(j, k);
    ratios.push(r);
    ratioSum += r;
  }

  // When loading bars are shown, each row track needs extra height.
  // Subtract the total loading bar height from the available vertical
  // space before computing the base size so previews stay square.
  const loadingExtra = showLoadings ? LOADING_BAR_HEIGHT : 0;
  const totalLoadingHeight = showLoadings ? numTracks * loadingExtra : 0;

  // Derive all track sizes from the short side so previews stay square.
  // Available = shortSide - gaps between tracks.
  // baseSize is the unit; corner = 1.0×base, mid-edge = 0.8×base, etc.
  const effectiveHeight = containerHeight - totalLoadingHeight;
  const shortSide = Math.min(containerWidth, effectiveHeight);
  const availableForCells = shortSide - (numTracks - 1) * GAP;
  const baseSize = Math.min(MAX_SIZE, availableForCells / ratioSum) * scale;

  const colTemplate = ratios.map((r) => `${Math.round(baseSize * r)}px`).join(' ');
  const rowTemplate = ratios.map((r) => `${Math.round(baseSize * r + loadingExtra)}px`).join(' ');
  const previewSize = baseSize;

  // Per-preview sizes: each preview's size depends on its edge position.
  // Preview i sits at edge position j = i % k, sized by sizeRatio(j, k).
  const sizes: number[] = [];
  for (let i = 0; i < previewCount; i++) {
    sizes.push(baseSize * sizeRatio(i % k, k));
  }

  const padX = previewSize / 2;
  const padY = previewSize / 2;

  return {
    gridTemplateColumns: colTemplate,
    gridTemplateRows: rowTemplate,
    sizes,
    previewSize,
    padX,
    padY,
  };
}
