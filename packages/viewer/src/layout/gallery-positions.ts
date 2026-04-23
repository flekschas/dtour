/** Gap between adjacent previews (CSS px). */
export const GAP = 32;
/** Maximum preview size (CSS px). */
export const MAX_SIZE = 320;
/** Height of the loading bar below/above each preview (CSS px). */
export const LOADING_BAR_HEIGHT = 18;
/** Space between previews and the container edges. */
export const PREVIEW_SPACING = 8;
/**
 * Per-edge-count ratio arrays.
 *   k=1 (4 previews)  → [1]             all same
 *   k=2 (8 previews)  → [1, 0.5]        corners 1, middle 0.5
 *   k=3 (12 previews) → [1, 0.75]       corners 1, mid 0.75
 *   k=4 (16 previews) → [1, 0.75, 0.5]  corners 1, near-corner 0.75, middle 0.5
 *   k=5 (6-col grids) → [1, 0.75, 0.5]  same as k=4
 */
const RATIOS_BY_K: Record<number, readonly number[]> = {
  1: [1],
  2: [1, 0.5],
  3: [1, 0.75],
  4: [1, 0.75, 0.5],
  5: [1, 0.75, 0.5],
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

// ---------------------------------------------------------------------------
// Layout: grid dimensions + per-item positions for any previewCount 2–16
// ---------------------------------------------------------------------------

export type LayoutPosition = { col: number; row: number };

export type LayoutInfo = {
  cols: number;
  rows: number;
  positions: LayoutPosition[];
};

/**
 * Grid spec lookup: [cols, rows, hasLeft, topCount?].
 * `hasLeft` indicates items on the left edge (full perimeter vs U-shape).
 * `topCount` defaults to `cols`; when smaller, items are spread across the
 * row with evenly-spaced column indices (used by n=14 to skip the center).
 */
const GRID_SPEC: Record<number, readonly [number, number, boolean, number?]> = {
  // n=2 and n=3 are special-cased in computeLayout
  4: [2, 2, false],
  5: [2, 3, false],
  6: [2, 3, true],
  7: [3, 3, false],
  8: [3, 3, true],
  9: [4, 3, false],
  10: [4, 3, true],
  11: [5, 3, false],
  12: [4, 4, true],
  13: [5, 5, false],
  14: [5, 5, true, 4],
  15: [6, 5, false],
  16: [5, 5, true],
};

/**
 * Spread `count` items evenly across `total` positions (0-indexed),
 * always including position 0 and position `total - 1`.
 */
function spreadIndices(count: number, total: number): number[] {
  if (count >= total) {
    return Array.from({ length: total }, (_, i) => i);
  }
  const indices: number[] = [];
  for (let i = 0; i < count; i++) {
    indices.push(Math.round((i * (total - 1)) / (count - 1)));
  }
  return indices;
}

/**
 * Compute grid dimensions and per-item (col, row) positions for any
 * preview count 2–16.  Items walk clockwise: top → right → bottom → left.
 */
export function computeLayout(n: number): LayoutInfo {
  // Special cases
  if (n <= 1) return { cols: 1, rows: 1, positions: [{ col: 0, row: 0 }] };
  if (n === 2) {
    return {
      cols: 2,
      rows: 2,
      positions: [
        { col: 0, row: 0 },
        { col: 1, row: 1 },
      ],
    };
  }
  if (n === 3) {
    return {
      cols: 2,
      rows: 3,
      positions: [
        { col: 0, row: 0 },
        { col: 1, row: 1 },
        { col: 0, row: 2 },
      ],
    };
  }

  const spec = GRID_SPEC[n];
  if (!spec) return { cols: 1, rows: 1, positions: [{ col: 0, row: 0 }] };

  const [cols, rows, hasLeft, topCountOverride] = spec;
  const topCount = topCountOverride ?? cols;
  const positions: LayoutPosition[] = [];

  // Top row: spread topCount items across cols columns (left to right)
  const topCols = spreadIndices(topCount, cols);
  for (const c of topCols) positions.push({ col: c, row: 0 });

  // Right column: interior rows top to bottom
  for (let r = 1; r < rows - 1; r++) positions.push({ col: cols - 1, row: r });

  // Bottom row: spread items across cols columns (right to left)
  const bottomCols = [...topCols].reverse();
  for (const c of bottomCols) positions.push({ col: c, row: rows - 1 });

  // Left column: interior rows bottom to top (perimeter only)
  if (hasLeft) {
    for (let r = rows - 2; r >= 1; r--) positions.push({ col: 0, row: r });
  }

  return { cols, rows, positions };
}

/**
 * Compute the circular slider start angle (SVG degrees) for n previews.
 *
 * The right-center item is anchored at SVG 0° (3 o'clock) and items are
 * evenly spaced at 360/n degrees.  For n=4,8,12,16 this returns −135°,
 * matching the previous hard-coded constant.
 */
export function computeStartAngle(n: number): number {
  if (n <= 1) return -135;
  const { rows, positions } = computeLayout(n);
  const topCount = positions.filter((p) => p.row === 0).length;
  const rightEdgeLen = Math.max(0, rows - 2);
  // Index (possibly fractional) of the right-center item in the walk order
  const rightCenterIdx = rightEdgeLen > 0 ? topCount + (rightEdgeLen - 1) / 2 : topCount - 0.5;
  return -(rightCenterIdx / n) * 360;
}

// ---------------------------------------------------------------------------
// Gallery sizing
// ---------------------------------------------------------------------------

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
 * Column ratios are derived from `sizeRatio(c, cols-1)` — tapering from
 * edges to centre.  Row ratios follow the same pattern with `rows-1`.
 * Each preview is sized as `min(colRatio, rowRatio) × baseSize` so it
 * stays square and fits its cell.
 *
 * For n=4,8,12,16 (square grids) this produces results identical to the
 * previous k-based computation.
 */
export function computeGallerySizes(
  containerWidth: number,
  containerHeight: number,
  previewCount: number,
  scale = 1,
  showLoadings = false,
): GallerySizes {
  const { cols, rows, positions } = computeLayout(previewCount);

  // Column ratios
  const colRatios: number[] = [];
  let colRatioSum = 0;
  for (let c = 0; c < cols; c++) {
    const r = sizeRatio(c, cols - 1);
    colRatios.push(r);
    colRatioSum += r;
  }

  // Row ratios
  const rowRatios: number[] = [];
  let rowRatioSum = 0;
  for (let r = 0; r < rows; r++) {
    const ratio = sizeRatio(r, rows - 1);
    rowRatios.push(ratio);
    rowRatioSum += ratio;
  }

  // Loading bar space
  const loadingExtra = showLoadings ? LOADING_BAR_HEIGHT : 0;
  const totalLoadingHeight = showLoadings ? rows * loadingExtra : 0;
  const effectiveHeight = containerHeight - totalLoadingHeight;

  // baseSize: constrained by both axes
  const baseSize =
    Math.min(
      MAX_SIZE,
      (containerWidth - Math.max(0, cols - 1) * GAP) / colRatioSum,
      (effectiveHeight - Math.max(0, rows - 1) * GAP) / rowRatioSum,
    ) * scale;

  const colTemplate = colRatios.map((r) => `${Math.round(baseSize * r)}px`).join(' ');
  const rowTemplate = rowRatios
    .map((r) => `${Math.round(baseSize * r + loadingExtra)}px`)
    .join(' ');

  // Per-preview sizes: min(colRatio, rowRatio) × baseSize
  const sizes: number[] = [];
  for (let i = 0; i < previewCount; i++) {
    const pos = positions[i];
    if (pos) {
      sizes.push(baseSize * Math.min(colRatios[pos.col]!, rowRatios[pos.row]!));
    } else {
      sizes.push(baseSize);
    }
  }

  const previewSize = baseSize;
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
