import {
  GAP,
  LOADING_BAR_HEIGHT,
  MAX_SIZE,
  computeLayout,
  sizeRatio,
} from './gallery-positions.ts';

const MIN_SIZE = 80;

/**
 * CircularSlider draws its ring at `size * RING_RATIO` from centre.
 * The selector diameter we return must account for this ratio so
 * the *ring* (not the SVG bounding-box) respects the padding.
 */
const RING_RATIO = 0.4;

/**
 * Compute the largest selector diameter (px) whose visible ring does
 * not overlap any preview in the gallery layout.
 *
 * For every preview bounding-box we compute the Euclidean distance from
 * the container centre to the nearest point on that box.  The container
 * edges (accounting for overlayOffsetY) are an additional constraint.
 * The returned diameter satisfies `size * RING_RATIO + padding ≤ minDist`,
 * i.e. `size = (minDist − padding) / RING_RATIO`, clamped to
 * {@link MIN_SIZE}.
 */
export function computeSelectorSize(
  containerWidth: number,
  containerHeight: number,
  previewCount: number,
  toolbarHeight: number,
  padding: number,
  scale = 1,
  /** Number of radial metric tracks rendered outside the ring. */
  metricTrackCount = 0,
  /** Whether loading bars are shown below/above previews. */
  showLoadings = false,
): number {
  if (previewCount === 0 || containerWidth <= 0 || containerHeight <= 0) {
    return Math.max(MIN_SIZE, Math.min(containerWidth, containerHeight));
  }

  const { cols, rows, positions } = computeLayout(previewCount);
  const loadingExtra = showLoadings ? LOADING_BAR_HEIGHT : 0;

  // Must match Gallery CSS: left-4, right-4, top/bottom = verticalInset.
  // The overlay wrapper shifts down by toolbarHeight/2, so we bump
  // vertical insets to keep a 16px visual gap below the toolbar.
  const gridLeft = 16;
  const toolbarOffset = toolbarHeight / 2;
  const verticalInset = 16 + toolbarOffset;
  const gridW = containerWidth - 2 * gridLeft;
  const gridH = containerHeight - 2 * verticalInset;

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

  const totalLoadingHeight = rows * loadingExtra;
  const effectiveHeight = gridH - totalLoadingHeight;

  const baseSize =
    Math.min(
      MAX_SIZE,
      (gridW - Math.max(0, cols - 1) * GAP) / colRatioSum,
      (effectiveHeight - Math.max(0, rows - 1) * GAP) / rowRatioSum,
    ) * scale;

  const colTrackSizes = colRatios.map((r) => baseSize * r);
  const rowTrackSizes = rowRatios.map((r) => baseSize * r + loadingExtra);
  const colTrackTotal = colTrackSizes.reduce((a, b) => a + b, 0);
  const rowTrackTotal = rowTrackSizes.reduce((a, b) => a + b, 0);

  // Effective gaps: CSS justify-content/align-content: space-between
  // distributes remaining free space equally between inter-track gutters.
  const freeX = gridW - colTrackTotal - Math.max(0, cols - 1) * GAP;
  const freeY = gridH - rowTrackTotal - Math.max(0, rows - 1) * GAP;
  const effGapX = GAP + (cols > 1 ? Math.max(0, freeX) / (cols - 1) : 0);
  const effGapY = GAP + (rows > 1 ? Math.max(0, freeY) / (rows - 1) : 0);

  // Track start positions relative to grid origin
  const colStarts: number[] = [0];
  for (let c = 1; c < cols; c++) {
    colStarts.push(colStarts[c - 1]! + colTrackSizes[c - 1]! + effGapX);
  }
  const rowStarts: number[] = [0];
  for (let r = 1; r < rows; r++) {
    rowStarts.push(rowStarts[r - 1]! + rowTrackSizes[r - 1]! + effGapY);
  }

  // Selector centre in the overlay-wrapper coordinate system
  const cx = containerWidth / 2;
  const cy = containerHeight / 2;

  // Container edge constraint: the wrapper is shifted by toolbarOffset,
  // so the visible bottom edge is closer to centre than the top edge.
  let minDist = Math.min(containerWidth / 2, containerHeight / 2 - toolbarOffset);

  for (let i = 0; i < previewCount; i++) {
    const pos = positions[i];
    if (!pos) continue;
    const { col, row } = pos;

    const cellX = gridLeft + colStarts[col]!;
    const cellY = verticalInset + rowStarts[row]!;
    const cellW = colTrackSizes[col]!;
    const cellH = rowTrackSizes[row]!;
    const size = baseSize * Math.min(colRatios[col]!, rowRatios[row]!);

    // Alignment within cell (matches Gallery flex alignment)
    let px: number;
    if (col === 0) px = cellX;
    else if (col === cols - 1) px = cellX + cellW - size;
    else px = cellX + (cellW - size) / 2;

    // Vertical: the cell now includes loading bar height.
    // Bottom-edge previews have the bar above (flex-col-reverse),
    // so the preview canvas sits at the bottom of the cell.
    let py: number;
    if (row === 0) py = cellY;
    else if (row === rows - 1) py = cellY + cellH - size;
    else py = cellY + (cellH - size) / 2;

    // Bounding box includes loading bar
    const boxH = size + loadingExtra;
    const boxY = row === rows - 1 ? py - loadingExtra : py;

    // Euclidean distance from centre to nearest point on preview rect
    const dx = Math.max(0, px - cx, cx - (px + size));
    const dy = Math.max(0, boxY - cy, cy - (boxY + boxH));
    minDist = Math.min(minDist, Math.sqrt(dx * dx + dy * dy));
  }

  // The visible ring sits at size * RING_RATIO from centre. Metric tracks
  // extend outward from the ring; each track is ~18px (16px bar + 2px gap).
  const metricThickness = metricTrackCount * 18;

  // Outermost point = size * RING_RATIO + metricThickness, so:
  // size * RING_RATIO + metricThickness + padding ≤ minDist
  // size = (minDist - padding - metricThickness) / RING_RATIO
  return Math.max(MIN_SIZE, (minDist - padding - metricThickness) / RING_RATIO);
}
