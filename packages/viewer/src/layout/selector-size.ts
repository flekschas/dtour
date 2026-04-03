import { GAP, LOADING_BAR_HEIGHT, MAX_SIZE, sizeRatio } from './gallery-positions.ts';

const MIN_SIZE = 80;

/**
 * CircularSlider draws its ring at `size * RING_RATIO` from centre.
 * The selector diameter we return must account for this ratio so
 * the *ring* (not the SVG bounding-box) respects the padding.
 */
const RING_RATIO = 0.4;

/**
 * Compute the largest selector diameter (px) whose visible ring does
 * not overlap any preview in the gallery perimeter layout.
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
  isToolbarVisible: boolean,
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

  const k = Math.max(1, previewCount / 4);
  const numTracks = k + 1;
  const loadingExtra = showLoadings ? LOADING_BAR_HEIGHT : 0;

  // Must match Gallery CSS: left-4, right-4, top/bottom = verticalInset.
  // When toolbar is visible overlayOffsetY = toolbarHeight/2 = 20 shifts
  // the wrapper down, so we bump vertical insets to keep 16px visual gap.
  const gridLeft = 16;
  const toolbarOffset = isToolbarVisible ? 20 : 0; // toolbarHeight / 2
  const verticalInset = 16 + toolbarOffset;
  const gridW = containerWidth - 2 * gridLeft;
  const gridH = containerHeight - 2 * verticalInset;

  // Track sizes (same logic as computeGallerySizes)
  const ratios: number[] = [];
  let ratioSum = 0;
  for (let j = 0; j <= k; j++) {
    const r = sizeRatio(j, k);
    ratios.push(r);
    ratioSum += r;
  }
  const totalLoadingHeight = numTracks * loadingExtra;
  const effectiveHeight = gridH - totalLoadingHeight;
  const shortSide = Math.min(gridW, effectiveHeight);
  const availableForCells = shortSide - (numTracks - 1) * GAP;
  const baseSize = Math.min(MAX_SIZE, availableForCells / ratioSum) * scale;
  const trackSizes = ratios.map((r) => baseSize * r);
  const rowTrackSizes = ratios.map((r) => baseSize * r + loadingExtra);
  const trackTotal = trackSizes.reduce((a, b) => a + b, 0);
  const rowTrackTotal = rowTrackSizes.reduce((a, b) => a + b, 0);

  // Effective gaps: CSS justify-content/align-content: space-between
  // distributes remaining free space equally between inter-track gutters.
  const freeX = gridW - trackTotal - (numTracks - 1) * GAP;
  const freeY = gridH - rowTrackTotal - (numTracks - 1) * GAP;
  const effGapX = GAP + (numTracks > 1 ? Math.max(0, freeX) / (numTracks - 1) : 0);
  const effGapY = GAP + (numTracks > 1 ? Math.max(0, freeY) / (numTracks - 1) : 0);

  // Track start positions relative to grid origin
  const colStarts: number[] = [0];
  const rowStarts: number[] = [0];
  for (let j = 1; j <= k; j++) {
    colStarts.push(colStarts[j - 1]! + trackSizes[j - 1]! + effGapX);
    rowStarts.push(rowStarts[j - 1]! + rowTrackSizes[j - 1]! + effGapY);
  }

  // Selector centre in the overlay-wrapper coordinate system
  const cx = containerWidth / 2;
  const cy = containerHeight / 2;

  // Container edge constraint: the wrapper is shifted by toolbarOffset,
  // so the visible bottom edge is closer to centre than the top edge.
  let minDist = Math.min(containerWidth / 2, containerHeight / 2 - toolbarOffset);

  for (let i = 0; i < previewCount; i++) {
    // Grid cell — same edge-walk order as Gallery
    let col: number;
    let row: number;
    if (i < k) {
      row = 0;
      col = i;
    } else if (i < 2 * k) {
      row = i - k;
      col = k;
    } else if (i < 3 * k) {
      row = k;
      col = 3 * k - i;
    } else {
      row = 4 * k - i;
      col = 0;
    }

    const cellX = gridLeft + colStarts[col]!;
    const cellY = verticalInset + rowStarts[row]!;
    const cellW = trackSizes[col]!;
    const cellH = rowTrackSizes[row]!;
    const size = baseSize * sizeRatio(i % k, k);

    // Alignment within cell (matches Gallery flex alignment)
    let px: number;
    if (col === 0) px = cellX;
    else if (col === k) px = cellX + cellW - size;
    else px = cellX + (cellW - size) / 2;

    // Vertical: the cell now includes loading bar height.
    // Bottom-edge previews have the bar above (flex-col-reverse),
    // so the preview canvas sits at the bottom of the cell.
    let py: number;
    if (row === 0) py = cellY;
    else if (row === k) py = cellY + cellH - size;
    else py = cellY + (cellH - size) / 2;

    // Bounding box includes loading bar
    const boxH = size + loadingExtra;
    const boxY = row === k ? py - loadingExtra : py;

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
