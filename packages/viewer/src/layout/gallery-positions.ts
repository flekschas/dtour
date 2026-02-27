export type PreviewPosition = {
  /** Center x relative to container (px) */
  x: number;
  /** Center y relative to container (px) */
  y: number;
  /** Side length of the square preview (CSS px) */
  size: number;
  /** Angle in radians from center (standard atan2) */
  angle: number;
};

/** Gap between adjacent previews (CSS px). */
const GAP = 16;
/** Maximum base size (CSS px). */
const MAX_BASE = 320;
/** Size ratios: corner = 1.0, one from corner = 0.8, interior = 0.6. */
const RATIOS = [1.0, 0.8, 0.6] as const;

/**
 * Compute the size ratio for a preview at position `j` on an edge of `k` emitted points.
 *
 * distFromCorner = min(j, k − j)  →  0 = big (1.0×), 1 = med (0.8×), 2+ = small (0.6×)
 */
function sizeRatio(j: number, k: number): number {
  const dist = Math.min(j, k - j);
  return RATIOS[Math.min(dist, RATIOS.length - 1)] ?? RATIOS[0];
}

/**
 * Compute positions for preview canvases around the container perimeter.
 *
 * Layout: k points emitted per edge (corners at j=0), walking clockwise
 * from top-left. Each edge visually has k+1 positions (start + interior +
 * next corner), but the far corner belongs to the next edge — so corners
 * are never duplicated.
 *
 *   4 previews (k=1):  TL  TR  BR  BL
 *   8 previews (k=2):  TL TC | TR RC | BR BC | BL LC
 *  12 previews (k=3):  TL ·· | TR ·· | BR ·· | BL ··
 *  16 previews (k=4):  TL ··· | TR ··· | BR ··· | BL ···
 *
 * Sizing is responsive: baseSize is computed from the shorter container
 * dimension so previews + gaps fit exactly. Each preview's actual size
 * is `baseSize × ratio(distFromCorner)`.
 */
export const computeGalleryPositions = (
  containerWidth: number,
  containerHeight: number,
  previewCount: number,
): { positions: PreviewPosition[]; previewSize: number; padX: number; padY: number } => {
  const k = Math.max(1, previewCount / 4);
  // Subtract the gallery padding from the container size via `- 2 * GAP`
  const shortSide = Math.min(containerWidth, containerHeight) - 2 * GAP;

  // Each edge visually shows k+1 previews (k emitted + the next corner).
  // The ratio sum along the short-side edge determines baseSize.
  // There are k gaps between k+1 visual previews on the short edge.
  let ratioSum = 0;
  for (let j = 0; j <= k; j++) {
    ratioSum += sizeRatio(j, k);
  }
  const baseSize = Math.min(MAX_BASE, (shortSide - k * GAP) / ratioSum);

  // Largest preview size (for padding computation)
  const maxSize = baseSize * RATIOS[0];

  // Padding: half the largest preview so they don't overflow
  const padX = maxSize / 2;
  const padY = maxSize / 2;

  const cx = containerWidth / 2;
  const cy = containerHeight / 2;

  /**
   * Lay out k emitted previews along one edge.
   *
   * `startX/Y` and `endX/Y` are the centers of the first and last visual
   * preview on that edge (the corner that opens and the one that closes it).
   * The closing corner is NOT emitted — it belongs to the next edge.
   *
   * Centers are placed by accumulating half-sizes + gaps so previews tile
   * without overlapping.
   */
  function edgePositions(
    startX: number,
    startY: number,
    endX: number,
    endY: number,
  ): PreviewPosition[] {
    // Compute sizes for all k+1 visual slots (including the closing corner)
    const sizes: number[] = [];
    for (let j = 0; j <= k; j++) {
      sizes.push(baseSize * sizeRatio(j, k));
    }

    // Total span consumed by previews + gaps along the edge
    let usedSpan = 0;
    for (let j = 0; j <= k; j++) usedSpan += sizes[j]!;
    usedSpan += k * GAP;

    // Edge length
    const edgeLen = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
    // Extra space distributed as additional gap
    const extraGap = k > 0 ? Math.max(0, edgeLen - usedSpan) / k : 0;
    const totalGap = GAP + extraGap;

    // Direction unit vector
    const dx = edgeLen > 0 ? (endX - startX) / edgeLen : 0;
    const dy = edgeLen > 0 ? (endY - startY) / edgeLen : 0;

    // Place centers by accumulation
    const result: PreviewPosition[] = [];
    let cursor = sizes[0]! / 2; // first center at half its size from start
    for (let j = 0; j < k; j++) {
      const s = sizes[j]!;
      const sNext = sizes[j + 1]!;
      const px = startX + dx * cursor;
      const py = startY + dy * cursor;
      const angle = Math.atan2(py - cy, px - cx);
      result.push({ x: px, y: py, size: s, angle });
      // Advance: half current + gap + half next
      cursor += s / 2 + totalGap + sNext / 2;
    }
    return result;
  }

  // Edge corner centers (inset by half the corner preview size + a small gap)
  const inset = maxSize / 2 + GAP / 2;
  const left = inset;
  const right = containerWidth - inset;
  const top = inset;
  const bottom = containerHeight - inset;

  const positions: PreviewPosition[] = [
    ...edgePositions(left, top, right, top), // top: left → right
    ...edgePositions(right, top, right, bottom), // right: top → bottom
    ...edgePositions(right, bottom, left, bottom), // bottom: right → left
    ...edgePositions(left, bottom, left, top), // left: bottom → top
  ];

  return { positions, previewSize: maxSize, padX, padY };
};
