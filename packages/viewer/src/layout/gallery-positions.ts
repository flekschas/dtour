export type PreviewPosition = {
  /** Center x relative to container (px) */
  x: number;
  /** Center y relative to container (px) */
  y: number;
  /** Side length of the square preview (px) */
  size: number;
  /** Angle in radians (0 = 12 o'clock, clockwise) */
  angle: number;
};

/**
 * Compute positions for preview canvases placed in the corners of the container.
 * For 4 views: top-left, top-right, bottom-right, bottom-left.
 * For 8+ views: evenly distributed along the edges.
 *
 * Also returns the preview size and padding values so Dtour can
 * apply matching canvas insets.
 */
export const computeGalleryPositions = (
  containerWidth: number,
  containerHeight: number,
  previewCount: number,
): { positions: PreviewPosition[]; previewSize: number; padX: number; padY: number } => {
  // Preview size: ~12% of the shorter container dimension, clamped
  const shortSide = Math.min(containerWidth, containerHeight);
  const previewSize = Math.max(48, Math.min(120, shortSide * 0.12));

  // Padding: X = half preview height, padY = X, padX = 2X
  const X = previewSize / 2;
  const padY = X;
  const padX = 2 * X;

  // Inset centers: offset from container edges by padX/padY + half preview
  const margin = 4; // small gap from edge
  const left = padX + previewSize / 2 + margin;
  const right = containerWidth - padX - previewSize / 2 - margin;
  const top = padY + previewSize / 2 + margin;
  const bottom = containerHeight - padY - previewSize / 2 - margin;

  const positions: PreviewPosition[] = [];

  if (previewCount <= 4) {
    // Corner placement: TL, TR, BR, BL
    const corners: [number, number, number][] = [
      [left, top, -Math.PI / 4], // top-left, angle ~10:30
      [right, top, Math.PI / 4], // top-right
      [right, bottom, (3 * Math.PI) / 4], // bottom-right
      [left, bottom, (-3 * Math.PI) / 4], // bottom-left
    ];
    for (let i = 0; i < previewCount; i++) {
      const [x, y, angle] = corners[i]!;
      positions.push({ x, y, size: previewSize, angle });
    }
  } else {
    // Distribute evenly around the perimeter
    const cx = containerWidth / 2;
    const cy = containerHeight / 2;
    const rx = cx - padX - previewSize / 2 - margin;
    const ry = cy - padY - previewSize / 2 - margin;

    // Start from 10:30 position (-π/2 - π/12 ≈ -105°)
    const startAngle = -Math.PI / 2 - Math.PI / 12;
    for (let i = 0; i < previewCount; i++) {
      const angle = startAngle + (2 * Math.PI * i) / previewCount;
      positions.push({
        x: cx + rx * Math.cos(angle),
        y: cy + ry * Math.sin(angle),
        size: previewSize,
        angle,
      });
    }
  }

  return { positions, previewSize, padX, padY };
};
