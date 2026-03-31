/**
 * Convert tour position (arc-length parameterized) to visual position (equal spacing).
 *
 * In equal mode, keyframe `i` occupies visual range `[i/n, (i+1)/n]`.
 * We binary-search for the arc-length segment containing `tourPos`, compute
 * fractional progress within it, and map to the corresponding visual segment.
 */
export const tourToVisual = (tourPos: number, arcLengths: Float32Array): number => {
  const n = arcLengths.length - 1;
  if (n <= 0) return tourPos;
  const clamped = Math.max(0, Math.min(1, tourPos));

  // Binary search for the segment containing clamped
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arcLengths[mid + 1]! <= clamped) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const segStart = arcLengths[lo]!;
  const segEnd = arcLengths[lo + 1]!;
  const segLen = segEnd - segStart;
  const localT = segLen > 1e-10 ? (clamped - segStart) / segLen : 0;

  return (lo + localT) / n;
};

/**
 * Convert visual position (equal spacing) to tour position (arc-length).
 *
 * Visual position `v` maps to segment `floor(v * n)`, with local fraction
 * `frac(v * n)` interpolating within that arc-length segment.
 */
export const visualToTour = (visualPos: number, arcLengths: Float32Array): number => {
  const n = arcLengths.length - 1;
  if (n <= 0) return visualPos;
  const clamped = Math.max(0, Math.min(1, visualPos));

  const scaled = clamped * n;
  const segment = Math.min(Math.floor(scaled), n - 1);
  const localT = scaled - segment;

  const segStart = arcLengths[segment]!;
  const segEnd = arcLengths[segment + 1]!;
  return segStart + localT * (segEnd - segStart);
};
