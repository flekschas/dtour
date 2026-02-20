import { geodesicDistance, interpolateBases } from './geodesic.ts';

/**
 * Compute cumulative arc-length table for a cyclic sequence of basis matrices.
 * The last basis connects back to the first (closed loop).
 *
 * Returns a Float32Array of length bases.length + 1:
 *   [0, d01, d01+d12, ..., total]
 * Normalized to [0, 1] so the slider maps directly.
 *
 * @param bases - array of p×2 column-major basis matrices
 * @param p     - number of dimensions
 */
export const computeArcLengths = (bases: Float32Array[], p: number): Float32Array => {
  const n = bases.length;
  const cumulative = new Float32Array(n + 1);
  cumulative[0] = 0;

  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const d = geodesicDistance(bases[i]!, bases[next]!, p);
    cumulative[i + 1] = cumulative[i]! + d;
  }

  // Normalize to [0, 1]
  const total = cumulative[n]!;
  if (total > 1e-10) {
    for (let i = 1; i <= n; i++) {
      cumulative[i] = cumulative[i]! / total;
    }
  }

  return cumulative;
};

export type ResolvedPosition = {
  segment: number; // index of start basis
  localT: number; // interpolation parameter within segment [0, 1]
};

/**
 * Given a normalized position t in [0, 1] along the total arc,
 * find which segment it falls in and the local interpolation parameter.
 *
 * Uses binary search for O(log n) lookup.
 */
export const resolvePosition = (arcLengths: Float32Array, t: number): ResolvedPosition => {
  const n = arcLengths.length - 1; // number of segments
  const clamped = Math.max(0, Math.min(1, t));

  // Binary search for the segment
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

  return { segment: lo, localT };
};

/**
 * Convenience: resolve position and compute the interpolated basis in one call.
 */
export const interpolateAtPosition = (
  out: Float32Array,
  bases: Float32Array[],
  arcLengths: Float32Array,
  p: number,
  t: number,
): Float32Array => {
  const n = bases.length;
  const { segment, localT } = resolvePosition(arcLengths, t);
  const next = (segment + 1) % n;
  return interpolateBases(out, bases[segment]!, bases[next]!, p, localT);
};
