/**
 * Create default "little tour" view matrices that cycle through consecutive
 * dimension pairs: (d0,d1), (d1,d2), ..., wrapping back to the start.
 *
 * Each view is a p×2 column-major Float32Array:
 *   [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * When `activeIndices` is provided, only those dimensions get non-zero
 * basis weights — inactive dimensions contribute zero to the projection.
 *
 * @param dims          - total number of dimensions (p)
 * @param count         - number of views to generate (defaults to active dim count)
 * @param activeIndices - sorted array of active dimension indices (defaults to all)
 * @returns array of view matrices
 */
export const createDefaultViews = (
  dims: number,
  count?: number,
  activeIndices?: number[],
): Float32Array[] => {
  const indices = activeIndices ?? Array.from({ length: dims }, (_, i) => i);
  const activeDims = indices.length;
  const n = count ?? activeDims;
  const views: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const basis = new Float32Array(dims * 2);
    const idx = Math.floor((i / n) * activeDims);
    basis[indices[idx]!] = 1; // active dim → x
    basis[dims + indices[(idx + 1) % activeDims]!] = 1; // next active dim → y
    views.push(basis);
  }
  return views;
};
