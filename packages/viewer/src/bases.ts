/**
 * Create default "little tour" basis matrices that cycle through consecutive
 * dimension pairs: (d0,d1), (d1,d2), ..., wrapping back to the start.
 *
 * Each basis is a p×2 column-major Float32Array:
 *   [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * @param dims  - number of dimensions (p)
 * @param count - number of bases to generate (defaults to dims)
 * @returns array of basis matrices
 */
export const createDefaultBases = (dims: number, count?: number): Float32Array[] => {
  const n = count ?? dims;
  const bases: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const basis = new Float32Array(dims * 2);
    // Spread evenly across dimension pairs when count differs from dims
    const dimIndex = Math.floor((i / n) * dims);
    basis[dimIndex] = 1; // dim → x
    basis[dims + ((dimIndex + 1) % dims)] = 1; // next dim → y
    bases.push(basis);
  }
  return bases;
};
