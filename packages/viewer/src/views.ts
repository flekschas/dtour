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

/**
 * Create tour views from PCA eigenvectors.
 * Cycles through consecutive PC pairs: [PC1,PC2], [PC2,PC3], ..., wrapping.
 *
 * Each eigenvector becomes a column of the p×2 basis matrix. Eigenvectors
 * are assumed to be in the normalized space matching the projection shader.
 *
 * @param eigenvectors - sorted by descending eigenvalue, each of length pcaDims
 * @param totalDims    - total number of dimensions in the dataset (p)
 * @param pcaDims      - number of PCA dimensions (may be < totalDims if capped)
 * @param count        - number of views to generate (defaults to number of PCs)
 */
export const createPCAViews = (
  eigenvectors: Float32Array[],
  totalDims: number,
  pcaDims: number,
  count?: number,
): Float32Array[] => {
  const numPCs = eigenvectors.length;
  const n = count ?? numPCs;
  const views: Float32Array[] = [];

  for (let i = 0; i < n; i++) {
    const basis = new Float32Array(totalDims * 2);
    const pcX = i % numPCs;
    const pcY = (i + 1) % numPCs;

    const evX = eigenvectors[pcX]!;
    for (let d = 0; d < pcaDims; d++) {
      basis[d] = evX[d]!;
    }

    const evY = eigenvectors[pcY]!;
    for (let d = 0; d < pcaDims; d++) {
      basis[totalDims + d] = evY[d]!;
    }

    views.push(basis);
  }

  return views;
};
