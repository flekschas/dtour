/**
 * Create default "little tour" keyframes that cycle through consecutive
 * dimension pairs: (d0,d1), (d1,d2), ..., wrapping back to the start.
 *
 * Each keyframe is a p×2 column-major Float32Array:
 *   [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * When `activeIndices` is provided, only those dimensions get non-zero
 * basis weights — inactive dimensions contribute zero to the projection.
 *
 * @param dims          - total number of dimensions (p)
 * @param count         - number of keyframes to generate (defaults to active dim count)
 * @param activeIndices - sorted array of active dimension indices (defaults to all)
 * @returns array of keyframe bases
 */
export const createDefaultKeyframes = (
  dims: number,
  count?: number,
  activeIndices?: number[],
): Float32Array[] => {
  const indices = activeIndices ?? Array.from({ length: dims }, (_, i) => i);
  const activeDims = indices.length;
  const n = count ?? activeDims;
  const keyframes: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const basis = new Float32Array(dims * 2);
    const idx = Math.floor((i / n) * activeDims);
    basis[indices[idx]!] = 1; // active dim → x
    basis[dims + indices[(idx + 1) % activeDims]!] = 1; // next active dim → y
    keyframes.push(basis);
  }
  return keyframes;
};

/**
 * Names of the numeric columns a predefined tour of `nDims` dimensions
 * projects. Without names, the first `nDims` columns. With names, those names
 * when there are `nDims` unique names that all exist in the data. Otherwise,
 * null when `strict`, or the first `nDims` columns when not.
 */
export const resolveTourDimensions = (
  nDims: number,
  columnNames: string[],
  tourDimensions: string[] | null | undefined,
  strict: boolean,
): string[] | null => {
  const positional = columnNames.slice(0, nDims);
  if (!tourDimensions?.length) return positional;
  const valid =
    tourDimensions.length === nDims &&
    new Set(tourDimensions).size === nDims &&
    tourDimensions.every((name) => columnNames.includes(name));
  if (valid) return tourDimensions;
  return strict ? null : positional;
};

/**
 * Map a predefined tour onto the dataset's numeric columns (see
 * `resolveTourDimensions`). Returns the expanded keyframes and the columns
 * they project, or null when the tour doesn't fit the data.
 */
export const fitTour = (
  keyframes: Float32Array[],
  columnNames: string[],
  tourDimensions: string[] | null | undefined,
  strict: boolean,
): { keyframes: Float32Array[]; dimensions: string[] } | null => {
  const nDims = keyframes[0]!.length / 2;
  if (nDims > columnNames.length) return null;
  const dimensions = resolveTourDimensions(nDims, columnNames, tourDimensions, strict);
  if (!dimensions) return null;
  return {
    keyframes: expandBases(keyframes, dimensions, columnNames, columnNames.length),
    dimensions,
  };
};

/**
 * Expand basis matrices from a tour's dimension space to the full dataset
 * column space. Each input basis is `nDims × 2` (the tour's dim count);
 * each output basis is `totalDims × 2` with weights placed at the column
 * indices that correspond to `tourDimNames` in `allColumnNames`.
 *
 * Always returns new arrays.
 */
export const expandBases = (
  bases: Float32Array[],
  tourDimNames: string[],
  allColumnNames: string[],
  totalDims: number,
): Float32Array[] => {
  const nDims = tourDimNames.length;
  if (nDims === totalDims && tourDimNames.every((name, i) => name === allColumnNames[i])) {
    return bases.map((basis) => new Float32Array(basis));
  }

  // Map tour dim index → dataset column index
  const indexMap: number[] = [];
  for (const name of tourDimNames) {
    indexMap.push(allColumnNames.indexOf(name));
  }

  return bases.map((basis) => {
    const expanded = new Float32Array(totalDims * 2);
    for (let d = 0; d < nDims; d++) {
      const col = indexMap[d]!;
      if (col < 0) continue;
      expanded[col] = basis[d]!;
      expanded[totalDims + col] = basis[nDims + d]!;
    }
    return expanded;
  });
};

/**
 * Create tour keyframes from PCA eigenvectors.
 * Cycles through consecutive PC pairs: [PC1,PC2], [PC2,PC3], ..., wrapping.
 *
 * Each eigenvector becomes a column of the p×2 basis matrix. Eigenvectors
 * are assumed to be in the normalized space matching the projection shader.
 *
 * @param eigenvectors - sorted by descending eigenvalue, each of length pcaDims
 * @param totalDims    - total number of dimensions in the dataset (p)
 * @param pcaDims      - number of PCA dimensions (may be < totalDims if capped)
 * @param count        - number of keyframes to generate (defaults to number of PCs)
 */
export const createPCAKeyframes = (
  eigenvectors: Float32Array[],
  totalDims: number,
  pcaDims: number,
  count?: number,
): Float32Array[] => {
  const numPCs = eigenvectors.length;
  const n = count ?? numPCs;
  const keyframes: Float32Array[] = [];

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

    keyframes.push(basis);
  }

  return keyframes;
};
