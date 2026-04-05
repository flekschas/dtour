/**
 * Compute the first principal component of the residual orthogonal complement
 * of a p×2 column-major basis. Returns a p-element unit vector orthogonal to
 * both basis columns — the "most informative direction you can't currently see."
 *
 * Algorithm:
 * 1. For each point, project out the current 2D plane (subtract projection onto basis cols)
 * 2. Build the d×d covariance matrix of the residuals
 * 3. Power iteration for the top eigenvector (d is small, ≤100 — 15 iterations suffice)
 */

/**
 * @param basis   p×2 column-major Float32Array [x0..xp-1, y0..yp-1]
 * @param data    column-major N×p: p contiguous columns of N floats each
 * @param mins    per-dimension minimums (length p)
 * @param ranges  per-dimension ranges (length p)
 * @param dims    number of dimensions (p)
 * @param numPoints number of data points (N)
 * @returns       p-element Float32Array, unit vector orthogonal to basis columns
 */
export const computeResidualPC = (
  basis: Float32Array,
  data: Float32Array,
  mins: Float32Array,
  ranges: Float32Array,
  dims: number,
  numPoints: number,
): Float32Array => {
  // Normalize basis columns (they should already be orthonormal, but be safe)
  const bx = new Float64Array(dims);
  const by = new Float64Array(dims);
  for (let d = 0; d < dims; d++) {
    bx[d] = basis[d]!;
    by[d] = basis[dims + d]!;
  }

  // Build d×d covariance of residuals in a single pass.
  // For each point, compute the normalized value, subtract projection onto basis,
  // and accumulate into the covariance matrix.
  // cov[i][j] = (1/N) * Σ_k residual_i(k) * residual_j(k)
  const cov = new Float64Array(dims * dims);

  // Pre-allocate residual scratch
  const res = new Float64Array(dims);

  for (let k = 0; k < numPoints; k++) {
    // Compute normalized values and project onto basis
    let dotX = 0;
    let dotY = 0;
    for (let d = 0; d < dims; d++) {
      const range = ranges[d]!;
      const norm = range > 1e-12 ? (data[d * numPoints + k]! - mins[d]!) / range - 0.5 : 0;
      res[d] = norm;
      dotX += norm * bx[d]!;
      dotY += norm * by[d]!;
    }

    // Subtract projection onto 2D plane
    for (let d = 0; d < dims; d++) {
      res[d] = res[d]! - (dotX * bx[d]! + dotY * by[d]!);
    }

    // Accumulate outer product into covariance (symmetric, but fill all for simplicity)
    for (let i = 0; i < dims; i++) {
      const ri = res[i]!;
      for (let j = i; j < dims; j++) {
        const val = ri * res[j]!;
        cov[i * dims + j] = cov[i * dims + j]! + val;
        if (i !== j) cov[j * dims + i] = cov[j * dims + i]! + val;
      }
    }
  }

  // Power iteration to find top eigenvector of cov
  let vec = new Float64Array(dims);
  // Initialize with a vector that has energy in all dimensions
  for (let d = 0; d < dims; d++) vec[d] = 1.0;
  // Orthogonalize against basis columns to stay in the complement
  orthogonalizeAgainstBasis(vec, bx, by, dims);
  normalizeVec(vec, dims);

  let dst = new Float64Array(dims);
  for (let iter = 0; iter < 15; iter++) {
    // Matrix-vector multiply: dst = cov * vec
    for (let i = 0; i < dims; i++) {
      let sum = 0;
      for (let j = 0; j < dims; j++) {
        sum += cov[i * dims + j]! * vec[j]!;
      }
      dst[i] = sum;
    }
    // Orthogonalize against basis to prevent drift back into the 2D plane
    orthogonalizeAgainstBasis(dst, bx, by, dims);
    normalizeVec(dst, dims);

    // Ping-pong: swap buffers so vec always holds the latest result
    const prev = vec;
    vec = dst;
    dst = prev;
  }

  // Convert to Float32Array
  const result = new Float32Array(dims);
  for (let d = 0; d < dims; d++) result[d] = vec[d]!;
  return result;
};

const orthogonalizeAgainstBasis = (
  v: Float64Array,
  bx: Float64Array,
  by: Float64Array,
  dims: number,
): void => {
  let dotX = 0;
  let dotY = 0;
  for (let d = 0; d < dims; d++) {
    dotX += v[d]! * bx[d]!;
    dotY += v[d]! * by[d]!;
  }
  for (let d = 0; d < dims; d++) {
    v[d] = v[d]! - (dotX * bx[d]! + dotY * by[d]!);
  }
};

const normalizeVec = (v: Float64Array, dims: number): void => {
  let norm = 0;
  for (let d = 0; d < dims; d++) norm += v[d]! * v[d]!;
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let d = 0; d < dims; d++) v[d] = v[d]! / norm;
  }
};
