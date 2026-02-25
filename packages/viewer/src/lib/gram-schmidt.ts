/**
 * In-place Gram-Schmidt orthonormalization for a p×2 column-major basis.
 * Layout: [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * Normalizes column 0, orthogonalizes column 1 against column 0,
 * then normalizes column 1.
 */
export const gramSchmidt = (basis: Float32Array, dims: number): void => {
  // Normalize column 0
  let norm0 = 0;
  for (let i = 0; i < dims; i++) {
    norm0 += basis[i]! * basis[i]!;
  }
  norm0 = Math.sqrt(norm0);
  if (norm0 > 1e-12) {
    for (let i = 0; i < dims; i++) {
      basis[i]! /= norm0;
    }
  }

  // Orthogonalize column 1 against column 0
  let dot = 0;
  for (let i = 0; i < dims; i++) {
    dot += basis[i]! * basis[dims + i]!;
  }
  for (let i = 0; i < dims; i++) {
    basis[dims + i]! -= dot * basis[i]!;
  }

  // Normalize column 1
  let norm1 = 0;
  for (let i = 0; i < dims; i++) {
    norm1 += basis[dims + i]! * basis[dims + i]!;
  }
  norm1 = Math.sqrt(norm1);
  if (norm1 > 1e-12) {
    for (let i = 0; i < dims; i++) {
      basis[dims + i]! /= norm1;
    }
  }
};
