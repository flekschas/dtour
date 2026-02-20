export type SVD2x2Result = {
  U: Float32Array; // 2x2 row-major [u00, u01, u10, u11]
  S: Float32Array; // 2 singular values [s0, s1], descending
  V: Float32Array; // 2x2 row-major [v00, v01, v10, v11]
};

const EPS = 1e-10;

/**
 * Analytical SVD of a 2x2 matrix.
 *
 * Input A is row-major: A = [[A[0], A[1]], [A[2], A[3]]].
 * Returns U, S, V such that A ≈ U * diag(S) * V^T.
 */
export const svd2x2 = (A: Float32Array): SVD2x2Result => {
  const a = A[0]!;
  const b = A[1]!;
  const c = A[2]!;
  const d = A[3]!;

  // A^T * A
  const ata00 = a * a + c * c;
  const ata01 = a * b + c * d;
  const ata11 = b * b + d * d;

  // Eigenvalues of A^T * A via quadratic formula
  const trace = ata00 + ata11;
  const det = ata00 * ata11 - ata01 * ata01;
  const disc = Math.sqrt(Math.max(0, trace * trace * 0.25 - det));

  const lambda1 = trace * 0.5 + disc;
  const lambda2 = trace * 0.5 - disc;

  const sigma1 = Math.sqrt(Math.max(0, lambda1));
  const sigma2 = Math.sqrt(Math.max(0, lambda2));

  const S = new Float32Array([sigma1, sigma2]);

  // Right singular vectors V (eigenvectors of A^T * A)
  const V = new Float32Array(4);
  if (Math.abs(ata01) > EPS) {
    const dx1 = lambda1 - ata11;
    const n1 = Math.sqrt(dx1 * dx1 + ata01 * ata01);
    V[0] = dx1 / n1;
    V[1] = ata01 / n1;

    const dx2 = lambda2 - ata11;
    const n2 = Math.sqrt(dx2 * dx2 + ata01 * ata01);
    V[2] = dx2 / n2;
    V[3] = ata01 / n2;
  } else {
    // Already diagonal
    V[0] = 1;
    V[1] = 0;
    V[2] = 0;
    V[3] = 1;
  }

  // Left singular vectors: U = A * V * S^{-1}
  // U is row-major: column 0 = [U[0], U[2]], column 1 = [U[1], U[3]]
  const U = new Float32Array(4);
  if (sigma1 > EPS) {
    U[0] = (a * V[0]! + b * V[1]!) / sigma1;
    U[2] = (c * V[0]! + d * V[1]!) / sigma1;
  }
  if (sigma2 > EPS) {
    U[1] = (a * V[2]! + b * V[3]!) / sigma2;
    U[3] = (c * V[2]! + d * V[3]!) / sigma2;
  }

  // Ensure U columns are orthonormal when a singular value is zero
  if (sigma1 <= EPS && sigma2 > EPS) {
    // Column 0 = perpendicular to column 1
    U[0] = -U[3]!;
    U[2] = U[1]!;
  } else if (sigma2 <= EPS && sigma1 > EPS) {
    // Column 1 = perpendicular to column 0
    U[1] = -U[2]!;
    U[3] = U[0]!;
  } else if (sigma1 <= EPS && sigma2 <= EPS) {
    U[0] = 1;
    U[1] = 0;
    U[2] = 0;
    U[3] = 1;
  }

  return { U, S, V };
};
