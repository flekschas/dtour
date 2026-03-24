/**
 * Jacobi eigendecomposition of a symmetric d×d matrix.
 *
 * Iteratively applies Givens rotations to zero off-diagonal elements,
 * converging to a diagonal matrix of eigenvalues with the accumulated
 * rotations forming the eigenvector matrix.
 *
 * @param matrix  d×d row-major symmetric matrix (modified in place)
 * @param d       dimension
 * @returns eigenvalues (descending) and corresponding eigenvectors
 */
export const jacobiEigen = (
  matrix: Float32Array,
  d: number,
): { eigenvalues: Float32Array; eigenvectors: Float32Array[] } => {
  const A = matrix;

  // Eigenvector matrix, starts as identity
  const V = new Float32Array(d * d);
  for (let i = 0; i < d; i++) V[i * d + i] = 1;

  const MAX_SWEEPS = 50;
  const EPS = 1e-10;

  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    // Sum of squared off-diagonal elements
    let offDiag = 0;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        offDiag += A[p * d + q]! * A[p * d + q]!;
      }
    }
    if (offDiag < EPS) break;

    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = A[p * d + q]!;
        if (Math.abs(apq) < EPS * 0.01) continue;

        const app = A[p * d + p]!;
        const aqq = A[q * d + q]!;
        const tau = (aqq - app) / (2 * apq);
        const t =
          Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Update diagonal
        A[p * d + p] = app - t * apq;
        A[q * d + q] = aqq + t * apq;
        A[p * d + q] = 0;
        A[q * d + p] = 0;

        // Rotate rows/columns p and q
        for (let r = 0; r < d; r++) {
          if (r === p || r === q) continue;
          const arp = A[r * d + p]!;
          const arq = A[r * d + q]!;
          A[r * d + p] = c * arp - s * arq;
          A[p * d + r] = c * arp - s * arq;
          A[r * d + q] = s * arp + c * arq;
          A[q * d + r] = s * arp + c * arq;
        }

        // Accumulate rotation into eigenvector matrix
        for (let r = 0; r < d; r++) {
          const vrp = V[r * d + p]!;
          const vrq = V[r * d + q]!;
          V[r * d + p] = c * vrp - s * vrq;
          V[r * d + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  // Sort eigenvalues descending
  const eigenvalues = new Float32Array(d);
  const indices = Array.from({ length: d }, (_, i) => i);
  for (let i = 0; i < d; i++) eigenvalues[i] = A[i * d + i]!;
  indices.sort((a, b) => eigenvalues[b]! - eigenvalues[a]!);

  const sorted = new Float32Array(d);
  const eigenvectors: Float32Array[] = [];
  for (let i = 0; i < d; i++) {
    const idx = indices[i]!;
    sorted[i] = eigenvalues[idx]!;
    // Column idx of V
    const vec = new Float32Array(d);
    for (let r = 0; r < d; r++) vec[r] = V[r * d + idx]!;
    eigenvectors.push(vec);
  }

  return { eigenvalues: sorted, eigenvectors };
};
