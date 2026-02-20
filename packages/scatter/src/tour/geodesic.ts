import { svd2x2 } from './svd2x2.ts';

/**
 * Interpolate between two p×2 basis matrices via linear blend + Gram-Schmidt
 * orthonormalization (NLERP-style).
 *
 * Unlike SVD-aligned geodesic interpolation, this guarantees that the output
 * at t=0 is exactly Fa and at t=1 is exactly Fz — eliminating rotation jumps
 * at keyframe boundaries when stitching segments together.
 *
 * All basis matrices are column-major: [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * @param out - pre-allocated output (2p floats, column-major)
 * @param Fa  - start basis (2p floats, column-major)
 * @param Fz  - end basis (2p floats, column-major)
 * @param p   - number of dimensions
 * @param t   - interpolation parameter [0, 1]
 */
export const interpolateBases = (
  out: Float32Array,
  Fa: Float32Array,
  Fz: Float32Array,
  p: number,
  t: number,
): Float32Array => {
  const s = 1 - t;

  // 1. Linear blend
  for (let k = 0; k < p * 2; k++) {
    out[k] = s * Fa[k]! + t * Fz[k]!;
  }

  // 2. Gram-Schmidt orthonormalization
  // Column 0: out[0..p-1], Column 1: out[p..2p-1]

  // Normalize column 0
  let norm0 = 0;
  for (let k = 0; k < p; k++) {
    norm0 += out[k]! * out[k]!;
  }
  norm0 = Math.sqrt(norm0);
  if (norm0 > 1e-10) {
    for (let k = 0; k < p; k++) {
      out[k] = out[k]! / norm0;
    }
  }

  // Subtract projection of column 1 onto column 0
  let dot = 0;
  for (let k = 0; k < p; k++) {
    dot += out[k]! * out[p + k]!;
  }
  for (let k = 0; k < p; k++) {
    out[p + k] = out[p + k]! - dot * out[k]!;
  }

  // Normalize column 1
  let norm1 = 0;
  for (let k = 0; k < p; k++) {
    norm1 += out[p + k]! * out[p + k]!;
  }
  norm1 = Math.sqrt(norm1);
  if (norm1 > 1e-10) {
    for (let k = 0; k < p; k++) {
      out[p + k] = out[p + k]! / norm1;
    }
  }

  return out;
};

/**
 * Geodesic distance between two p×2 basis matrices.
 * Distance = sqrt(tau_0^2 + tau_1^2) where tau_i are principal angles.
 */
export const geodesicDistance = (Fa: Float32Array, Fz: Float32Array, p: number): number => {
  // Compute A = Fa^T * Fz
  const A = new Float32Array(4);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += Fa[i * p + k]! * Fz[j * p + k]!;
      }
      A[i * 2 + j] = sum;
    }
  }

  const { S } = svd2x2(A);
  const tau0 = Math.acos(Math.max(-1, Math.min(1, S[0]!)));
  const tau1 = Math.acos(Math.max(-1, Math.min(1, S[1]!)));

  return Math.sqrt(tau0 * tau0 + tau1 * tau1);
};
