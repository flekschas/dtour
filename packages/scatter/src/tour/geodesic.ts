import { svd2x2 } from './svd2x2.ts';

/**
 * Gram-Schmidt orthonormalization of a p×2 column-major matrix in-place.
 * Column 0: out[0..p-1], Column 1: out[p..2p-1]
 */
const gramSchmidt = (out: Float32Array, p: number): void => {
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
};

/**
 * Interpolate through four p×2 basis matrices using Catmull-Rom spline
 * + Gram-Schmidt orthonormalization.
 *
 * CR(t) = 0.5 * ((2*P1) + (-P0+P2)*t + (2*P0-5*P1+4*P2-P3)*t² + (-P0+3*P1-3*P2+P3)*t³)
 *
 * The spline passes exactly through P1 (at t=0) and P2 (at t=1), with
 * C1-continuous tangents derived from neighbors P0 and P3. After the
 * cubic blend, Gram-Schmidt restores orthonormality.
 *
 * All basis matrices are column-major: [x0, x1, ..., xp-1, y0, y1, ..., yp-1]
 *
 * @param out - pre-allocated output (2p floats, column-major)
 * @param P0  - previous basis (neighbor before start)
 * @param P1  - start basis
 * @param P2  - end basis
 * @param P3  - next basis (neighbor after end)
 * @param p   - number of dimensions
 * @param t   - interpolation parameter [0, 1]
 */
export const interpolateBases = (
  out: Float32Array,
  P0: Float32Array,
  P1: Float32Array,
  P2: Float32Array,
  P3: Float32Array,
  p: number,
  t: number,
): Float32Array => {
  const t2 = t * t;
  const t3 = t2 * t;

  // Catmull-Rom coefficients
  const c0 = 0.5 * (-t + 2 * t2 - t3);
  const c1 = 0.5 * (2 - 5 * t2 + 3 * t3);
  const c2 = 0.5 * (t + 4 * t2 - 3 * t3);
  const c3 = 0.5 * (-t2 + t3);

  for (let k = 0; k < p * 2; k++) {
    out[k] = c0 * P0[k]! + c1 * P1[k]! + c2 * P2[k]! + c3 * P3[k]!;
  }

  // Gram-Schmidt orthonormalization
  gramSchmidt(out, p);

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
