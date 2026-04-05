/**
 * Arcball rotation utilities — Shoemake-style trackball for 3D camera rotation.
 * All quaternions are [x, y, z, w] format. Rotation matrices are 3×3 column-major.
 */

export type Quat = [number, number, number, number];

export const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/** Project a screen point (in [-1,1]² NDC) onto the arcball sphere. */
export const projectToSphere = (x: number, y: number): [number, number, number] => {
  const r2 = x * x + y * y;
  if (r2 <= 1) {
    // On the sphere
    return [x, y, Math.sqrt(1 - r2)];
  }
  // Outside sphere — project onto edge
  const s = 1 / Math.sqrt(r2);
  return [x * s, y * s, 0];
};

/** Compute rotation quaternion that takes vector `from` to vector `to`. */
export const arcballQuat = (from: [number, number, number], to: [number, number, number]): Quat => {
  // Cross product = rotation axis, dot product = cos(angle)
  const cx = from[1] * to[2] - from[2] * to[1];
  const cy = from[2] * to[0] - from[0] * to[2];
  const cz = from[0] * to[1] - from[1] * to[0];
  const dot = from[0] * to[0] + from[1] * to[1] + from[2] * to[2];
  // q = (cross, 1 + dot), then normalize — avoids trig
  const w = 1 + dot;
  const len = Math.sqrt(cx * cx + cy * cy + cz * cz + w * w);
  if (len < 1e-12) return IDENTITY_QUAT;
  return [cx / len, cy / len, cz / len, w / len];
};

/** Multiply two quaternions: result = a * b. */
export const multiplyQuat = (a: Quat, b: Quat): Quat => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];

/** Normalize a quaternion in-place and return it. */
export const normalizeQuat = (q: Quat): Quat => {
  const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (len < 1e-12) return IDENTITY_QUAT;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
};

/** SLERP between two quaternions. t=0 → a, t=1 → b. */
export const slerp = (a: Quat, b: Quat, t: number): Quat => {
  // Ensure shortest path
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bx = b[0];
  let by = b[1];
  let bz = b[2];
  let bw = b[3];
  if (dot < 0) {
    dot = -dot;
    bx = -bx;
    by = -by;
    bz = -bz;
    bw = -bw;
  }

  if (dot > 0.9995) {
    // Very close — use linear interpolation to avoid division by zero
    return normalizeQuat([
      a[0] + t * (bx - a[0]),
      a[1] + t * (by - a[1]),
      a[2] + t * (bz - a[2]),
      a[3] + t * (bw - a[3]),
    ]);
  }

  const theta = Math.acos(dot);
  const sinTheta = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sinTheta;
  const wb = Math.sin(t * theta) / sinTheta;
  return [wa * a[0] + wb * bx, wa * a[1] + wb * by, wa * a[2] + wb * bz, wa * a[3] + wb * bw];
};

/** Convert quaternion to 3×3 column-major rotation matrix (9-element Float32Array). */
export const quatToMat3 = (q: Quat): Float32Array => {
  const [x, y, z, w] = q;
  const x2 = x + x;
  const y2 = y + y;
  const z2 = z + z;
  const xx = x * x2;
  const xy = x * y2;
  const xz = x * z2;
  const yy = y * y2;
  const yz = y * z2;
  const zz = z * z2;
  const wx = w * x2;
  const wy = w * y2;
  const wz = w * z2;

  // Column-major: col0, col1, col2
  return new Float32Array([
    1 - (yy + zz),
    xy + wz,
    xz - wy, // col0
    xy - wz,
    1 - (xx + zz),
    yz + wx, // col1
    xz + wy,
    yz - wx,
    1 - (xx + yy), // col2
  ]);
};

/** Check if a quaternion is approximately identity (no rotation). */
export const isIdentityQuat = (q: Quat, epsilon = 1e-4): boolean =>
  Math.abs(q[0]) < epsilon &&
  Math.abs(q[1]) < epsilon &&
  Math.abs(q[2]) < epsilon &&
  Math.abs(Math.abs(q[3]) - 1) < epsilon;
