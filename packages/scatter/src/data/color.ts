/**
 * Encode categorical indices into packed RGBA8 u32 values using a palette.
 * Each u32 is packed as 0xAABBGGRR (little-endian RGBA).
 */
export const encodeCategoricalColors = (
  indices: Uint32Array,
  palette: [number, number, number][],
): Uint32Array => {
  const n = indices.length;
  const colors = new Uint32Array(n);
  const paletteLen = palette.length;
  for (let i = 0; i < n; i++) {
    const [r, g, b] = palette[indices[i]! % paletteLen]!;
    colors[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return colors;
};

/**
 * Encode continuous float values into packed RGBA8 u32 values using a color ramp.
 * Values are linearly interpolated between ramp stops.
 */
export const encodeContinuousColors = (
  values: Float32Array,
  min: number,
  range: number,
  ramp: [number, number, number][],
): Uint32Array => {
  const n = values.length;
  const colors = new Uint32Array(n);
  const stops = ramp.length - 1;
  const invRange = range > 0 ? 1 / range : 0;

  for (let i = 0; i < n; i++) {
    // Normalize to [0, 1]
    const t = Math.max(0, Math.min(1, (values[i]! - min) * invRange));
    // Map to ramp position
    const pos = t * stops;
    const idx = Math.min(Math.floor(pos), stops - 1);
    const frac = pos - idx;

    const [r0, g0, b0] = ramp[idx]!;
    const [r1, g1, b1] = ramp[idx + 1]!;

    const r = Math.round(r0 + frac * (r1 - r0));
    const g = Math.round(g0 + frac * (g1 - g0));
    const b = Math.round(b0 + frac * (b1 - b0));

    colors[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
  return colors;
};
