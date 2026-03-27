/**
 * Bit-pack an array of selected point indices into a Uint32Array mask.
 * Each bit represents one point; bit i is set if index i appears in the input.
 *
 * @param indices - selected point indices (array, Int32Array, etc.)
 * @param numPoints - total number of points in the dataset
 * @returns Uint32Array of length ceil(numPoints / 32)
 */
export const bitPackIndices = (indices: ArrayLike<number>, numPoints: number): Uint32Array => {
  const mask = new Uint32Array(Math.ceil(numPoints / 32));
  for (let j = 0; j < indices.length; j++) {
    const idx = indices[j]!;
    if (idx >= 0 && idx < numPoints) {
      const w = idx >> 5;
      mask[w] = (mask[w] ?? 0) | (1 << (idx & 31));
    }
  }
  return mask;
};
