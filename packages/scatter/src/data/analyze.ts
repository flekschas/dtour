export type ColumnAnalysis = {
  min: number;
  max: number;
  range: number;
};

export const analyzeColumn = (values: Float32Array): ColumnAnalysis => {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  const range = max - min || 1e-6;

  return { min, max, range };
};
