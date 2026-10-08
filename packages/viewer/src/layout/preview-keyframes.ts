/**
 * Pick `previewCount` of `keyframeCount` keyframes, spaced as evenly as
 * possible along the tour's normalized arc length. The first and last keyframe
 * are always included; the others are the keyframes closest to evenly spaced
 * targets between them (least squares, in tour order, without repeats).
 *
 * `arcLengths` holds each keyframe's cumulative normalized arc length. Without
 * it, keyframes count as evenly spaced.
 */
export function selectPreviewKeyframes(
  keyframeCount: number,
  previewCount: number,
  arcLengths?: ArrayLike<number> | null,
): number[] {
  if (keyframeCount <= previewCount) return Array.from({ length: keyframeCount }, (_, i) => i);

  const k = keyframeCount;
  const n = previewCount;
  const position = (i: number): number =>
    arcLengths && arcLengths.length >= k ? arcLengths[i]! : i / (k - 1);
  const start = position(0);
  const end = position(k - 1);

  // cost[i]: smallest sum of squared distances to the targets so far when the
  // current preview shows keyframe i. Preview j can only show keyframes
  // j…k−n+j, which leaves room for the previews before and after it.
  let cost = new Float64Array(k).fill(Number.POSITIVE_INFINITY);
  cost[0] = 0;
  const previous = Array.from({ length: n }, () => new Int32Array(k));
  for (let j = 1; j < n; j++) {
    const next = new Float64Array(k).fill(Number.POSITIVE_INFINITY);
    const target = start + (j * (end - start)) / (n - 1);
    let best = Number.POSITIVE_INFINITY;
    let bestIndex = -1;
    for (let i = j; i <= k - n + j; i++) {
      if (cost[i - 1]! < best) {
        best = cost[i - 1]!;
        bestIndex = i - 1;
      }
      if (j === n - 1 && i !== k - 1) continue;
      const distance = position(i) - target;
      next[i] = best + distance * distance;
      previous[j]![i] = bestIndex;
    }
    cost = next;
  }

  const keyframes = new Array<number>(n);
  keyframes[n - 1] = k - 1;
  for (let j = n - 1; j > 0; j--) keyframes[j - 1] = previous[j]![keyframes[j]!]!;
  return keyframes;
}
