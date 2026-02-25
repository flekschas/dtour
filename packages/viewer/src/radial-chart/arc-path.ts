/** Angular start offset: 10:30 o'clock position (-135 degrees from +x axis). */
const START_DEG = -135;

/**
 * Center angle in radians for keyframe `index` out of `count` total keyframes.
 * Uses the same angular convention as CircularSlider (10:30 start, clockwise).
 */
export const keyframeAngle = (index: number, count: number): number =>
  (((index / count) * 360 + START_DEG) * Math.PI) / 180;

/**
 * SVG `d` attribute for an annular sector (arc segment between two radii).
 *
 * Draws: outer arc forward -> line to inner -> inner arc backward -> close.
 *
 * All angles in radians. Center is at (0, 0) — use a `<g transform>` to position.
 */
export const arcPath = (
  rInner: number,
  rOuter: number,
  angleStart: number,
  angleEnd: number,
): string => {
  const outerX1 = rOuter * Math.cos(angleStart);
  const outerY1 = rOuter * Math.sin(angleStart);
  const outerX2 = rOuter * Math.cos(angleEnd);
  const outerY2 = rOuter * Math.sin(angleEnd);

  const innerX1 = rInner * Math.cos(angleEnd);
  const innerY1 = rInner * Math.sin(angleEnd);
  const innerX2 = rInner * Math.cos(angleStart);
  const innerY2 = rInner * Math.sin(angleStart);

  // Determine if the arc spans more than 180 degrees
  let sweep = angleEnd - angleStart;
  if (sweep < 0) sweep += 2 * Math.PI;
  const largeArc = sweep > Math.PI ? 1 : 0;

  return [
    `M ${outerX1} ${outerY1}`,
    `A ${rOuter} ${rOuter} 0 ${largeArc} 1 ${outerX2} ${outerY2}`,
    `L ${innerX1} ${innerY1}`,
    `A ${rInner} ${rInner} 0 ${largeArc} 0 ${innerX2} ${innerY2}`,
    'Z',
  ].join(' ');
};
