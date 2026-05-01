import type { Metadata } from '@dtour/scatter';

export type HoverState = {
  pointIndex: number;
  /** Projection-space coords for stable highlight/tooltip anchoring. */
  pointProjX: number;
  pointProjY: number;
  /** Point data (loaded lazily). */
  data: {
    numericValues: Record<string, number>;
    categoricalValues: Record<string, number>;
  } | null;
};

/** Gap from circle edge to arrow tip (px). */
const POINT_R = 4;
const ARROW_W = 6;
const TOOLTIP_MAX_W = 240;

export const PointTooltip = ({
  hover,
  metadata,
  cx,
  cy,
  containerWidth,
}: {
  hover: HoverState;
  metadata: Metadata | null;
  cx: number;
  cy: number;
  containerWidth: number;
}) => {
  const { data } = hover;

  const gap = POINT_R + 4;
  const goRight = cx + gap + ARROW_W + TOOLTIP_MAX_W < containerWidth;

  // Format tooltip rows from lazy-loaded data
  const rows: { label: string; value: string }[] = [];
  if (data && metadata) {
    for (const catName of metadata.categoricalColumnNames) {
      const labelIdx = data.categoricalValues[catName];
      if (labelIdx !== undefined) {
        const labels = metadata.categoricalLabels[catName];
        rows.push({ label: catName, value: labels?.[labelIdx] ?? String(labelIdx) });
      }
    }
    for (let d = 0; d < metadata.columnNames.length; d++) {
      const val = data.numericValues[String(d)];
      if (val !== undefined) {
        rows.push({
          label: metadata.columnNames[d]!,
          value: Number.isInteger(val) ? String(val) : val.toPrecision(4),
        });
      }
    }
  }

  // Arrow: SVG path draws only the two outer edges (no base stroke over tooltip border).
  // Polygon fills the triangle with the tooltip background color.
  const arrowSvg = goRight ? (
    <svg
      style={{
        position: 'absolute',
        left: -ARROW_W,
        top: '50%',
        transform: 'translateY(-50%)',
        overflow: 'visible',
      }}
      width={ARROW_W}
      height={ARROW_W * 2}
    >
      <title>Tooltip arrow</title>
      <polygon
        points={`${ARROW_W},0 0,${ARROW_W} ${ARROW_W},${ARROW_W * 2}`}
        style={{ fill: 'var(--color-dtour-bg)' }}
      />
      <path
        d={`M ${ARROW_W},0.5 L 0,${ARROW_W} L ${ARROW_W},${ARROW_W * 2 - 0.5}`}
        fill="none"
        style={{ stroke: 'var(--color-dtour-border)', strokeWidth: 1 }}
      />
    </svg>
  ) : (
    <svg
      style={{
        position: 'absolute',
        right: -ARROW_W,
        top: '50%',
        transform: 'translateY(-50%)',
        overflow: 'visible',
      }}
      width={ARROW_W}
      height={ARROW_W * 2}
    >
      <title>Tooltip arrow</title>
      <polygon
        points={`0,0 ${ARROW_W},${ARROW_W} 0,${ARROW_W * 2}`}
        style={{ fill: 'var(--color-dtour-bg)' }}
      />
      <path
        d={`M 0,0.5 L ${ARROW_W},${ARROW_W} L 0,${ARROW_W * 2 - 0.5}`}
        fill="none"
        style={{ stroke: 'var(--color-dtour-border)', strokeWidth: 1 }}
      />
    </svg>
  );

  return (
    <div
      className="absolute z-50 pointer-events-none rounded border border-dtour-border bg-dtour-bg text-dtour-text px-2.5 py-1.5 text-xs shadow-md max-w-[240px]"
      style={
        goRight
          ? { left: cx + gap + ARROW_W, top: cy, transform: 'translateY(-50%)' }
          : { right: containerWidth - cx + gap + ARROW_W, top: cy, transform: 'translateY(-50%)' }
      }
    >
      {arrowSvg}
      <div className="font-medium mb-0.5 opacity-60">Point {hover.pointIndex.toLocaleString()}</div>
      {rows.length > 0 ? (
        <div className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
          {rows.map((row) => (
            <div key={row.label} className="contents">
              <span className="opacity-60 truncate">{row.label}</span>
              <span className="text-right font-mono truncate">{row.value}</span>
            </div>
          ))}
        </div>
      ) : (
        data === null && <div className="opacity-40">Loading...</div>
      )}
    </div>
  );
};
