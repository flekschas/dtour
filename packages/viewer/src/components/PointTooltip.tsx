import type { Metadata } from '@dtour/scatter';

const hexLuminance = (hex: string): number => {
  const r = Number.parseInt(hex.slice(1, 3), 16) / 255;
  const g = Number.parseInt(hex.slice(3, 5), 16) / 255;
  const b = Number.parseInt(hex.slice(5, 7), 16) / 255;
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
};

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

type Row = { label: string; value: string; isColor: boolean };

export const PointTooltip = ({
  hover,
  metadata,
  cx,
  cy,
  containerWidth,
  color,
  colorColumn,
  activeIndices,
}: {
  hover: HoverState;
  metadata: Metadata | null;
  cx: number;
  cy: number;
  containerWidth: number;
  color?: string;
  colorColumn?: string | null;
  activeIndices?: number[];
}) => {
  const { data } = hover;

  const gap = POINT_R + 4;
  const goRight = cx + gap + ARROW_W + TOOLTIP_MAX_W < containerWidth;

  // Build rows in order: color dimension, other categoricals, other numericals, projection
  const rows: Row[] = [];
  if (data && metadata) {
    // 1. Color dimension first
    if (colorColumn) {
      if (metadata.categoricalColumnNames.includes(colorColumn)) {
        const labelIdx = data.categoricalValues[colorColumn];
        if (labelIdx !== undefined) {
          const labels = metadata.categoricalLabels[colorColumn];
          rows.push({
            label: colorColumn,
            value: labels?.[labelIdx] ?? String(labelIdx),
            isColor: true,
          });
        }
      } else {
        const colIdx = metadata.columnNames.indexOf(colorColumn);
        if (colIdx >= 0) {
          const val = data.numericValues[String(colIdx)];
          if (val !== undefined) {
            rows.push({
              label: colorColumn,
              value: Number.isInteger(val) ? String(val) : val.toPrecision(4),
              isColor: true,
            });
          }
        }
      }
    }

    // 2. Other categorical columns
    for (const catName of metadata.categoricalColumnNames) {
      if (catName === colorColumn) continue;
      const labelIdx = data.categoricalValues[catName];
      if (labelIdx !== undefined) {
        const labels = metadata.categoricalLabels[catName];
        rows.push({
          label: catName,
          value: labels?.[labelIdx] ?? String(labelIdx),
          isColor: false,
        });
      }
    }

    // 3. Projection dimensions (active in the tour), then remaining numerical
    const activeSet = activeIndices ? new Set(activeIndices) : null;
    const addNumRow = (d: number) => {
      if (metadata.columnNames[d] === colorColumn) return;
      const val = data.numericValues[String(d)];
      if (val === undefined) return;
      rows.push({
        label: metadata.columnNames[d]!,
        value: Number.isInteger(val) ? String(val) : val.toPrecision(4),
        isColor: false,
      });
    };
    if (activeSet) {
      for (const d of activeIndices!) addNumRow(d);
      for (let d = 0; d < metadata.columnNames.length; d++) {
        if (!activeSet.has(d)) addNumRow(d);
      }
    } else {
      for (let d = 0; d < metadata.columnNames.length; d++) addNumRow(d);
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
      {rows.length > 0 ? (
        <div className="flex flex-col gap-0.5">
          {rows.map((row) => (
            <div
              key={row.label}
              className={`flex gap-2 ${row.isColor ? 'rounded-sm px-1 -mx-1' : ''}`}
              style={
                row.isColor && color?.startsWith('#')
                  ? {
                      backgroundColor: color,
                      color: hexLuminance(color) > 0.4 ? '#000' : '#fff',
                    }
                  : undefined
              }
            >
              <span className={`truncate ${row.isColor ? '' : 'opacity-60'}`}>{row.label}</span>
              <span className="ml-auto text-right font-mono truncate">{row.value}</span>
            </div>
          ))}
        </div>
      ) : (
        data === null && <div className="opacity-40">Loading...</div>
      )}
    </div>
  );
};
