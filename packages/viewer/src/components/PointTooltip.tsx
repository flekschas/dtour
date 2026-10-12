import type { Metadata } from '@dtour/scatter';
import { useAtomValue } from 'jotai';
import { useState } from 'react';
import { tooltipColumnsAtom, tooltipImageFitAtom } from '../state/atoms.ts';

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
/** Content width of the widest tooltip: its max width less padding and border. */
const CONTENT_MAX_W = TOOLTIP_MAX_W - 22;
/** Text rows the default selection of columns shows before a "+N more" row. */
const DEFAULT_MAX_ROWS = 10;
/** Images narrower than this scale up to it. Images at most half as wide render pixelated. */
const MIN_IMAGE_W = 96;

type Row = { label: string; value: string; isColor: boolean; isImage: boolean };

const formatNumber = (val: number) => (Number.isInteger(val) ? String(val) : val.toPrecision(4));

/**
 * An image as wide as the tooltip and at least as wide as its natural width,
 * within limits, in a box with an aspect ratio between 1 and 2 so wide and tall
 * images don't make the tooltip too tall.
 */
const TooltipImage = ({
  src,
  alt,
  fit,
}: {
  src: string;
  alt: string;
  fit: 'contain' | 'cover';
}) => {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const ratio = size ? Math.min(Math.max(size.width / size.height, 1), 2) : 1;
  return (
    <img
      src={src}
      alt={alt}
      className="block w-full rounded-xs"
      style={{
        aspectRatio: ratio,
        objectFit: fit,
        minWidth: Math.min(Math.max(size?.width ?? 0, MIN_IMAGE_W), CONTENT_MAX_W),
        imageRendering: size && size.width * 2 <= MIN_IMAGE_W ? 'pixelated' : undefined,
      }}
      onLoad={(e) =>
        setSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })
      }
    />
  );
};

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
  const tooltipColumns = useAtomValue(tooltipColumnsAtom);
  const imageFit = useAtomValue(tooltipImageFitAtom);

  if (tooltipColumns?.length === 0) return null;

  const gap = POINT_R + 4;
  const goRight = cx + gap + ARROW_W + TOOLTIP_MAX_W < containerWidth;

  const rows: Row[] = [];
  let hiddenCount = 0;
  if (data && metadata) {
    const rowOf = (name: string): Row | null => {
      const isColor = name === colorColumn;
      const labels = metadata.categoricalLabels[name];
      if (labels) {
        const labelIdx = data.categoricalValues[name];
        if (labelIdx === undefined) return null;
        const isImage = metadata.imageColumnNames.includes(name);
        const value = labels[labelIdx] ?? String(labelIdx);
        if (isImage && !value) return null;
        return { label: name, value, isColor, isImage };
      }
      const val = data.numericValues[String(metadata.columnNames.indexOf(name))];
      if (val === undefined) return null;
      return { label: name, value: formatNumber(val), isColor, isImage: false };
    };

    // By default: images, the color column, other categoricals, the tour's numeric columns, other numerics
    let names = tooltipColumns;
    if (!names) {
      const active = activeIndices ?? [];
      const activeSet = new Set(active);
      const numeric = [
        ...active.map((d) => metadata.columnNames[d]!),
        ...metadata.columnNames.filter((_, d) => !activeSet.has(d)),
      ];
      const others = [...metadata.categoricalColumnNames, ...numeric].filter(
        (name) => name !== colorColumn,
      );
      const textNames = [...(colorColumn ? [colorColumn] : []), ...others];
      hiddenCount = Math.max(textNames.length - DEFAULT_MAX_ROWS, 0);
      names = [...metadata.imageColumnNames, ...textNames.slice(0, DEFAULT_MAX_ROWS)];
    }
    for (const name of names) {
      const row = rowOf(name);
      if (row) rows.push(row);
    }
  }

  if (data && rows.length === 0) return null;

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
          {rows.map((row) =>
            row.isImage ? (
              <TooltipImage key={row.label} src={row.value} alt={row.label} fit={imageFit} />
            ) : (
              <div
                key={row.label}
                className={`flex gap-2 ${row.isColor ? 'rounded-xs px-1 -mx-1' : ''}`}
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
            ),
          )}
          {hiddenCount > 0 && <div className="opacity-40">+{hiddenCount} more</div>}
        </div>
      ) : (
        <div className="opacity-40">Loading...</div>
      )}
    </div>
  );
};
