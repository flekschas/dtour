import type { Metadata, ScatterInstance } from '@dtour/scatter';
import {
  GLASBEY_DARK,
  GLASBEY_LIGHT,
  MAGMA_25,
  OKABE_ITO,
  VIRIDIS_25,
  bitPackIndices,
} from '@dtour/scatter';
import { useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useLongPressIndicator } from '../hooks/useLongPressIndicator.ts';
import type { SpatialIndex } from '../hooks/useSpatialIndex.ts';
import { isHexColor } from '../lib/color-utils.ts';
import {
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  color2dEnabledAtom,
  colorMapAtom,
  currentBasisAtom,
  grandExitTargetAtom,
  legendSelectionAtom,
  metadataAtom,
  paletteAtom,
  pointColorAtom,
  resolvedThemeAtom,
  viewModeAtom,
} from '../state/atoms.ts';

type LassoOverlayProps = {
  scatter: ScatterInstance | null;
  width: number;
  height: number;
  /** Vertical offset of the overlay within the full canvas (e.g. toolbar height). */
  offsetY: number;
  /** Spatial index for hover picking. */
  spatialIndex: React.RefObject<SpatialIndex | null>;
  /** Camera inset offset Y (NDC). */
  insetOffsetY: number;
  /** Camera inset zoom multiplier. */
  insetZoom: number;
};

const LONG_PRESS_MS = 750;
const MIN_MOVE_PX = 5;
const MIN_POINT_DISTANCE = 5;
const THROTTLE_MS = 10;

/** Convert overlay-local CSS coords to full-canvas clip-space NDC [-1,1].
 *  The overlay may be offset from the top of the canvas (e.g. by a toolbar),
 *  so we translate to full-canvas pixel coords before normalizing.
 *  The GPU worker then inverts the full camera transform (zoom, pan, inset). */
const cssToNdc = (
  x: number,
  y: number,
  canvasWidth: number,
  canvasHeight: number,
  offsetY: number,
): [number, number] => {
  const ndcX = (x / canvasWidth) * 2 - 1;
  const ndcY = -(((y + offsetY) / canvasHeight) * 2 - 1);
  return [ndcX, ndcY];
};

const HIT_RADIUS_PX = 3;

const HOVER_SEARCH_RADIUS_PX = 12;

/** Convert projection-space coords to overlay-local CSS coords (inverse of queryNearest's transform). */
const projToCss = (
  projX: number,
  projY: number,
  width: number,
  height: number,
  offsetY: number,
  insetOffsetY: number,
  insetZoom: number,
  cameraZoom: number,
  cameraPanX: number,
  cameraPanY: number,
): [number, number] => {
  const canvasH = height + offsetY;
  const aspect = width / canvasH || 1;
  const zoomIz = cameraZoom * insetZoom;
  const ndcX = ((projX + cameraPanX) * zoomIz) / aspect;
  const ndcY = (projY + cameraPanY) * zoomIz + insetOffsetY;
  return [((ndcX + 1) / 2) * width, ((1 - ndcY) / 2) * canvasH - offsetY];
};

/** Query kdbush for the nearest point index at the given CSS position. Returns -1 if no hit. */
const queryNearest = (
  si: SpatialIndex,
  cssX: number,
  cssY: number,
  width: number,
  height: number,
  offsetY: number,
  insetOffsetY: number,
  insetZoom: number,
  cameraZoom: number,
  cameraPanX: number,
  cameraPanY: number,
  radiusPx: number,
): number => {
  const canvasH = height + offsetY;
  const [ndcX, ndcY] = cssToNdc(cssX, cssY, width, canvasH, offsetY);
  const aspect = width / canvasH || 1;
  const zoomIz = cameraZoom * insetZoom;
  const projX = (ndcX * aspect) / zoomIz - cameraPanX;
  const projY = (ndcY - insetOffsetY) / zoomIz - cameraPanY;
  const radiusProj = ((radiusPx / Math.min(width, canvasH)) * 2 * Math.max(aspect, 1)) / zoomIz;

  const hits = si.index.within(projX, projY, radiusProj);
  if (hits.length === 0) return -1;

  let bestIdx = hits[0]!;
  let bestDist2 = Number.POSITIVE_INFINITY;
  for (const idx of hits) {
    const dx = si.positions[idx * 2]! - projX;
    const dy = si.positions[idx * 2 + 1]! - projY;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestDist2) {
      bestDist2 = d2;
      bestIdx = idx;
    }
  }
  return bestIdx;
};

const rgb255ToHex = (r: number, g: number, b: number): string =>
  `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;

const samplePalette = (pal: [number, number, number][], t: number): string => {
  const n = pal.length - 1;
  const f = Math.max(0, Math.min(n, t * n));
  const i = Math.min(n - 1, Math.floor(f));
  const frac = f - i;
  const a = pal[i]!;
  const b = pal[i + 1]!;
  return rgb255ToHex(
    Math.round(a[0] + (b[0] - a[0]) * frac),
    Math.round(a[1] + (b[1] - a[1]) * frac),
    Math.round(a[2] + (b[2] - a[2]) * frac),
  );
};

const resolveHoverColor = (
  pointColor: [number, number, number] | string,
  hoverData: {
    numericValues: Record<string, number>;
    categoricalValues: Record<string, number>;
  } | null,
  metadata: Metadata | null,
  colorMap: Record<string, string | { light: string; dark: string }> | null,
  palette: 'viridis' | 'magma',
  theme: 'light' | 'dark',
): string => {
  // Uniform RGB tuple (0–1 range)
  if (Array.isArray(pointColor)) {
    return rgb255ToHex(
      Math.round(pointColor[0] * 255),
      Math.round(pointColor[1] * 255),
      Math.round(pointColor[2] * 255),
    );
  }
  // Uniform hex string
  if (isHexColor(pointColor)) return pointColor;
  // Column-encoded color — need lazy data to resolve
  if (!hoverData || !metadata) return 'white';

  const colName = pointColor;

  // Categorical
  if (metadata.categoricalColumnNames.includes(colName)) {
    const labelIdx = hoverData.categoricalValues[colName];
    if (labelIdx === undefined) return 'white';
    if (colorMap) {
      const labelName = metadata.categoricalLabels[colName]?.[labelIdx];
      if (labelName !== undefined && colorMap[labelName] !== undefined) {
        const entry = colorMap[labelName]!;
        return typeof entry === 'string' ? entry : entry[theme];
      }
    }
    const glasbey = theme === 'light' ? GLASBEY_LIGHT : GLASBEY_DARK;
    const [r, g, b] =
      labelIdx < OKABE_ITO.length
        ? OKABE_ITO[labelIdx]!
        : glasbey[(labelIdx - OKABE_ITO.length) % glasbey.length]!;
    return rgb255ToHex(r, g, b);
  }

  // Continuous
  const colIdx = metadata.columnNames.indexOf(colName);
  if (colIdx >= 0) {
    const val = hoverData.numericValues[String(colIdx)];
    if (val === undefined) return 'white';
    const min = metadata.mins[colIdx] ?? 0;
    const max = metadata.maxes[colIdx] ?? 1;
    const t = max > min ? (val - min) / (max - min) : 0;
    const basePal = palette === 'magma' ? MAGMA_25 : VIRIDIS_25;
    const colorPal =
      theme === 'light' ? ([...basePal].reverse() as [number, number, number][]) : basePal;
    return samplePalette(colorPal, Math.max(0, Math.min(1, t)));
  }

  return 'white';
};

type HoverState = {
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

const HoverHighlight = ({ cx, cy, color }: { cx: number; cy: number; color: string }) => (
  <svg
    className="absolute pointer-events-none"
    width={10}
    height={10}
    style={{ left: cx, top: cy, transform: 'translate(-50%, -50%)' }}
  >
    <title>Hovered point</title>
    <circle cx={5} cy={5} r={4} fill={color} />
  </svg>
);

export const LassoOverlay = ({
  scatter,
  width,
  height,
  offsetY,
  spatialIndex,
  insetOffsetY,
  insetZoom,
}: LassoOverlayProps) => {
  const viewMode = useAtomValue(viewModeAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setLegendSelection = useSetAtom(legendSelectionAtom);
  const store = useStore();
  const metadata = useAtomValue(metadataAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const color2dEnabled = useAtomValue(color2dEnabledAtom);
  const colorMap = useAtomValue(colorMapAtom);
  const palette = useAtomValue(paletteAtom);
  const theme = useAtomValue(resolvedThemeAtom);
  const cameraPanX = useAtomValue(cameraPanXAtom);
  const cameraPanY = useAtomValue(cameraPanYAtom);
  const cameraZoom = useAtomValue(cameraZoomAtom);

  const [lassoMode, setLassoMode] = useState(false);
  const [path, setPath] = useState<[number, number][]>([]);
  const [hover, setHover] = useState<HoverState | null>(null);
  const pendingPointDataRef = useRef<number>(-1);
  const hoverPointRef = useRef<number>(-1);

  // Refs for values read inside pointermove to avoid recreating the callback
  const hoverCtxRef = useRef({
    scatter,
    width,
    height,
    offsetY,
    insetOffsetY,
    insetZoom,
    cameraPanX,
    cameraPanY,
    cameraZoom,
  });
  hoverCtxRef.current = {
    scatter,
    width,
    height,
    offsetY,
    insetOffsetY,
    insetZoom,
    cameraPanX,
    cameraPanY,
    cameraZoom,
  };

  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startPos = useRef<[number, number] | null>(null);
  const lastPointTime = useRef(0);
  const overlayRef = useRef<HTMLDivElement>(null);

  const { show: showIndicator, hide: hideIndicator } = useLongPressIndicator();

  // Clear hover whenever the projection changes so the highlight/tooltip
  // don't hang over a stale position until the pointer moves again.
  useEffect(() => {
    return store.sub(currentBasisAtom, () => {
      if (hoverPointRef.current !== -1) {
        hoverPointRef.current = -1;
        setHover(null);
      }
    });
  }, [store]);

  const clearLongPress = useCallback(() => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
      hideIndicator(); // Only revert if cancelling (timer was still pending)
    }
  }, [hideIndicator]);

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (lassoMode || e.button !== 0) return;
      // Prevent native drag-start (e.g. image/element drag in Marimo widgets)
      e.preventDefault();
      startPos.current = [e.clientX, e.clientY];
      hoverPointRef.current = -1;
      setHover(null);

      showIndicator(e.clientX, e.clientY);

      longPressTimer.current = setTimeout(() => {
        setLassoMode(true);
        setPath([]);
        longPressTimer.current = null;
      }, LONG_PRESS_MS);
    },
    [lassoMode, showIndicator],
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: spatialIndex and hoverCtxRef are refs read inside the callback — we intentionally avoid deps to keep the handler stable
  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      // Cancel long press if moved too far
      if (!lassoMode && startPos.current && longPressTimer.current) {
        const dx = e.clientX - startPos.current[0];
        const dy = e.clientY - startPos.current[1];
        if (Math.sqrt(dx * dx + dy * dy) > MIN_MOVE_PX) {
          clearLongPress();
        }
        return;
      }

      // Hover picking when idle (no button pressed, no lasso)
      if (!lassoMode && !startPos.current) {
        const si = spatialIndex.current;
        if (!si) {
          if (hoverPointRef.current !== -1) {
            hoverPointRef.current = -1;
            setHover(null);
          }
          return;
        }

        const ctx = hoverCtxRef.current;
        const rect = overlayRef.current?.getBoundingClientRect();
        if (!rect) return;

        const cssX = e.clientX - rect.left;
        const cssY = e.clientY - rect.top;

        const bestIdx = queryNearest(
          si,
          cssX,
          cssY,
          ctx.width,
          ctx.height,
          ctx.offsetY,
          ctx.insetOffsetY,
          ctx.insetZoom,
          ctx.cameraZoom,
          ctx.cameraPanX,
          ctx.cameraPanY,
          HOVER_SEARCH_RADIUS_PX,
        );

        if (bestIdx === -1) {
          if (hoverPointRef.current !== -1) {
            hoverPointRef.current = -1;
            setHover(null);
          }
          return;
        }

        // Only update if the point changed
        if (hoverPointRef.current === bestIdx) return;

        hoverPointRef.current = bestIdx;
        setHover({
          pointIndex: bestIdx,
          pointProjX: si.positions[bestIdx * 2]!,
          pointProjY: si.positions[bestIdx * 2 + 1]!,
          data: null,
        });

        // Lazily fetch point data
        if (ctx.scatter) {
          pendingPointDataRef.current = bestIdx;
          ctx.scatter.getPointData(bestIdx).then((result) => {
            if (pendingPointDataRef.current !== bestIdx) return;
            setHover((prev) =>
              prev && prev.pointIndex === bestIdx
                ? {
                    ...prev,
                    data: {
                      numericValues: result.numericValues,
                      categoricalValues: result.categoricalValues,
                    },
                  }
                : prev,
            );
          });
        }
        return;
      }

      if (!lassoMode) return;

      const now = performance.now();
      if (now - lastPointTime.current < THROTTLE_MS) return;

      const rect = overlayRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      setPath((prev) => {
        if (prev.length > 0) {
          const last = prev[prev.length - 1]!;
          const dx = x - last[0];
          const dy = y - last[1];
          if (Math.sqrt(dx * dx + dy * dy) < MIN_POINT_DISTANCE) return prev;
        }
        return [...prev, [x, y]];
      });

      lastPointTime.current = now;
    },
    [lassoMode, clearLongPress],
  );

  const handlePointerLeave = useCallback(() => {
    // Pointer left the canvas area (e.g. moved to sidebar / resize handle).
    // Cancel the long-press timer so we don't accidentally enter lasso mode.
    if (!lassoMode) {
      clearLongPress();
      startPos.current = null;
    }
    hoverPointRef.current = -1;
    setHover(null);
  }, [lassoMode, clearLongPress]);

  // Also cancel on window blur (e.g. user switches tabs/apps mid-press)
  useEffect(() => {
    const handleBlur = () => {
      if (!lassoMode) {
        clearLongPress();
        startPos.current = null;
      }
    };
    window.addEventListener('blur', handleBlur);
    return () => window.removeEventListener('blur', handleBlur);
  }, [lassoMode, clearLongPress]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: spatialIndex and hoverCtxRef are refs read inside the callback
  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      clearLongPress();
      hideIndicator();

      if (!lassoMode || path.length < 3 || !scatter) {
        // Not a lasso — check if it's a short click (no significant movement)
        if (!lassoMode && scatter && startPos.current && metadata) {
          const dx = e.clientX - startPos.current[0];
          const dy = e.clientY - startPos.current[1];
          if (Math.sqrt(dx * dx + dy * dy) <= MIN_MOVE_PX) {
            const si = spatialIndex.current;
            if (si) {
              const rect = overlayRef.current?.getBoundingClientRect();
              if (rect) {
                const ctx = hoverCtxRef.current;
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const bestIdx = queryNearest(
                  si,
                  x,
                  y,
                  ctx.width,
                  ctx.height,
                  ctx.offsetY,
                  ctx.insetOffsetY,
                  ctx.insetZoom,
                  ctx.cameraZoom,
                  ctx.cameraPanX,
                  ctx.cameraPanY,
                  HIT_RADIUS_PX * 2,
                );

                if (bestIdx !== -1) {
                  if (e.altKey) {
                    // Alt+click: select all points with same category/band
                    scatter.getPointData(bestIdx).then((result) => {
                      const color = pointColor;
                      if (typeof color !== 'string' || isHexColor(color)) return;

                      // Categorical
                      if (metadata.categoricalColumnNames.includes(color)) {
                        const labelIdx = result.categoricalValues[color];
                        if (labelIdx !== undefined) {
                          setLegendSelection(new Set([labelIdx]));
                        }
                        return;
                      }

                      // Continuous
                      const colIndex = metadata.columnNames.indexOf(color);
                      if (colIndex !== -1) {
                        const val = result.numericValues[String(colIndex)];
                        if (val !== undefined) {
                          const min = metadata.mins[colIndex]!;
                          const max = metadata.maxes[colIndex]!;
                          const range = max - min;
                          const t = range > 0 ? (val - min) / range : 0;
                          const stopIdx = Math.max(
                            0,
                            Math.min(12, Math.round(Math.max(0, Math.min(1, t)) * 12)),
                          );
                          setLegendSelection(new Set([stopIdx]));
                        }
                      }
                    });
                  } else {
                    // Plain click: select just this point (fully synchronous)
                    const mask = bitPackIndices([bestIdx], metadata.rowCount);
                    scatter.setSelectionMask(mask);
                    setLegendSelection(null);
                  }
                }
              }
            }
          }
        }
        setLassoMode(false);
        setPath([]);
        startPos.current = null;
        return;
      }

      // Convert CSS path to NDC polygon and send to GPU worker
      const polygon = new Float32Array(path.length * 2);
      for (let i = 0; i < path.length; i++) {
        const [ndcX, ndcY] = cssToNdc(path[i]![0], path[i]![1], width, height + offsetY, offsetY);
        polygon[i * 2] = ndcX;
        polygon[i * 2 + 1] = ndcY;
      }

      scatter.lassoSelect(polygon);
      setLegendSelection(null);

      setLassoMode(false);
      setPath([]);
      startPos.current = null;
    },
    [
      lassoMode,
      path,
      scatter,
      width,
      height,
      offsetY,
      clearLongPress,
      hideIndicator,
      setLegendSelection,
      metadata,
      pointColor,
    ],
  );

  // Double-click or Escape clears selection
  const handleDoubleClick = useCallback(() => {
    scatter?.clearSelection();
    setLegendSelection(null);
  }, [scatter, setLegendSelection]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (viewMode === 'grand') {
          // In grand mode, route through the exit atom so the ease-out
          // animation completes before the mode switch.
          setGrandExitTarget('guided');
        } else {
          scatter?.clearSelection();
        }
        setLegendSelection(null);
        setLassoMode(false);
        setPath([]);
        clearLongPress();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [scatter, clearLongPress, viewMode, setGrandExitTarget, setLegendSelection]);

  // Build path string for SVG polygon
  const pathStr = path.map(([x, y]) => `${x},${y}`).join(' ');

  return (
    <div
      ref={overlayRef}
      className="absolute top-0 left-0 touch-none"
      style={{ width, height, cursor: lassoMode ? 'crosshair' : undefined }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
      onDoubleClick={handleDoubleClick}
    >
      {/* Lasso polygon */}
      {lassoMode && path.length > 1 && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Lasso selection path"
          className="absolute top-0 left-0 pointer-events-none"
        >
          <polygon
            points={pathStr}
            fill="rgba(79, 143, 247, 0.1)"
            stroke="#4f8ff7"
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
        </svg>
      )}

      {/* Hover highlight + tooltip, both anchored at the projected point position */}
      {hover &&
        (() => {
          const [cx, cy] = projToCss(
            hover.pointProjX,
            hover.pointProjY,
            width,
            height,
            offsetY,
            insetOffsetY,
            insetZoom,
            cameraZoom,
            cameraPanX,
            cameraPanY,
          );
          const color = color2dEnabled
            ? 'white'
            : resolveHoverColor(pointColor, hover.data, metadata, colorMap, palette, theme);
          return (
            <>
              <HoverHighlight cx={cx} cy={cy} color={color} />
              <PointTooltip
                hover={hover}
                metadata={metadata}
                cx={cx}
                cy={cy}
                containerWidth={width}
              />
            </>
          );
        })()}
    </div>
  );
};

/** Gap from circle edge to arrow tip (px). */
const POINT_R = 4;
const ARROW_W = 6;
const TOOLTIP_MAX_W = 240;

const PointTooltip = ({
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
