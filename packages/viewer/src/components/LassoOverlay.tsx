import type { Metadata, ScatterInstance } from '@dtour/scatter';
import { bitPackIndices } from '@dtour/scatter';
import { useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useLongPressIndicator } from '../hooks/useLongPressIndicator.ts';
import type { SpatialIndex } from '../hooks/useSpatialIndex.ts';
import { isHexColor } from '../lib/color-utils.ts';
import {
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  guidedSuspendedAtom,
  legendSelectionAtom,
  metadataAtom,
  pointColorAtom,
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

const HOVER_SEARCH_RADIUS_PX = 20;

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

type HoverState = {
  pointIndex: number;
  /** CSS position for tooltip. */
  x: number;
  y: number;
  /** Point data (loaded lazily). */
  data: {
    numericValues: Record<string, number>;
    categoricalValues: Record<string, number>;
  } | null;
};

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
  const setViewMode = useSetAtom(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setLegendSelection = useSetAtom(legendSelectionAtom);
  const metadata = useAtomValue(metadataAtom);
  const pointColor = useAtomValue(pointColorAtom);
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
        if (hoverPointRef.current === bestIdx) {
          setHover((prev) => (prev ? { ...prev, x: cssX, y: cssY } : null));
          return;
        }

        hoverPointRef.current = bestIdx;
        setHover({ pointIndex: bestIdx, x: cssX, y: cssY, data: null });

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
          // In grand mode, Escape returns to guided mode
          setGuidedSuspended(true);
          setViewMode('guided');
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
  }, [scatter, clearLongPress, viewMode, setViewMode, setGuidedSuspended, setLegendSelection]);

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

      {/* Hover tooltip */}
      {hover && <PointTooltip hover={hover} metadata={metadata} />}
    </div>
  );
};

const TOOLTIP_OFFSET = 12;

const PointTooltip = ({
  hover,
  metadata,
}: {
  hover: HoverState;
  metadata: Metadata | null;
}) => {
  const { data } = hover;

  // Format tooltip rows from lazy-loaded data
  const rows: { label: string; value: string }[] = [];
  if (data && metadata) {
    // Categorical columns
    for (const catName of metadata.categoricalColumnNames) {
      const labelIdx = data.categoricalValues[catName];
      if (labelIdx !== undefined) {
        const labels = metadata.categoricalLabels[catName];
        rows.push({
          label: catName,
          value: labels?.[labelIdx] ?? String(labelIdx),
        });
      }
    }
    // Numeric columns (keyed by dim index string)
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

  return (
    <div
      className="absolute z-50 pointer-events-none rounded bg-dtour-highlight text-dtour-bg px-2.5 py-1.5 text-xs shadow-[0_1px_4px_rgba(0,0,0,0.6)] max-w-[240px]"
      style={{
        left: hover.x + TOOLTIP_OFFSET,
        top: hover.y + TOOLTIP_OFFSET,
      }}
    >
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
