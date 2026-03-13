import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useLongPressIndicator } from '../hooks/useLongPressIndicator.ts';
import {
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  guidedSuspendedAtom,
  legendSelectionAtom,
  viewModeAtom,
} from '../state/atoms.ts';

type LassoOverlayProps = {
  scatter: ScatterInstance | null;
  width: number;
  height: number;
};

const LONG_PRESS_MS = 750;
const MIN_MOVE_PX = 5;
const MIN_POINT_DISTANCE = 5;
const THROTTLE_MS = 10;

/** Convert CSS coords to NDC accounting for camera. */
const cssToNdc = (
  x: number,
  y: number,
  width: number,
  height: number,
  panX: number,
  panY: number,
  zoom: number,
): [number, number] => {
  const aspect = width / height || 1;
  const ndcX = (((x / width) * 2 - 1) * aspect) / zoom - panX;
  const ndcY = -((y / height) * 2 - 1) / zoom - panY;
  return [ndcX, ndcY];
};

export const LassoOverlay = ({ scatter, width, height }: LassoOverlayProps) => {
  const panX = useAtomValue(cameraPanXAtom);
  const panY = useAtomValue(cameraPanYAtom);
  const zoom = useAtomValue(cameraZoomAtom);
  const viewMode = useAtomValue(viewModeAtom);
  const setViewMode = useSetAtom(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setLegendSelection = useSetAtom(legendSelectionAtom);

  const [lassoMode, setLassoMode] = useState(false);
  const [path, setPath] = useState<[number, number][]>([]);

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
      startPos.current = [e.clientX, e.clientY];

      showIndicator(e.clientX, e.clientY);

      longPressTimer.current = setTimeout(() => {
        setLassoMode(true);
        setPath([]);
        longPressTimer.current = null;
      }, LONG_PRESS_MS);
    },
    [lassoMode, showIndicator],
  );

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

  const handlePointerUp = useCallback(() => {
    clearLongPress();
    hideIndicator();

    if (!lassoMode || path.length < 3 || !scatter) {
      setLassoMode(false);
      setPath([]);
      return;
    }

    // Convert CSS path to NDC polygon and send to GPU worker
    const polygon = new Float32Array(path.length * 2);
    for (let i = 0; i < path.length; i++) {
      const [ndcX, ndcY] = cssToNdc(path[i]![0], path[i]![1], width, height, panX, panY, zoom);
      polygon[i * 2] = ndcX;
      polygon[i * 2 + 1] = ndcY;
    }

    scatter.lassoSelect(polygon);
    setLegendSelection(null);

    setLassoMode(false);
    setPath([]);
  }, [lassoMode, path, scatter, width, height, panX, panY, zoom, clearLongPress, hideIndicator, setLegendSelection]);

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
    </div>
  );
};
