import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { gramSchmidt } from '../lib/gram-schmidt.ts';
import {
  activeIndicesAtom,
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  currentBasisAtom,
  metadataAtom,
} from '../state/atoms.ts';

type AxisOverlayProps = {
  scatter: ScatterInstance | null;
  width: number;
  height: number;
};

/** Convert SVG pixel delta to NDC delta (inverse camera). */
const svgDeltaToNdc = (
  dx: number,
  dy: number,
  width: number,
  height: number,
  zoom: number,
): [number, number] => {
  const aspect = width / height || 1;
  const ndcDx = ((dx / width) * 2 * aspect) / zoom;
  const ndcDy = (-(dy / height) * 2) / zoom;
  return [ndcDx, ndcDy];
};

const HANDLE_RADIUS = 6;
const AXIS_COLORS = ['#e06060', '#60a0e0', '#60c060', '#c080e0', '#e0a040', '#40c0c0'];

export const AxisOverlay = ({ scatter, width, height }: AxisOverlayProps) => {
  const metadata = useAtomValue(metadataAtom);
  const panX = useAtomValue(cameraPanXAtom);
  const panY = useAtomValue(cameraPanYAtom);
  const zoom = useAtomValue(cameraZoomAtom);
  const activeIndices = useAtomValue(activeIndicesAtom);
  const store = useStore();
  const setCurrentBasis = useSetAtom(currentBasisAtom);

  const basisRef = useRef<Float32Array | null>(null);
  const [, forceRender] = useState(0);
  const draggingRef = useRef<number | null>(null);

  const dims = metadata?.dimCount ?? 0;
  const columnNames = metadata?.columnNames ?? [];

  const activeSet = useMemo(() => new Set(activeIndices), [activeIndices]);

  // Initialize basis from the current projection (read imperatively to avoid
  // subscribing). This preserves the view when switching into manual mode.
  // Re-runs when activeIndices change to zero out inactive dimensions.
  useEffect(() => {
    if (dims < 2 || activeIndices.length < 2) return;
    const current = store.get(currentBasisAtom);
    let basis: Float32Array;
    if (current && current.length === dims * 2) {
      basis = new Float32Array(current);
    } else {
      basis = new Float32Array(dims * 2);
      basis[activeIndices[0]!] = 1;
      basis[dims + activeIndices[1]!] = 1;
    }
    // Zero out inactive dimensions and re-orthonormalize
    for (let d = 0; d < dims; d++) {
      if (!activeSet.has(d)) {
        basis[d] = 0;
        basis[dims + d] = 0;
      }
    }
    gramSchmidt(basis, dims);
    basisRef.current = basis;
    forceRender((n) => n + 1);
    // No setDirectBasis here — the GPU already shows this projection
  }, [dims, activeIndices, activeSet, store]);

  const sendBasis = useCallback(() => {
    if (!scatter || !basisRef.current) return;
    const copy = basisRef.current.slice();
    scatter.setDirectBasis(copy);
    setCurrentBasis(new Float32Array(copy));
  }, [scatter, setCurrentBasis]);

  const handlePointerDown = useCallback((dimIndex: number, e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as Element).setPointerCapture(e.pointerId);
    draggingRef.current = dimIndex;
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = draggingRef.current;
      if (d === null || !basisRef.current) return;
      e.preventDefault();

      const [ndcDx, ndcDy] = svgDeltaToNdc(e.movementX, e.movementY, width, height, zoom);
      const basis = basisRef.current;

      // Update basis row d: column 0 (x projection) and column 1 (y projection)
      basis[d]! += ndcDx;
      basis[dims + d]! += ndcDy;

      // Re-orthonormalize
      gramSchmidt(basis, dims);

      forceRender((n) => n + 1);
      sendBasis();
    },
    [width, height, zoom, dims, sendBasis],
  );

  const handlePointerUp = useCallback(() => {
    draggingRef.current = null;
  }, []);

  const handleAltClick = useCallback(
    (dimIndex: number, e: React.MouseEvent) => {
      if (!e.altKey || !basisRef.current) return;
      e.preventDefault();
      const basis = basisRef.current;
      // Negate basis row d (flip axis direction)
      basis[dimIndex]! = -basis[dimIndex]!;
      basis[dims + dimIndex]! = -basis[dims + dimIndex]!;
      forceRender((n) => n + 1);
      sendBasis();
    },
    [dims, sendBasis],
  );

  if (!metadata || dims < 2 || !basisRef.current || width === 0) return null;

  const basis = basisRef.current;
  const cx = width / 2;
  const cy = height / 2;

  // Scale factor for axis lines (proportion of view)
  const scale = Math.min(width, height) * 0.35;

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label="Axis overlay for manual projection control"
      className="absolute top-0 left-0 pointer-events-none"
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {activeIndices.map((d) => {
        // Basis row d gives (x, y) projection weights
        const bx = basis[d]!;
        const by = basis[dims + d]!;

        // Use scale to determine line endpoint from center
        const lineEndX = cx + bx * scale;
        const lineEndY = cy - by * scale; // flip Y

        const color = AXIS_COLORS[d % AXIS_COLORS.length]!;
        const label = columnNames[d] ?? `dim${d}`;

        return (
          <g key={label}>
            {/* Axis line outline */}
            <line
              x1={cx}
              y1={cy}
              x2={lineEndX}
              y2={lineEndY}
              stroke="#000"
              strokeWidth={3}
              strokeOpacity={0.6}
            />
            {/* Axis line */}
            <line
              x1={cx}
              y1={cy}
              x2={lineEndX}
              y2={lineEndY}
              stroke={color}
              strokeWidth={1}
              strokeOpacity={0.6}
            />
            {/* Draggable handle */}
            <circle
              cx={lineEndX}
              cy={lineEndY}
              r={HANDLE_RADIUS}
              fill={color}
              stroke="#000"
              strokeWidth={1}
              className="cursor-grab pointer-events-auto"
              onPointerDown={(e) => handlePointerDown(d, e)}
              onKeyDown={(e) =>
                e.key === 'Enter' && handleAltClick(d, e as unknown as React.MouseEvent)
              }
              onClick={(e) => handleAltClick(d, e)}
            />
            {/* Label */}
            <text
              x={lineEndX + (bx >= 0 ? 10 : -10)}
              y={lineEndY + 4}
              fill={color}
              fontSize={11}
              fontFamily="monospace"
              textAnchor={bx >= 0 ? 'start' : 'end'}
              className="pointer-events-none select-none"
              stroke="black"
              strokeWidth={2}
              paintOrder="stroke"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
