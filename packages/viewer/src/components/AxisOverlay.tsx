import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue, useSetAtom, useStore } from 'jotai';
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
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
  /** When true, axes track `currentBasisAtom` reactively and are not draggable. */
  readOnly?: boolean;
};

export type AxisOverlayHandle = {
  /** Imperatively update axis positions without a React re-render. */
  setBasis: (basis: Float32Array) => void;
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

type AxisElementRefs = {
  outlineLine: SVGLineElement;
  colorLine: SVGLineElement;
  circle: SVGCircleElement;
  text: SVGTextElement;
};

export const AxisOverlay = forwardRef<AxisOverlayHandle, AxisOverlayProps>(
  ({ scatter, width, height, readOnly }, ref) => {
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

    // Refs for imperative setBasis — must not depend on React state
    const sizeRef = useRef({ width, height });
    sizeRef.current = { width, height };
    const dimsRef = useRef(dims);
    dimsRef.current = dims;
    const activeIndicesRef = useRef(activeIndices);
    activeIndicesRef.current = activeIndices;

    // Per-dimension SVG element refs for imperative DOM updates
    const elementRefsMap = useRef(new Map<number, AxisElementRefs>());

    // Callback ref factory — captures SVG element refs during render
    const setElementRef = useCallback(
      (dimIndex: number, kind: keyof AxisElementRefs) => (el: SVGElement | null) => {
        if (!el) return;
        let entry = elementRefsMap.current.get(dimIndex);
        if (!entry) {
          entry = {} as AxisElementRefs;
          elementRefsMap.current.set(dimIndex, entry);
        }
        // biome-ignore lint/suspicious/noExplicitAny: assigning specific SVG element subtype
        (entry as any)[kind] = el;
      },
      [],
    );

    // Imperative setBasis — updates SVG attributes directly, no React re-render
    const updateDom = useCallback((basis: Float32Array) => {
      basisRef.current = basis;
      const { width: w, height: h } = sizeRef.current;
      const cx = w / 2;
      const cy = h / 2;
      const scale = Math.min(w, h) * 0.35;
      const d = dimsRef.current;

      for (const dim of activeIndicesRef.current) {
        const bx = basis[dim]!;
        const by = basis[d + dim]!;
        const endX = cx + bx * scale;
        const endY = cy - by * scale;
        const refs = elementRefsMap.current.get(dim);
        if (!refs) continue;
        const ex = String(endX);
        const ey = String(endY);
        refs.outlineLine.setAttribute('x2', ex);
        refs.outlineLine.setAttribute('y2', ey);
        refs.colorLine.setAttribute('x2', ex);
        refs.colorLine.setAttribute('y2', ey);
        refs.circle.setAttribute('cx', ex);
        refs.circle.setAttribute('cy', ey);
        refs.text.setAttribute('x', String(endX + (bx >= 0 ? 10 : -10)));
        refs.text.setAttribute('y', String(endY + 4));
        refs.text.setAttribute('text-anchor', bx >= 0 ? 'start' : 'end');
      }
    }, []);

    useImperativeHandle(ref, () => ({ setBasis: updateDom }), [updateDom]);

    // Read-only mode: subscribe to currentBasisAtom and mirror into basisRef
    // on every change so the axes track the guided tour interpolation.
    // This handles initial mount, slider drag, and wheel scrub.
    // During playback, the imperative setBasis path provides smoother updates.
    const currentBasis = useAtomValue(currentBasisAtom);
    useEffect(() => {
      if (!readOnly) return;
      if (currentBasis && currentBasis.length === dims * 2) {
        basisRef.current = new Float32Array(currentBasis);
        forceRender((n) => n + 1);
      }
    }, [readOnly, currentBasis, dims]);

    // Interactive mode: initialize basis once from the current projection
    // (read imperatively to avoid subscribing). Preserves the view when
    // switching into manual mode. Re-runs when activeIndices change to
    // zero out inactive dimensions.
    useEffect(() => {
      if (readOnly) return;
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
    }, [readOnly, dims, activeIndices, activeSet, store]);

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
        aria-label={readOnly ? 'Axis overlay' : 'Axis overlay for manual projection control'}
        className="absolute top-0 left-0 pointer-events-none"
        onPointerMove={readOnly ? undefined : handlePointerMove}
        onPointerUp={readOnly ? undefined : handlePointerUp}
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
                ref={setElementRef(d, 'outlineLine')}
                x1={cx}
                y1={cy}
                x2={lineEndX}
                y2={lineEndY}
                stroke="var(--color-dtour-bg)"
                strokeWidth={3}
                strokeOpacity={0.6}
              />
              {/* Axis line */}
              <line
                ref={setElementRef(d, 'colorLine')}
                x1={cx}
                y1={cy}
                x2={lineEndX}
                y2={lineEndY}
                stroke={color}
                strokeWidth={1}
                strokeOpacity={0.6}
              />
              {/* Handle: draggable in interactive mode, decorative dot in read-only */}
              <circle
                ref={setElementRef(d, 'circle')}
                cx={lineEndX}
                cy={lineEndY}
                r={readOnly ? HANDLE_RADIUS * 0.6 : HANDLE_RADIUS}
                fill={color}
                stroke="var(--color-dtour-bg)"
                strokeWidth={1}
                className={readOnly ? undefined : 'cursor-grab pointer-events-auto'}
                onPointerDown={readOnly ? undefined : (e) => handlePointerDown(d, e)}
                onKeyDown={
                  readOnly
                    ? undefined
                    : (e) =>
                        e.key === 'Enter' && handleAltClick(d, e as unknown as React.MouseEvent)
                }
                onClick={readOnly ? undefined : (e) => handleAltClick(d, e)}
              />
              {/* Label */}
              <text
                ref={setElementRef(d, 'text')}
                x={lineEndX + (bx >= 0 ? 10 : -10)}
                y={lineEndY + 4}
                fill={color}
                fontSize={11}
                fontFamily="monospace"
                textAnchor={bx >= 0 ? 'start' : 'end'}
                className="pointer-events-none select-none"
                stroke="var(--color-dtour-bg)"
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
  },
);
