import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import { cn } from '../lib/utils';

export type CircularSliderHandle = {
  /** Update slider position imperatively without triggering a React re-render. */
  setPosition: (value: number) => void;
};

export type PreviewCenter = { x: number; y: number; size: number };

export type CircularSliderProps = {
  /** Current position [0, 1]. In equal mode this is visual position. */
  value: number;
  /** Called on each drag move (immediate position update). */
  onChange: (value: number) => void;
  /** Called on initial click/tap (animated seek). Falls back to onChange if omitted. */
  onSeek?: (value: number) => void;
  /** Called on first drag move after mousedown (lets parent cancel animations). */
  onDragStart?: () => void;
  /** Number of tick marks around the ring (typically = number of tour views). */
  tickCount?: number;
  /** SVG diameter in px. Default 200. */
  size?: number;
  /** Cumulative arc-lengths for variable-width ring and geodesic tick positions. */
  arcLengths?: Float32Array | null;
  /** Slider spacing mode. Default 'equal'. */
  spacingMode?: 'equal' | 'geodesic';
  /** Index of the keyframe nearest to the current tour position, or null. */
  currentKeyframe?: number | null;
  /** Index of the gallery preview being hovered, or null. */
  hoveredKeyframe?: number | null;
  /** Preview center positions relative to the container center, with sizes. */
  previewCenters?: PreviewCenter[];
};

const START_DEG = -135;
const BASE_STROKE = 3;
const MIN_STROKE = 1;
const MAX_STROKE = 7;

const TICK_LEN = 5; // normal tick length (outward from ring)
const TICK_GAP = 3; // gap between ring and tick start
const ACTIVE_EXTRA = 4; // extra length for active tick
const ACTIVE_WIDTH = 4; // stroke width for active tick
const CP1_OFFSET = 24; // bezier control point 1 offset along radial direction

export const CircularSlider = forwardRef<CircularSliderHandle, CircularSliderProps>(
  (
    {
      value,
      onChange,
      onSeek,
      onDragStart,
      tickCount = 8,
      size = 200,
      arcLengths,
      spacingMode = 'equal',
      currentKeyframe,
      hoveredKeyframe,
      previewCenters,
    },
    ref,
  ) => {
    const [isDragging, setIsDragging] = useState(false);
    const hasDraggedRef = useRef(false);
    const svgRef = useRef<SVGSVGElement>(null);
    const handleCircleRef = useRef<SVGCircleElement>(null);
    const hitAreaRef = useRef<SVGCircleElement>(null);
    const arcRef = useRef<SVGPathElement>(null);
    const prevValueRef = useRef(value);

    const center = size / 2;
    const radius = size * 0.4;
    const startRad = (START_DEG * Math.PI) / 180;
    const startX = center + radius * Math.cos(startRad);
    const startY = center + radius * Math.sin(startRad);

    /** Compute the angle (radians) for a given keyframe index. */
    const tickRad = useCallback(
      (i: number): number => {
        let tickFraction: number;
        if (spacingMode === 'geodesic' && arcLengths && i < arcLengths.length) {
          tickFraction = arcLengths[i]!;
        } else {
          tickFraction = i / tickCount;
        }
        return ((tickFraction * 360 + START_DEG) * Math.PI) / 180;
      },
      [spacingMode, arcLengths, tickCount],
    );

    /** Update SVG DOM elements directly — no React re-render needed. */
    const updateDom = useCallback(
      (val: number) => {
        const isWrapping = Math.abs(val - prevValueRef.current) > 0.5;
        prevValueRef.current = val;

        const handleRad = ((val * 360 + START_DEG) * Math.PI) / 180;
        const hx = center + radius * Math.cos(handleRad);
        const hy = center + radius * Math.sin(handleRad);

        handleCircleRef.current?.setAttribute('cx', String(hx));
        handleCircleRef.current?.setAttribute('cy', String(hy));
        hitAreaRef.current?.setAttribute('cx', String(hx));
        hitAreaRef.current?.setAttribute('cy', String(hy));

        if (val > 0.001 && !isWrapping) {
          const largeArc = val > 0.5 ? 1 : 0;
          arcRef.current?.setAttribute(
            'd',
            `M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${hx} ${hy}`,
          );
          arcRef.current?.removeAttribute('display');
        } else {
          arcRef.current?.setAttribute('display', 'none');
        }
      },
      [center, radius, startX, startY],
    );

    // Expose imperative handle for parent to update position without re-renders
    useImperativeHandle(ref, () => ({ setPosition: updateDom }), [updateDom]);

    // Sync DOM when value prop changes (debounced atom updates, seek animations)
    useEffect(() => {
      updateDom(value);
    }, [value, updateDom]);

    const angleFromPointer = useCallback(
      (clientX: number, clientY: number): number => {
        if (!svgRef.current) return 0;
        const rect = svgRef.current.getBoundingClientRect();
        const dx = clientX - (rect.left + center);
        const dy = clientY - (rect.top + center);
        // atan2 gives angle from +x axis; rotate so 0 is at 10:30 position, clockwise
        // 10:30 = -135° from +x axis, so offset by +135°
        let deg = (Math.atan2(dy, dx) * 180) / Math.PI + 135;
        if (deg < 0) deg += 360;
        if (deg >= 360) deg -= 360;
        return deg / 360; // normalize to [0, 1]
      },
      [center],
    );

    const handlePointerDown = useCallback(
      (e: React.MouseEvent | React.TouchEvent) => {
        setIsDragging(true);
        hasDraggedRef.current = false;
        const pt = 'touches' in e ? e.touches[0] : e;
        if (pt) {
          const pos = angleFromPointer(pt.clientX, pt.clientY);
          // Animated seek on click; falls back to immediate onChange
          (onSeek ?? onChange)(pos);
        }
      },
      [angleFromPointer, onChange, onSeek],
    );

    useEffect(() => {
      if (!isDragging) return;

      const onMove = (e: MouseEvent | TouchEvent) => {
        if (!hasDraggedRef.current) {
          hasDraggedRef.current = true;
          onDragStart?.(); // let parent cancel any running animation
        }
        const pt = 'touches' in e ? e.touches[0] : e;
        if (pt) onChange(angleFromPointer(pt.clientX, pt.clientY));
        if ('touches' in e) e.preventDefault();
      };
      const onUp = () => setIsDragging(false);

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
      document.addEventListener('touchmove', onMove, { passive: false });
      document.addEventListener('touchend', onUp);

      return () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend', onUp);
      };
    }, [isDragging, angleFromPointer, onChange, onDragStart]);

    // Initial positions from value prop (first paint; updateDom takes over after mount)
    const handleRad = ((value * 360 + START_DEG) * Math.PI) / 180;
    const handleX = center + radius * Math.cos(handleRad);
    const handleY = center + radius * Math.sin(handleRad);

    // Tick marks — outward facing, positioned at arc-length fractions (geodesic)
    // or evenly spaced (equal). Active tick is longer and wider.
    const ticks = useMemo(() => {
      return Array.from({ length: tickCount }, (_, i) => {
        const angle = tickRad(i);
        const isActive = i === currentKeyframe;
        const extra = isActive ? ACTIVE_EXTRA : 0;
        const r1 = radius + TICK_GAP;
        const r2 = radius + TICK_GAP + TICK_LEN + extra;
        return (
          <line
            // biome-ignore lint/suspicious/noArrayIndexKey: tick key
            key={i}
            x1={center + r1 * Math.cos(angle)}
            y1={center + r1 * Math.sin(angle)}
            x2={center + r2 * Math.cos(angle)}
            y2={center + r2 * Math.sin(angle)}
            stroke={isActive ? 'white' : 'var(--color-dtour-text-muted)'}
            strokeWidth={isActive ? ACTIVE_WIDTH : 2}
          />
        );
      });
    }, [tickCount, tickRad, radius, center, currentKeyframe]);

    // Bezier connector — cubic bezier from tick outward end to preview center.
    // Shown on hover (hoveredKeyframe) or during playback (currentKeyframe).
    const connector = useMemo(() => {
      const kf = hoveredKeyframe ?? currentKeyframe;
      if (kf == null || kf >= tickCount) return null;
      const pc = previewCenters?.[kf];
      if (!pc || (pc.x === 0 && pc.y === 0 && pc.size === 0)) return null;

      const angle = tickRad(kf);
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      // Start: tick outward end
      const tickEndR = radius + TICK_GAP + TICK_LEN + ACTIVE_EXTRA;
      const sx = center + tickEndR * cos;
      const sy = center + tickEndR * sin;

      // Control point 1: 24px further along the radial from center through tick
      const cp1x = center + (tickEndR + CP1_OFFSET) * cos;
      const cp1y = center + (tickEndR + CP1_OFFSET) * sin;

      // End: preview center in SVG coordinates
      const ex = center + pc.x;
      const ey = center + pc.y;

      // Control point 2: from preview center, move toward SVG center by half preview size
      const dx = center - ex;
      const dy = center - ey;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const halfSize = pc.size / 2;
      const cp2x = dist > 1e-6 ? ex + (dx / dist) * halfSize : ex;
      const cp2y = dist > 1e-6 ? ey + (dy / dist) * halfSize : ey;

      return (
        <path
          d={`M ${sx} ${sy} C ${cp1x} ${cp1y} ${cp2x} ${cp2y} ${ex} ${ey}`}
          fill="none"
          stroke="var(--color-dtour-highlight)"
          strokeWidth="1"
          strokeDasharray="3 3"
          opacity="0.4"
          className="pointer-events-none"
        />
      );
    }, [hoveredKeyframe, currentKeyframe, tickCount, tickRad, radius, center, previewCenters]);

    // Variable-width ring segments for equal mode — stroke width proportional
    // to the segment's geodesic length. Thin = small geodesic distance (stretched),
    // thick = large geodesic distance (compressed).
    const ringSegments = useMemo(() => {
      if (spacingMode !== 'equal' || !arcLengths || arcLengths.length < 2) return null;
      const n = arcLengths.length - 1;
      const expected = 1 / n;
      const segments: React.ReactElement[] = [];
      for (let i = 0; i < n; i++) {
        const segLen = arcLengths[i + 1]! - arcLengths[i]!;
        const ratio = expected > 1e-10 ? segLen / expected : 1;
        const strokeW = Math.max(MIN_STROKE, Math.min(MAX_STROKE, BASE_STROKE * ratio));

        const startFrac = i / n;
        const endFrac = (i + 1) / n;
        const a1 = ((startFrac * 360 + START_DEG) * Math.PI) / 180;
        const a2 = ((endFrac * 360 + START_DEG) * Math.PI) / 180;

        const x1 = center + radius * Math.cos(a1);
        const y1 = center + radius * Math.sin(a1);
        const x2 = center + radius * Math.cos(a2);
        const y2 = center + radius * Math.sin(a2);

        const sweep = endFrac - startFrac;
        const largeArc = sweep > 0.5 ? 1 : 0;

        segments.push(
          <path
            key={i}
            d={`M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`}
            fill="none"
            stroke="var(--color-dtour-border)"
            strokeWidth={strokeW}
            className="pointer-events-none"
          />,
        );
      }
      return segments;
    }, [spacingMode, arcLengths, radius, center]);

    return (
      <svg
        ref={svgRef}
        width={size}
        height={size}
        overflow="visible"
        onMouseDown={handlePointerDown}
        onTouchStart={handlePointerDown}
        className="pointer-events-none select-none touch-none"
      >
        <title>Circular slider for Dtour</title>
        {/* Transparent hit area for track ring — wider for easier clicking */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="transparent"
          strokeWidth="20"
          className="cursor-pointer pointer-events-auto"
        />
        {/* Track ring — variable width in equal mode, constant in geodesic */}
        {ringSegments ?? (
          <circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke="var(--color-dtour-border)"
            strokeWidth="3"
            className="pointer-events-none"
          />
        )}
        {/* Ticks (outward facing) */}
        {ticks}
        {/* Bezier connector — from tick to preview center */}
        {connector}
        {/* Arc showing position — always rendered, visibility controlled imperatively */}
        <path
          ref={arcRef}
          d={`M ${startX} ${startY} A ${radius} ${radius} 0 ${value > 0.5 ? 1 : 0} 1 ${handleX} ${handleY}`}
          display={value > 0.001 ? undefined : 'none'}
          fill="none"
          stroke="var(--color-dtour-highlight)"
          strokeWidth="4"
          strokeLinecap="round"
          className="pointer-events-none"
        />
        {/* Center dot */}
        <circle
          cx={center}
          cy={center}
          r="3"
          fill="var(--color-dtour-text-muted)"
          className="pointer-events-none"
        />
        {/* Transparent hit area for handle — larger for easier grabbing */}
        <circle
          ref={hitAreaRef}
          cx={handleX}
          cy={handleY}
          r="16"
          fill="transparent"
          className={cn(
            'cursor-grab pointer-events-auto',
            isDragging ? 'cursor-grabbing' : 'cursor-grab',
          )}
        />
        {/* Handle */}
        <circle
          ref={handleCircleRef}
          cx={handleX}
          cy={handleY}
          r="8"
          fill="var(--color-dtour-highlight)"
          stroke="var(--color-dtour-bg)"
          strokeWidth="2"
          className="pointer-events-none"
        />
      </svg>
    );
  },
);
