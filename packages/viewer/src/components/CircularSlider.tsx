import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { cn } from '../lib/utils';

export type CircularSliderHandle = {
  /** Update slider position imperatively without triggering a React re-render. */
  setPosition: (value: number) => void;
};

export type CircularSliderProps = {
  /** Current position [0, 1]. */
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
};

export const CircularSlider = forwardRef<CircularSliderHandle, CircularSliderProps>(
  ({ value, onChange, onSeek, onDragStart, tickCount = 8, size = 200 }, ref) => {
    const [isDragging, setIsDragging] = useState(false);
    const hasDraggedRef = useRef(false);
    const svgRef = useRef<SVGSVGElement>(null);
    const handleCircleRef = useRef<SVGCircleElement>(null);
    const hitAreaRef = useRef<SVGCircleElement>(null);
    const arcRef = useRef<SVGPathElement>(null);
    const prevValueRef = useRef(value);

    const center = size / 2;
    const radius = size * 0.4;
    const startDeg = -135;
    const startRad = (startDeg * Math.PI) / 180;
    const startX = center + radius * Math.cos(startRad);
    const startY = center + radius * Math.sin(startRad);

    /** Update SVG DOM elements directly — no React re-render needed. */
    const updateDom = useCallback(
      (val: number) => {
        const isWrapping = Math.abs(val - prevValueRef.current) > 0.5;
        prevValueRef.current = val;

        const handleRad = ((val * 360 + startDeg) * Math.PI) / 180;
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
    const handleRad = ((value * 360 + startDeg) * Math.PI) / 180;
    const handleX = center + radius * Math.cos(handleRad);
    const handleY = center + radius * Math.sin(handleRad);

    // Tick marks (static — don't change with position)
    const ticks = Array.from({ length: tickCount }, (_, i) => {
      const tickRad = (((i / tickCount) * 360 + startDeg) * Math.PI) / 180;
      const r1 = radius - 8;
      const r2 = radius - 3;
      return (
        <line
          // biome-ignore lint/suspicious/noArrayIndexKey: tick key
          key={i}
          x1={center + r1 * Math.cos(tickRad)}
          y1={center + r1 * Math.sin(tickRad)}
          x2={center + r2 * Math.cos(tickRad)}
          y2={center + r2 * Math.sin(tickRad)}
          stroke="var(--color-dtour-text-muted)"
          strokeWidth="2"
        />
      );
    });

    return (
      <svg
        ref={svgRef}
        width={size}
        height={size}
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
        {/* Track ring */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="var(--color-dtour-border)"
          strokeWidth="3"
          className="pointer-events-none"
        />
        {/* Ticks */}
        {ticks}
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
