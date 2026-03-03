import { useCallback, useEffect, useRef, useState } from 'react';
import { cn } from '../lib/utils';

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

export const CircularSlider = ({
  value,
  onChange,
  onSeek,
  onDragStart,
  tickCount = 8,
  size = 200,
}: CircularSliderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const hasDraggedRef = useRef(false);
  const svgRef = useRef<SVGSVGElement>(null);
  const prevValue = useRef(value);

  // Detect wrap-around (e.g. 0.98 → 0.02) to suppress arc flicker
  const isWrapping = Math.abs(value - prevValue.current) > 0.5;
  prevValue.current = value;

  const center = size / 2;
  const radius = size * 0.4;

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

  // Convert value [0,1] back to handle position
  // 10:30 start = -135° from +x axis
  const startDeg = -135;
  const handleRad = ((value * 360 + startDeg) * Math.PI) / 180;
  const handleX = center + radius * Math.cos(handleRad);
  const handleY = center + radius * Math.sin(handleRad);

  // Arc from 10:30 to handle
  const startRad = (startDeg * Math.PI) / 180;
  const startX = center + radius * Math.cos(startRad);
  const startY = center + radius * Math.sin(startRad);
  const largeArc = value > 0.5 ? 1 : 0;

  // Tick marks
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
        stroke="#555"
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
        stroke="#333"
        strokeWidth="3"
        className="pointer-events-none"
      />
      {/* Ticks */}
      {ticks}
      {/* Arc showing position */}
      {value > 0.001 && !isWrapping && (
        <path
          d={`M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${handleX} ${handleY}`}
          fill="none"
          stroke="#e8a040"
          strokeWidth="4"
          strokeLinecap="round"
          className="pointer-events-none"
        />
      )}
      {/* Center dot */}
      <circle
        cx={center}
        cy={center}
        r="3"
        fill="#666"
        className="pointer-events-none"
      />
      {/* Transparent hit area for handle — larger for easier grabbing */}
      <circle
        cx={handleX}
        cy={handleY}
        r="16"
        fill="transparent"
        className={cn("cursor-grab pointer-events-auto", isDragging ? 'cursor-grabbing' : 'cursor-grab')}
      />
      {/* Handle */}
      <circle
        cx={handleX}
        cy={handleY}
        r="8"
        fill="#e8a040"
        stroke="#1a1a2e"
        strokeWidth="2"
        className="pointer-events-none"
      />
    </svg>
  );
};
