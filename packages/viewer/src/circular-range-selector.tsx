import { useCallback, useEffect, useRef, useState } from 'react';

export type CircularSelectorProps = {
  /** Current position [0, 1]. */
  value: number;
  /** Called when the user drags the handle. */
  onChange: (value: number) => void;
  /** Number of tick marks around the ring (typically = number of tour views). */
  tickCount?: number;
  /** SVG diameter in px. Default 200. */
  size?: number;
};

export const CircularSelector = ({
  value,
  onChange,
  tickCount = 8,
  size = 200,
}: CircularSelectorProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const svgRef = useRef<SVGSVGElement>(null);

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
      const pt = 'touches' in e ? e.touches[0] : e;
      if (pt) onChange(angleFromPointer(pt.clientX, pt.clientY));
    },
    [angleFromPointer, onChange],
  );

  useEffect(() => {
    if (!isDragging) return;

    const onMove = (e: MouseEvent | TouchEvent) => {
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
  }, [isDragging, angleFromPointer, onChange]);

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
      style={{ cursor: 'pointer', userSelect: 'none', touchAction: 'none' }}
      onMouseDown={handlePointerDown}
      onTouchStart={handlePointerDown}
    >
      {/* Track ring */}
      <circle cx={center} cy={center} r={radius} fill="none" stroke="#333" strokeWidth="3" />
      {/* Ticks */}
      {ticks}
      {/* Arc showing position */}
      {value > 0.001 && (
        <path
          d={`M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArc} 1 ${handleX} ${handleY}`}
          fill="none"
          stroke="#4f8ff7"
          strokeWidth="4"
          strokeLinecap="round"
        />
      )}
      {/* Center dot */}
      <circle cx={center} cy={center} r="3" fill="#666" />
      {/* Handle */}
      <circle
        cx={handleX}
        cy={handleY}
        r="8"
        fill="#4f8ff7"
        stroke="#1a1a2e"
        strokeWidth="2"
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      />
    </svg>
  );
};
