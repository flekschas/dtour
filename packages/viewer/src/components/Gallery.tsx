import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { computeGalleryPositions } from '../layout/gallery-positions.ts';
import { cn } from '../lib/utils.ts';
import {
  selectedKeyframeAtom,
  tourPlayingAtom,
  tourPositionAtom,
  viewCountAtom,
} from '../state/atoms.ts';

/** 360° of travel = 1000ms animation */
const MS_PER_FULL_ROTATION = 1000;
/** Minimum animation duration to keep it perceptible */
const MIN_ANIMATION_MS = 80;

export type GalleryProps = {
  /** Fixed pool of preview canvas elements (created at scatter init). */
  previewCanvases: HTMLCanvasElement[];
  /** Container width (px). */
  containerWidth: number;
  /** Container height (px). */
  containerHeight: number;
};

export const Gallery = ({ previewCanvases, containerWidth, containerHeight }: GalleryProps) => {
  const viewCount = useAtomValue(viewCountAtom);
  const position = useAtomValue(tourPositionAtom);
  const [selectedKeyframe, setSelectedKeyframe] = useAtom(selectedKeyframeAtom);
  const setPosition = useSetAtom(tourPositionAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);
  const animRef = useRef<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const { positions } = useMemo(
    () => computeGalleryPositions(containerWidth, containerHeight, viewCount),
    [containerWidth, containerHeight, viewCount],
  );

  // Adopt each canvas into its wrapper div (once, on mount)
  useEffect(() => {
    for (let i = 0; i < previewCanvases.length; i++) {
      const wrapper = wrapperRefs.current[i];
      const canvas = previewCanvases[i];
      if (wrapper && canvas && canvas.parentElement !== wrapper) {
        wrapper.appendChild(canvas);
      }
    }
  }, [previewCanvases]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animRef.current !== null) cancelAnimationFrame(animRef.current);
    };
  }, []);

  const currentKeyframe = Math.round(position * viewCount) % viewCount;

  const getBorderColor = (i: number): string => {
    if (i === hoveredIndex) return '#ffffff';
    if (i === selectedKeyframe) return '#e8a040';
    if (i === currentKeyframe) return '#4080e8';
    return '#2a2a3a'; // matches toolbar border (dtour-border)
  };

  const getBoxShadow = (i: number): string => {
    if (i === selectedKeyframe) return '0 0 8px rgba(232, 160, 64, 0.3)';
    if (i === currentKeyframe) return '0 0 8px rgba(64, 128, 232, 0.3)';
    if (i === hoveredIndex) return '0 0 6px rgba(255, 255, 255, 0.15)';
    return 'none';
  };

  const handleClick = useCallback(
    (i: number) => {
      // Cancel any running animation
      if (animRef.current !== null) {
        cancelAnimationFrame(animRef.current);
        animRef.current = null;
      }

      setSelectedKeyframe(i);
      setPlaying(false);

      const target = i / viewCount;

      // Read current position via the atom getter callback
      setPosition((current) => {
        // Shortest angular distance on [0,1) circle
        let delta = target - current;
        if (delta > 0.5) delta -= 1;
        if (delta < -0.5) delta += 1;

        const absDelta = Math.abs(delta);
        const durationMs = Math.max(MIN_ANIMATION_MS, absDelta * MS_PER_FULL_ROTATION);

        // If already there, no animation needed
        if (absDelta < 0.001) return target;

        // Start animation from current
        const startPos = current;
        const startTime = performance.now();

        const tick = (now: number) => {
          const elapsed = now - startTime;
          const t = Math.min(1, elapsed / durationMs);
          // Ease-out cubic
          const eased = 1 - (1 - t) * (1 - t) * (1 - t);

          let pos = startPos + delta * eased;
          // Wrap to [0, 1)
          pos = pos - Math.floor(pos);

          setPosition(pos);

          if (t < 1) {
            animRef.current = requestAnimationFrame(tick);
          } else {
            animRef.current = null;
          }
        };

        animRef.current = requestAnimationFrame(tick);

        // Return current unchanged — the rAF will drive updates
        return current;
      });
    },
    [viewCount, setSelectedKeyframe, setPlaying, setPosition],
  );

  return (
    <>
      {previewCanvases.map((_, i) => {
        const pos = i < positions.length ? positions[i]! : null;

        return (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: fixed pool keyed by slot index
            key={i}
            ref={(el) => {
              wrapperRefs.current[i] = el;
            }}
            onClick={pos ? () => handleClick(i) : undefined}
            onKeyDown={undefined}
            onMouseEnter={pos ? () => setHoveredIndex(i) : undefined}
            onMouseLeave={pos ? () => setHoveredIndex(null) : undefined}
            className={cn(
              'absolute overflow-hidden border-2 border-solid rounded-sm transition-[border-color,box-shadow] duration-200 ease-in-out',
              pos ? 'block cursor-pointer' : 'hidden',
            )}
            style={{
              left: pos ? pos.x - pos.size / 2 : 0,
              top: pos ? pos.y - pos.size / 2 : 0,
              width: pos?.size ?? 0,
              height: pos?.size ?? 0,
              borderColor: getBorderColor(i),
              boxShadow: getBoxShadow(i),
            }}
          />
        );
      })}
    </>
  );
};
