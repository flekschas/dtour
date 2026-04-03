import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useAnimatePosition } from '../hooks/useAnimatePosition.ts';
import { LOADING_BAR_HEIGHT, computeGallerySizes } from '../layout/gallery-positions.ts';
import { cn } from '../lib/utils.ts';
import type { FrameLoading } from '../spec.ts';
import {
  arcLengthsAtom,
  currentKeyframeAtom,
  frameLoadingsAtom,
  guidedSuspendedAtom,
  hoveredKeyframeAtom,
  previewCentersAtom,
  previewCountAtom,
  previewScaleAtom,
  selectedKeyframeAtom,
  showFrameLoadingsAtom,
  showFrameNumbersAtom,
  tourModeAtom,
  tourPlayingAtom,
} from '../state/atoms.ts';

export type GalleryProps = {
  /** Fixed pool of preview canvas elements (created at scatter init). */
  previewCanvases: HTMLCanvasElement[];
  /** Container width (px). */
  containerWidth: number;
  /** Container height (px). */
  containerHeight: number;
  /** Is toolbar visible? */
  isToolbarVisible: boolean;
};

// ---------------------------------------------------------------------------
// Loading pill helpers
// ---------------------------------------------------------------------------

/** Whether the two loadings in a pair have the same sign (co-vary vs contrast). */
function sameSign(pairs: FrameLoading[]): boolean {
  if (pairs.length < 2) return true;
  return pairs[0]![1] * pairs[1]![1] >= 0;
}

export const Gallery = ({
  previewCanvases,
  containerWidth,
  containerHeight,
  isToolbarVisible,
}: GalleryProps) => {
  const previewCount = useAtomValue(previewCountAtom);
  const previewScale = useAtomValue(previewScaleAtom);
  const currentKeyframe = useAtomValue(currentKeyframeAtom);
  const [selectedKeyframe, setSelectedKeyframe] = useAtom(selectedKeyframeAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const arcLengths = useAtomValue(arcLengthsAtom);
  const [hoveredIndex, setHoveredIndex] = useAtom(hoveredKeyframeAtom);
  const showFrameNumbers = useAtomValue(showFrameNumbersAtom);
  const showFrameLoadings = useAtomValue(showFrameLoadingsAtom);
  const frameLoadings = useAtomValue(frameLoadingsAtom);
  const tourMode = useAtomValue(tourModeAtom);
  const setPreviewCenters = useSetAtom(previewCentersAtom);
  const { animateTo } = useAnimatePosition();
  const galleryRef = useRef<HTMLDivElement>(null);
  const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Whether loading pills are actually visible (data available + user toggle on)
  const loadingsVisible = showFrameLoadings && frameLoadings !== null && frameLoadings.length > 0;

  // Compute max absolute loading across all frames for tooltip normalization
  const maxAbsLoading = useMemo(() => {
    if (!frameLoadings) return 0;
    let max = 0;
    for (const frame of frameLoadings) {
      for (const [, val] of frame) {
        max = Math.max(max, Math.abs(val));
      }
    }
    return max;
  }, [frameLoadings]);

  // Grid area = container minus its CSS insets.
  const verticalInset = isToolbarVisible ? 36 : 16;
  const gridWidth = containerWidth - 32;
  const gridHeight = containerHeight - 2 * verticalInset;

  const { gridTemplateColumns, gridTemplateRows, sizes } = useMemo(
    () => computeGallerySizes(gridWidth, gridHeight, previewCount, previewScale, loadingsVisible),
    [gridWidth, gridHeight, previewCount, previewScale, loadingsVisible],
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

  // Measure preview center positions relative to the container center.
  useEffect(() => {
    const galleryEl = galleryRef.current;
    if (!galleryEl) return;
    const galleryRect = galleryEl.getBoundingClientRect();
    const centers: { x: number; y: number; size: number }[] = [];
    for (let i = 0; i < previewCount; i++) {
      const wrapper = wrapperRefs.current[i];
      if (!wrapper) {
        centers.push({ x: 0, y: 0, size: sizes[i] ?? 0 });
        continue;
      }
      const r = wrapper.getBoundingClientRect();
      const cx = r.left - galleryRect.left + r.width / 2;
      const cy = r.top - galleryRect.top + r.height / 2;
      centers.push({
        x: cx + 16 - containerWidth / 2,
        y: cy + verticalInset - containerHeight / 2,
        size: sizes[i] ?? r.width,
      });
    }
    setPreviewCenters(centers);
  }, [containerWidth, containerHeight, previewCount, sizes, verticalInset, setPreviewCenters]);

  const getBorderColor = (i: number): string | undefined => {
    const isActive = i === selectedKeyframe || i === currentKeyframe;
    if (isActive || i === hoveredIndex) return 'var(--color-dtour-highlight)';
    return undefined;
  };

  const getBorderWidth = (i: number): number | undefined => {
    const isActive = i === selectedKeyframe || i === currentKeyframe;
    if (isActive) return 2;
    return undefined;
  };

  const getBoxShadow = (i: number): string => {
    if (i === selectedKeyframe)
      return '0 0 8px color-mix(in srgb, var(--color-dtour-highlight) 30%, transparent)';
    if (i === currentKeyframe)
      return '0 0 8px color-mix(in srgb, var(--color-dtour-highlight) 30%, transparent)';
    if (i === hoveredIndex) return '0 0 6px rgba(255, 255, 255, 0.15)';
    return 'none';
  };

  const handleClick = useCallback(
    (i: number) => {
      setGuidedSuspended(false);
      setSelectedKeyframe(i);
      setPlaying(false);
      const target = arcLengths && i < arcLengths.length ? arcLengths[i]! : i / previewCount;
      animateTo(target);
    },
    [previewCount, arcLengths, setSelectedKeyframe, setPlaying, setGuidedSuspended, animateTo],
  );

  const k = previewCount / 4;
  const modeLabel = tourMode === 'discriminative' ? 'discriminant' : 'structure';

  return (
    <div
      ref={galleryRef}
      className="absolute left-4 right-4 grid gap-8 justify-between content-between pointer-events-none"
      style={{ top: verticalInset, bottom: verticalInset, gridTemplateColumns, gridTemplateRows }}
    >
      {previewCanvases.map((_, i) => {
        const visible = i < previewCount;

        let col: number;
        let row: number;
        if (i < k) {
          row = 0;
          col = i;
        } else if (i < 2 * k) {
          row = i - k;
          col = k;
        } else if (i < 3 * k) {
          row = k;
          col = 3 * k - i;
        } else {
          row = 4 * k - i;
          col = 0;
        }

        const verticalAlignment =
          row === 0 ? 'items-start' : row < k ? 'items-center' : 'items-end';
        const horizontalAlignment =
          col === 0 ? 'justify-start' : col < k ? 'justify-center' : 'justify-end';

        // For bottom-edge previews, put loading bar above (flex-col-reverse)
        const isBottomEdge = row === k;
        const loadingPairs: FrameLoading[] | null =
          loadingsVisible && frameLoadings && i < frameLoadings.length ? frameLoadings[i]! : null;

        return (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: fixed pool keyed by slot index
            key={i}
            className={cn('flex pointer-events-none', verticalAlignment, horizontalAlignment)}
            style={{ gridColumn: col + 1, gridRow: row + 1 }}
          >
            <div
              className={cn(
                'flex pointer-events-none',
                isBottomEdge ? 'flex-col-reverse' : 'flex-col',
                visible ? '' : 'hidden',
              )}
            >
              <div
                ref={(el) => {
                  wrapperRefs.current[i] = el;
                }}
                onClick={visible ? () => handleClick(i) : undefined}
                onKeyDown={undefined}
                onMouseEnter={visible ? () => setHoveredIndex(i) : undefined}
                onMouseLeave={visible ? () => setHoveredIndex(null) : undefined}
                className={cn(
                  'pointer-events-auto overflow-hidden border border-dtour-border rounded transition-[border-color,border-width,box-shadow] duration-200 ease-in-out z-20 relative group',
                  visible ? 'block cursor-pointer' : 'hidden',
                )}
                style={{
                  width: visible ? sizes[i] : 0,
                  height: visible ? sizes[i] : 0,
                  borderColor: getBorderColor(i),
                  borderWidth: getBorderWidth(i),
                  boxShadow: getBoxShadow(i),
                }}
              >
                {visible && showFrameNumbers && (
                  <span
                    className={cn(
                      'absolute text-xs leading-none text-white pointer-events-none transition-opacity duration-200',
                      row === 0 ? 'top-0.5' : row === k ? 'bottom-0.5' : 'top-1/2 -translate-y-1/2',
                      col === 0 ? 'left-1' : col === k ? 'right-1' : 'left-1/2 -translate-x-1/2',
                      i === selectedKeyframe || i === currentKeyframe
                        ? 'opacity-100'
                        : 'opacity-40 group-hover:opacity-100',
                    )}
                  >
                    {i + 1}
                  </span>
                )}
              </div>
              {/* Loading pills: [name1] [≠ or =] [name2] */}
              {visible &&
                loadingPairs &&
                loadingPairs.length >= 2 &&
                (() => {
                  const [n0, v0] = loadingPairs[0]!;
                  const [n1, v1] = loadingPairs[1]!;
                  const same = sameSign(loadingPairs);
                  const pct0 =
                    maxAbsLoading > 0 ? Math.round((Math.abs(v0) / maxAbsLoading) * 100) : 0;
                  const pct1 =
                    maxAbsLoading > 0 ? Math.round((Math.abs(v1) / maxAbsLoading) * 100) : 0;
                  const relation = same ? 'co-vary' : 'contrast';
                  return (
                    <div
                      className="flex items-center pointer-events-auto cursor-default select-none"
                      style={{ width: sizes[i], height: LOADING_BAR_HEIGHT }}
                    >
                      <div
                        className="flex-1 flex items-center justify-center rounded-l-sm overflow-hidden bg-white/10 h-full"
                        title={`${n0}: ${pct0}% relative importance (top ${modeLabel} loading)`}
                      >
                        <span className="text-[10px] leading-none text-white truncate px-1">
                          {n0}
                        </span>
                      </div>
                      <span
                        className="text-[10px] leading-none text-white/40 px-0.5 shrink-0"
                        title={`${n0} and ${n1} ${relation} along this ${modeLabel} axis`}
                      >
                        {same ? '=' : '≠'}
                      </span>
                      <div
                        className="flex-1 flex items-center justify-center rounded-r-sm overflow-hidden bg-white/10 h-full"
                        title={`${n1}: ${pct1}% relative importance (top ${modeLabel} loading)`}
                      >
                        <span className="text-[10px] leading-none text-white truncate px-1">
                          {n1}
                        </span>
                      </div>
                    </div>
                  );
                })()}
            </div>
          </div>
        );
      })}
    </div>
  );
};
