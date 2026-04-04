import { ArrowsLeftRightIcon, EqualsIcon } from '@phosphor-icons/react';
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
  tourFrameDescriptionAtom,
  tourPlayingAtom,
} from '../state/atoms.ts';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip.tsx';

export type GalleryProps = {
  /** Fixed pool of preview canvas elements (created at scatter init). */
  previewCanvases: HTMLCanvasElement[];
  /** Container width (px). */
  containerWidth: number;
  /** Container height (px). */
  containerHeight: number;
  /** Effective toolbar height in px (0 when hidden). */
  toolbarHeight: number;
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
  toolbarHeight,
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
  const tourFrameDescription = useAtomValue(tourFrameDescriptionAtom);
  const setPreviewCenters = useSetAtom(previewCentersAtom);
  const { animateTo } = useAnimatePosition();
  const galleryRef = useRef<HTMLDivElement>(null);
  const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Whether loading pills are actually visible (data available + user toggle on)
  const loadingsVisible = showFrameLoadings && frameLoadings !== null && frameLoadings.length > 0;

  // Grid area = container minus its CSS insets.
  const verticalInset = 16 + toolbarHeight / 2;
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
        const hasLoadingPills = loadingPairs !== null && loadingPairs.length >= 2;

        return (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: fixed pool keyed by slot index
            key={i}
            className={cn('flex pointer-events-none', verticalAlignment, horizontalAlignment)}
            style={{ gridColumn: col + 1, gridRow: row + 1 }}
          >
            <div
              className={cn(
                'flex pointer-events-auto group/preview',
                isBottomEdge ? 'flex-col-reverse' : 'flex-col',
                visible ? '' : 'hidden',
              )}
              onMouseEnter={visible ? () => setHoveredIndex(i) : undefined}
              onMouseLeave={visible ? () => setHoveredIndex(null) : undefined}
            >
              <div
                ref={(el) => {
                  wrapperRefs.current[i] = el;
                }}
                onClick={visible ? () => handleClick(i) : undefined}
                onKeyDown={undefined}
                className={cn(
                  'overflow-hidden border border-dtour-border transition-[border-color,border-width,box-shadow] duration-200 ease-in-out z-20 relative group',
                  hasLoadingPills ? (isBottomEdge ? 'rounded-b' : 'rounded-t') : 'rounded',
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
                  const [n0] = loadingPairs[0]!;
                  const [n1] = loadingPairs[1]!;
                  const same = sameSign(loadingPairs);
                  const isActive = i === selectedKeyframe || i === currentKeyframe;
                  return (
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div
                            className={cn(
                              'flex items-center cursor-default select-none transition-[color,background-color,border-color] duration-200 relative z-20',
                              'border border-dtour-border',
                              isBottomEdge ? 'border-b-0 rounded-t-sm' : 'border-t-0 rounded-b-sm',
                            )}
                            style={{
                              width: sizes[i],
                              height: LOADING_BAR_HEIGHT,
                              borderColor: getBorderColor(i),
                              backgroundColor:
                                isActive || i === hoveredIndex
                                  ? 'var(--color-dtour-highlight)'
                                  : 'var(--color-dtour-border)',
                            }}
                          >
                            <div className="flex-1 flex items-center justify-center rounded-l-sm overflow-hidden h-full">
                              <span
                                className={cn(
                                  'text-[10px] leading-none transition-colors duration-200 truncate px-1',
                                  isActive || i === hoveredIndex
                                    ? 'text-dtour-bg'
                                    : 'text-dtour-highlight/70',
                                )}
                              >
                                {n0}
                              </span>
                            </div>
                            <span
                              className={cn(
                                'text-[10px] leading-none transition-colors duration-200 px-0.5 shrink-0',
                                isActive || i === hoveredIndex
                                  ? 'text-dtour-bg'
                                  : 'text-dtour-highlight/70',
                              )}
                            >
                              {same ? (
                                <EqualsIcon size={10} weight="bold" />
                              ) : (
                                <ArrowsLeftRightIcon size={10} weight="bold" />
                              )}
                            </span>
                            <div className="flex-1 flex items-center justify-center rounded-r-sm overflow-hidden h-full">
                              <span
                                className={cn(
                                  'text-[10px] leading-none transition-colors duration-200 truncate px-1',
                                  isActive || i === hoveredIndex
                                    ? 'text-dtour-bg'
                                    : 'text-dtour-highlight/70',
                                )}
                              >
                                {n1}
                              </span>
                            </div>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent side={isBottomEdge ? 'top' : 'bottom'} sideOffset={0}>
                          {tourFrameDescription
                            ? tourFrameDescription
                                .replace('{dim1}', n0)
                                .replace('{dim2}', n1)
                                .replace('{relation}', same ? 'co-varying' : 'contrasting')
                            : `${same ? 'Co-varying' : 'Contrasting'} ${n0} and ${n1}`}
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  );
                })()}
            </div>
          </div>
        );
      })}
    </div>
  );
};
