import { ArrowsLeftRightIcon, EqualsIcon } from '@phosphor-icons/react';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { type ReactNode, useCallback, useEffect, useMemo, useRef } from 'react';
import { useAnimatePosition } from '../hooks/useAnimatePosition.ts';
import {
  computeGallerySizes,
  computeLayout,
  LOADING_BAR_HEIGHT,
  PREVIEW_SPACING,
} from '../layout/gallery-positions.ts';
import { cn } from '../lib/utils.ts';
import type { KeyframeLoading } from '../spec.ts';
import {
  arcLengthsAtom,
  currentKeyframeAtom,
  hoveredKeyframeAtom,
  keyframeCountAtom,
  keyframeDescriptionsAtom,
  keyframeLoadingsAtom,
  previewCentersAtom,
  previewKeyframesAtom,
  resolvedPreviewCountAtom,
  resolvedPreviewKeyframeNumbersAtom,
  resolvedPreviewLabelContentAtom,
  resolvedPreviewLabelVisibilityAtom,
  resolvedPreviewScaleAtom,
  tourPlayingAtom,
} from '../state/atoms.ts';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip.tsx';

export type GalleryProps = {
  /** Preview canvases in preview-slot order. Each one shows the keyframe at the same slot. */
  previewCanvases: HTMLCanvasElement[];
  /** Container width (px). */
  containerWidth: number;
  /** Container height (px). */
  containerHeight: number;
  /** Effective toolbar height in px (0 when hidden). */
  toolbarHeight: number;
  /** Called when a preview is clicked; should animate back to guided mode. */
  onResumeGuided: (durationMs: number) => void;
};

// ---------------------------------------------------------------------------
// Loading pill helpers
// ---------------------------------------------------------------------------

/** Whether the primary and secondary loadings have the same sign (co-vary vs contrast). */
function sameSign(loading: KeyframeLoading): boolean {
  return loading.primary[1] * loading.secondary[1] >= 0;
}

/** Resolve a keyframe description for index `i` given the descriptions and loading. */
function resolveDescription(
  descriptions: string | string[] | null,
  loading: KeyframeLoading | null,
  i: number,
): string | null {
  if (descriptions === null) return null;
  if (Array.isArray(descriptions)) {
    return i < descriptions.length ? descriptions[i]! : null;
  }
  // Template string — requires loading data
  if (!loading) return null;
  const same = sameSign(loading);
  return descriptions
    .replace('{primary}', loading.primary[0])
    .replace('{secondary}', loading.secondary[0])
    .replace('{relation}', same ? 'co-varying' : 'contrasting');
}

export const Gallery = ({
  previewCanvases,
  containerWidth,
  containerHeight,
  toolbarHeight,
  onResumeGuided,
}: GalleryProps) => {
  const previewCount = useAtomValue(resolvedPreviewCountAtom);
  const previewKeyframes = useAtomValue(previewKeyframesAtom);
  const keyframeCount = useAtomValue(keyframeCountAtom);
  const previewScale = useAtomValue(resolvedPreviewScaleAtom);
  const currentKeyframe = useAtomValue(currentKeyframeAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const arcLengths = useAtomValue(arcLengthsAtom);
  const [hoveredKeyframe, setHoveredKeyframe] = useAtom(hoveredKeyframeAtom);
  const showKeyframeNumbers = useAtomValue(resolvedPreviewKeyframeNumbersAtom) === 'visible';
  const labelContent = useAtomValue(resolvedPreviewLabelContentAtom);
  const labelVisibility = useAtomValue(resolvedPreviewLabelVisibilityAtom);
  const keyframeLoadings = useAtomValue(keyframeLoadingsAtom);
  const keyframeDescriptions = useAtomValue(keyframeDescriptionsAtom);
  const setPreviewCenters = useSetAtom(previewCentersAtom);
  const { animateTo } = useAnimatePosition();
  const galleryRef = useRef<HTMLDivElement>(null);
  const wrapperRefs = useRef<(HTMLDivElement | null)[]>([]);

  const loadingsVisible = labelVisibility !== 'hidden' && labelContent === 'loadings';
  const descriptionsVisible = labelVisibility !== 'hidden' && labelContent === 'description';
  // Only visible labels take up space in the grid
  const showBarSpace = labelVisibility === 'visible';
  const labelsInside = labelVisibility === 'interactive';

  // Grid area = container minus its CSS insets.
  const verticalInset = PREVIEW_SPACING + toolbarHeight / 2;
  const gridWidth = containerWidth - PREVIEW_SPACING * 2;
  const gridHeight = containerHeight - PREVIEW_SPACING * 2;

  const { gridTemplateColumns, gridTemplateRows, sizes } = useMemo(
    () => computeGallerySizes(gridWidth, gridHeight, previewCount, previewScale, showBarSpace),
    [gridWidth, gridHeight, previewCount, previewScale, showBarSpace],
  );

  // Keep only the current preview canvas in each wrapper. React runs all effect
  // cleanups before any effect setup, so a canvas that was already replaced
  // can still have been adopted.
  useEffect(() => {
    for (let i = 0; i < previewCanvases.length; i++) {
      const wrapper = wrapperRefs.current[i];
      const canvas = previewCanvases[i];
      if (!wrapper || !canvas) continue;
      for (const child of wrapper.querySelectorAll(':scope > canvas')) {
        if (child !== canvas) child.remove();
      }
      if (canvas.parentElement !== wrapper) wrapper.appendChild(canvas);
    }
  }, [previewCanvases]);

  // Without a gallery there are no previews for the slider to point at
  useEffect(() => () => setPreviewCenters([]), [setPreviewCenters]);

  // Measure preview center positions relative to the container center.
  const canvasCount = previewCanvases.length;
  useEffect(() => {
    const galleryEl = galleryRef.current;
    if (!galleryEl || canvasCount < previewCount) return;
    const galleryRect = galleryEl.getBoundingClientRect();
    const centers: { x: number; y: number; size: number }[] = [];
    for (let i = 0; i < previewCount; i++) {
      const keyframe = previewKeyframes[i]!;
      const wrapper = wrapperRefs.current[i];
      if (!wrapper) {
        centers[keyframe] = { x: 0, y: 0, size: sizes[i] ?? 0 };
        continue;
      }
      const r = wrapper.getBoundingClientRect();
      const cx = r.left - galleryRect.left + r.width / 2;
      const cy = r.top - galleryRect.top + r.height / 2;
      centers[keyframe] = {
        x: cx + 16 - containerWidth / 2,
        y: cy + verticalInset - containerHeight / 2,
        size: sizes[i] ?? r.width,
      };
    }
    setPreviewCenters(centers);
  }, [
    containerWidth,
    containerHeight,
    previewCount,
    previewKeyframes,
    canvasCount,
    sizes,
    verticalInset,
    setPreviewCenters,
  ]);

  const getBorderColor = (keyframe: number): string | undefined => {
    if (keyframe === currentKeyframe || keyframe === hoveredKeyframe) {
      return 'var(--color-dtour-highlight)';
    }
    return undefined;
  };

  const getBoxShadow = (keyframe: number): string => {
    if (keyframe === currentKeyframe)
      return '0 0 8px color-mix(in srgb, var(--color-dtour-highlight) 30%, transparent)';
    if (keyframe === hoveredKeyframe) return '0 0 6px rgba(255, 255, 255, 0.15)';
    return 'none';
  };

  const handleClick = useCallback(
    (keyframe: number) => {
      onResumeGuided(300);
      setPlaying(false);
      const target =
        arcLengths && keyframe < arcLengths.length
          ? arcLengths[keyframe]!
          : keyframe / keyframeCount;
      animateTo(target);
    },
    [keyframeCount, arcLengths, setPlaying, onResumeGuided, animateTo],
  );

  const layout = useMemo(() => computeLayout(previewCount), [previewCount]);

  return (
    <div
      ref={galleryRef}
      className="absolute left-2 right-2 grid gap-4 justify-between content-between pointer-events-none"
      style={{ top: verticalInset, bottom: verticalInset, gridTemplateColumns, gridTemplateRows }}
    >
      {previewCanvases.map((_, i) => {
        const visible = i < previewCount;
        const keyframe = previewKeyframes[i] ?? i;

        const pos = layout.positions[i];
        const col = pos?.col ?? 0;
        const row = pos?.row ?? 0;

        const verticalAlignment =
          row === 0 ? 'items-start' : row < layout.rows - 1 ? 'items-center' : 'items-end';
        const horizontalAlignment =
          col === 0 ? 'justify-start' : col < layout.cols - 1 ? 'justify-center' : 'justify-end';

        // For bottom-edge previews, put the label above (flex-col-reverse)
        const isBottomEdge = row === layout.rows - 1;
        const isHighlighted = keyframe === currentKeyframe || keyframe === hoveredKeyframe;
        const loading: KeyframeLoading | null =
          loadingsVisible && keyframeLoadings && keyframe < keyframeLoadings.length
            ? keyframeLoadings[keyframe]!
            : null;
        const keyframeDescription =
          !loading && descriptionsVisible && Array.isArray(keyframeDescriptions)
            ? (keyframeDescriptions[keyframe] ?? null)
            : null;
        const hasLabelBelow = !labelsInside && (loading !== null || keyframeDescription !== null);

        // Visible labels attach to the outside of the preview; interactive
        // labels sit inside along the same edge.
        const labelClassName = labelsInside
          ? cn(
              'absolute inset-x-0 z-10',
              isBottomEdge ? 'top-0' : 'bottom-0',
              isHighlighted ? 'opacity-100' : 'opacity-0 pointer-events-none',
            )
          : cn(
              'relative z-20 border border-dtour-border',
              isBottomEdge ? 'border-b-0 rounded-t-sm' : 'border-t-0 rounded-b-sm',
            );
        const labelStyle = {
          width: labelsInside ? undefined : sizes[i],
          height: LOADING_BAR_HEIGHT,
          borderColor: labelsInside ? undefined : getBorderColor(keyframe),
          backgroundColor: isHighlighted
            ? 'var(--color-dtour-highlight)'
            : 'var(--color-dtour-border)',
        };
        const labelTextClassName = isHighlighted ? 'text-dtour-bg' : 'text-dtour-highlight/70';

        let label: ReactNode = null;
        if (visible && loading) {
          // Loading pills: [primary] [≠ or =] [secondary]
          const n0 = loading.primary[0];
          const n1 = loading.secondary[0];
          const same = sameSign(loading);
          const tooltipText = resolveDescription(keyframeDescriptions, loading, keyframe);
          label = (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div
                    className={cn(
                      'flex items-center cursor-default select-none transition-[color,background-color,border-color,opacity] duration-200',
                      labelClassName,
                    )}
                    style={labelStyle}
                  >
                    <div className="flex-1 flex items-center justify-center rounded-l-sm overflow-hidden h-full">
                      <span
                        className={cn(
                          'text-[10px] transition-colors duration-200 truncate px-1',
                          labelTextClassName,
                        )}
                      >
                        {n0}
                      </span>
                    </div>
                    <span
                      className={cn(
                        'text-[10px] leading-none transition-colors duration-200 px-0.5 shrink-0',
                        labelTextClassName,
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
                          'text-[10px] transition-colors duration-200 truncate px-1',
                          labelTextClassName,
                        )}
                      >
                        {n1}
                      </span>
                    </div>
                  </div>
                </TooltipTrigger>
                <TooltipContent side={isBottomEdge ? 'top' : 'bottom'} sideOffset={0}>
                  {tooltipText ?? `${same ? 'Co-varying' : 'Contrasting'} ${n0} and ${n1}`}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          );
        } else if (visible && keyframeDescription) {
          label = (
            <div
              className={cn(
                'flex items-center justify-center select-none transition-[color,background-color,border-color,opacity] duration-200',
                labelClassName,
              )}
              style={labelStyle}
            >
              <span
                className={cn(
                  'text-[10px] truncate px-1 transition-colors duration-200',
                  labelTextClassName,
                )}
              >
                {keyframeDescription}
              </span>
            </div>
          );
        }

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
              onMouseEnter={visible ? () => setHoveredKeyframe(keyframe) : undefined}
              onMouseLeave={visible ? () => setHoveredKeyframe(null) : undefined}
            >
              <div
                ref={(el) => {
                  wrapperRefs.current[i] = el;
                }}
                onClick={visible ? () => handleClick(keyframe) : undefined}
                onKeyDown={undefined}
                className={cn(
                  'overflow-hidden border-2 border-dtour-border bg-dtour-bg transition-[border-color,box-shadow] duration-200 ease-in-out z-20 relative group',
                  hasLabelBelow ? (isBottomEdge ? 'rounded-b' : 'rounded-t') : 'rounded',
                  visible ? 'block cursor-pointer' : 'hidden',
                )}
                style={{
                  width: visible ? sizes[i] : 0,
                  height: visible ? sizes[i] : 0,
                  borderColor: getBorderColor(keyframe),
                  boxShadow: getBoxShadow(keyframe),
                }}
              >
                {visible && showKeyframeNumbers && (
                  <span
                    className={cn(
                      'absolute z-10 text-xs leading-none text-dtour-text pointer-events-none transition-opacity duration-200',
                      row === 0
                        ? 'top-0.5'
                        : row === layout.rows - 1
                          ? 'bottom-0.5'
                          : 'top-1/2 -translate-y-1/2',
                      col === 0
                        ? 'left-1'
                        : col === layout.cols - 1
                          ? 'right-1'
                          : 'left-1/2 -translate-x-1/2',
                      keyframe === currentKeyframe
                        ? 'opacity-100'
                        : 'opacity-40 group-hover:opacity-100',
                    )}
                  >
                    {keyframe + 1}
                  </span>
                )}
                {labelsInside && label}
              </div>
              {!labelsInside && label}
            </div>
          </div>
        );
      })}
    </div>
  );
};
