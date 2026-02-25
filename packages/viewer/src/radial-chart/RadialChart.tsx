import { useMemo } from 'react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../components/ui/tooltip.tsx';
import { arcPath, keyframeAngle } from './arc-path.ts';
import type { ParsedTrack } from './types.ts';

export type RadialChartProps = {
  tracks: ParsedTrack[];
  keyframeCount: number;
  /** Current tour position [0, 1] for flanking highlight. */
  position: number;
  /** SVG viewport size (same as selectorSize). */
  size: number;
  /** Inner radius = selector ring radius (selectorSize * 0.4). */
  innerRadius: number;
};

const TRACK_GAP = 2;
const BAR_PAD_RAD = 0.02; // angular padding between bars in radians

export const RadialChart = ({
  tracks,
  keyframeCount,
  position,
  size,
  innerRadius,
}: RadialChartProps) => {
  const center = size / 2;

  // Compute radial layout for each track (cumulative offsets)
  const trackLayout = useMemo(() => {
    let offset = innerRadius + TRACK_GAP;
    return tracks.map((t) => {
      const rInner = offset;
      const rOuter = offset + t.height;
      offset = rOuter + TRACK_GAP;
      return { rInner, rOuter };
    });
  }, [tracks, innerRadius]);

  // Flanking keyframes based on current position
  const fractionalIndex = position * keyframeCount;
  const leftKf = Math.floor(fractionalIndex) % keyframeCount;
  const rightKf = (leftKf + 1) % keyframeCount;

  const segmentAngle = (2 * Math.PI) / keyframeCount;

  if (import.meta.env.DEV) {
    for (const track of tracks) {
      if (track.normalizedValues.length !== keyframeCount) {
        console.warn(
          `[dtour] RadialChart track "${track.label}" has ${track.normalizedValues.length} values but keyframeCount is ${keyframeCount}. Bars will be clamped to the smaller count.`,
        );
      }
    }
  }

  return (
    <TooltipProvider delayDuration={150}>
      {/* biome-ignore lint/a11y/noSvgWithoutTitle: decorative chart, no screen reader title needed */}
      <svg width={size} height={size} className="overflow-visible pointer-events-none">
        <g transform={`translate(${center}, ${center})`}>
          {tracks.map((track, trackIdx) => {
            const layout = trackLayout[trackIdx]!;
            const { rInner, rOuter } = layout;
            return (
              <g key={track.label}>
                {track.normalizedValues.map((normVal, kfIdx) => {
                  if (kfIdx >= keyframeCount) return null;

                  const centerAngle = keyframeAngle(kfIdx, keyframeCount);
                  let angleStart: number;
                  let angleEnd: number;

                  if (track.barWidth === 'full') {
                    angleStart = centerAngle - segmentAngle / 2 + BAR_PAD_RAD;
                    angleEnd = centerAngle + segmentAngle / 2 - BAR_PAD_RAD;
                  } else {
                    // Convert pixel width to angular width at the midpoint radius
                    const midR = (rInner + rOuter) / 2;
                    const halfAngle = midR > 0 ? track.barWidth / 2 / midR : 0;
                    angleStart = centerAngle - halfAngle;
                    angleEnd = centerAngle + halfAngle;
                  }

                  // Bar grows outward from rInner
                  const barOuter = rInner + normVal * track.height;
                  const isHighlighted = kfIdx === leftKf || kfIdx === rightKf;
                  const rawValue = track.rawValues[kfIdx] as number;

                  return (
                    <Tooltip key={`${track.label}-${kfIdx}`}>
                      <TooltipTrigger asChild>
                        <path
                          d={arcPath(rInner, barOuter, angleStart, angleEnd)}
                          fill={track.color}
                          fillOpacity={0.85}
                          className="pointer-events-auto cursor-default transition-[filter] duration-150 ease-in"
                          style={{
                            filter: isHighlighted
                              ? `drop-shadow(0 0 4px ${track.color})`
                              : undefined,
                          }}
                        />
                      </TooltipTrigger>
                      <TooltipContent>
                        {track.label}: {rawValue.toFixed(3)}
                      </TooltipContent>
                    </Tooltip>
                  );
                })}
              </g>
            );
          })}
        </g>
      </svg>
    </TooltipProvider>
  );
};
