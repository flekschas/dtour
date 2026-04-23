import { useCallback, useMemo, useState } from 'react';
import { tourToVisual } from '../lib/position-remap.ts';
import { arcPath, keyframeAngle, rectBarPath } from './arc-path.ts';
import type { ParsedTrack } from './types.ts';

const START_DEG = -135;

export type RadialChartProps = {
  tracks: ParsedTrack[];
  keyframeCount: number;
  /** Current tour position [0, 1] for flanking highlight. */
  position: number;
  /** SVG viewport size (same as selectorSize). */
  size: number;
  /** Inner radius = selector ring radius (selectorSize * 0.4). */
  innerRadius: number;
  /** Cumulative arc-lengths for geodesic tick positioning. */
  arcLengths?: Float32Array | null;
  /** Slider spacing mode. Default 'equal'. */
  spacingMode?: 'equal' | 'geodesic';
};

const TRACK_GAP = 2;
const STACK_GAP = 1;
const BAR_PAD_RAD = 0.02; // angular padding between bars in radians

type HoverInfo = { label: string; value: number; x: number; y: number };

export const RadialChart = ({
  tracks,
  keyframeCount,
  position,
  size,
  innerRadius,
  arcLengths,
  spacingMode = 'equal',
}: RadialChartProps) => {
  const center = size / 2;
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const handleEnter = useCallback((label: string, value: number, x: number, y: number) => {
    setHover({ label, value, x, y });
  }, []);

  const handleLeave = useCallback(() => {
    setHover(null);
  }, []);

  const stacked = tracks.length > 0 && tracks.every((t) => t.barWidth !== 'full');

  // Track-based layout: each track gets its own radial band
  const trackLayout = useMemo(() => {
    if (stacked) return [];
    let offset = innerRadius + TRACK_GAP;
    return tracks.map((t) => {
      const rInner = offset;
      const rOuter = offset + t.height;
      offset = rOuter + TRACK_GAP;
      return { rInner, rOuter };
    });
  }, [tracks, innerRadius, stacked]);

  // Compute angle for keyframe index, respecting spacing mode.
  // In geodesic mode, bars sit at arc-length positions; in equal mode, uniform.
  const getAngle = useCallback(
    (index: number): number => {
      if (spacingMode === 'geodesic' && arcLengths && index < arcLengths.length) {
        return ((arcLengths[index]! * 360 + START_DEG) * Math.PI) / 180;
      }
      return keyframeAngle(index, keyframeCount);
    },
    [spacingMode, arcLengths, keyframeCount],
  );

  // Angular span for a segment from keyframe i to i+1
  const getSegmentSpan = useCallback(
    (index: number): number => {
      if (spacingMode === 'geodesic' && arcLengths && arcLengths.length > 1) {
        const n = arcLengths.length - 1;
        const start = arcLengths[index]!;
        const end = arcLengths[(index + 1) % (n + 1)]!;
        const span = end > start ? end - start : 1 - start + end;
        return span * 2 * Math.PI;
      }
      return (2 * Math.PI) / keyframeCount;
    },
    [spacingMode, arcLengths, keyframeCount],
  );

  // Flanking keyframes based on current position
  const visualPos =
    spacingMode === 'equal' && arcLengths ? tourToVisual(position, arcLengths) : position;
  const fractionalIndex = visualPos * keyframeCount;
  const leftKf = Math.floor(fractionalIndex) % keyframeCount;
  const rightKf = (leftKf + 1) % keyframeCount;

  const segmentAngle = (2 * Math.PI) / keyframeCount;
  const baseR = innerRadius + STACK_GAP;

  if (import.meta.env.DEV) {
    for (const track of tracks) {
      if (track.normalizedValues.length !== keyframeCount) {
        console.warn(
          `[dtour] RadialChart track "${track.label}" has ${track.normalizedValues.length} values but keyframeCount is ${keyframeCount}. Bars will be clamped to the smaller count.`,
        );
      }
    }
  }

  // Stacked mode: bars share a common baseline per keyframe, stacked outward
  const stackedBars = useMemo(() => {
    if (!stacked) return [];
    const bars: {
      key: string;
      d: string;
      color: string;
      label: string;
      rawValue: number;
      tipX: number;
      tipY: number;
    }[] = [];

    const barWidthPx = (tracks[0]?.barWidth as number) ?? 0;

    for (let kfIdx = 0; kfIdx < keyframeCount; kfIdx++) {
      const centerAngle = getAngle(kfIdx);

      let stackBase = baseR;
      for (let trackIdx = 0; trackIdx < tracks.length; trackIdx++) {
        const track = tracks[trackIdx]!;
        if (kfIdx >= track.normalizedValues.length) continue;

        const normVal = track.normalizedValues[kfIdx]!;
        const barHeight = normVal * track.height;
        const barOuter = stackBase + barHeight;
        const rawValue = track.rawValues[kfIdx] as number;

        const midR = (stackBase + barOuter) / 2;
        const tipX = center + midR * Math.cos(centerAngle);
        const tipY = center + midR * Math.sin(centerAngle);

        bars.push({
          key: `${track.label}-${kfIdx}`,
          d: rectBarPath(stackBase, barOuter, centerAngle, barWidthPx),
          color: track.color,
          label: track.label,
          rawValue,
          tipX,
          tipY,
        });

        stackBase = barOuter + STACK_GAP;
      }
    }
    return bars;
  }, [stacked, tracks, keyframeCount, baseR, center, getAngle]);

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* biome-ignore lint/a11y/noSvgWithoutTitle: decorative chart, no screen reader title needed */}
      <svg width={size} height={size} className="overflow-visible pointer-events-none">
        <g transform={`translate(${center}, ${center})`}>
          {stacked
            ? stackedBars.map((bar) => (
                <path
                  key={bar.key}
                  d={bar.d}
                  fill={bar.color}
                  className="pointer-events-auto cursor-default opacity-60 hover:opacity-100 transition-[fill-opacity] duration-150 ease-out"
                  onMouseEnter={() => handleEnter(bar.label, bar.rawValue, bar.tipX, bar.tipY)}
                  onMouseLeave={handleLeave}
                />
              ))
            : tracks.map((track, trackIdx) => {
                const { rInner } = trackLayout[trackIdx]!;
                return (
                  <g key={track.label}>
                    {track.normalizedValues.map((normVal, kfIdx) => {
                      if (kfIdx >= keyframeCount) return null;

                      const centerAngle = getAngle(kfIdx);
                      const halfSpan = getSegmentSpan(kfIdx) / 2;
                      const angleStart = centerAngle - halfSpan + BAR_PAD_RAD;
                      const angleEnd = centerAngle + halfSpan - BAR_PAD_RAD;

                      const barOuter = rInner + normVal * track.height;
                      const rawValue = track.rawValues[kfIdx] as number;

                      const midR = (rInner + barOuter) / 2;
                      const tipX = center + midR * Math.cos(centerAngle);
                      const tipY = center + midR * Math.sin(centerAngle);

                      return (
                        <path
                          key={`${track.label}-${kfIdx}`}
                          d={arcPath(rInner, barOuter, angleStart, angleEnd)}
                          fill={track.color}
                          className="pointer-events-auto cursor-default opacity-60 hover:opacity-100 transition-[fill-opacity] duration-150 ease-out"
                          onMouseEnter={() => handleEnter(track.label, rawValue, tipX, tipY)}
                          onMouseLeave={handleLeave}
                        />
                      );
                    })}
                  </g>
                );
              })}
        </g>
      </svg>

      {/* Tooltip at bar center */}
      {hover && (
        <div
          className="absolute pointer-events-none z-50"
          style={{ left: hover.x, top: hover.y, transform: 'translate(-50%, -50%)' }}
        >
          <div className="rounded border border-dtour-border bg-dtour-bg px-3 py-1.5 text-xs text-dtour-text shadow-md whitespace-nowrap">
            {hover.label}: {hover.value.toFixed(3)}
          </div>
        </div>
      )}
    </div>
  );
};
