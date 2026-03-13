import { GLASBEY_DARK, MAGMA_25, OKABE_ITO, VIRIDIS_25 } from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback } from 'react';
import {
  legendClearGenAtom,
  legendSelectionAtom,
  metadataAtom,
  paletteAtom,
  pointColorAtom,
} from '../state/atoms.ts';

const handleSwatchClick = (
  index: number,
  event: React.MouseEvent,
  setSelection: (update: (prev: Set<number> | null) => Set<number> | null) => void,
  setClearGen: (update: (prev: number) => number) => void,
) => {
  const isMeta = event.metaKey || event.ctrlKey;
  const isAlt = event.altKey;

  setSelection((prev) => {
    if (isMeta) {
      const next = new Set(prev ?? []);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      if (next.size === 0) {
        setClearGen((g) => g + 1);
        return null;
      }
      return next;
    }

    if (isAlt) {
      if (!prev) return prev;
      const next = new Set(prev);
      next.delete(index);
      if (next.size === 0) {
        setClearGen((g) => g + 1);
        return null;
      }
      return next;
    }

    // Plain click: select only this one, or toggle off if sole selection
    if (prev && prev.size === 1 && prev.has(index)) {
      setClearGen((g) => g + 1);
      return null;
    }
    return new Set([index]);
  });
};

export const ColorLegend = () => {
  const metadata = useAtomValue(metadataAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const palette = useAtomValue(paletteAtom);
  const [legendSelection, setLegendSelection] = useAtom(legendSelectionAtom);
  const setClearGen = useSetAtom(legendClearGenAtom);

  const onSwatchClick = useCallback(
    (index: number, event: React.MouseEvent) => {
      handleSwatchClick(index, event, setLegendSelection, setClearGen);
    },
    [setLegendSelection, setClearGen],
  );

  if (typeof pointColor !== 'string' || !metadata) return null;
  const column = pointColor;

  const isCategorical = metadata.categoricalColumnNames.includes(column);
  const hasSelection = legendSelection !== null;

  if (isCategorical) {
    const labels = metadata.categoricalLabels[column] ?? [];
    const colors = labels.length <= OKABE_ITO.length
    ? OKABE_ITO
    : [
        ...OKABE_ITO,
        ...Array(Math.ceil((labels.length - OKABE_ITO.length) / GLASBEY_DARK.length)).fill(undefined).flatMap(() => GLASBEY_DARK)
      ] as [number, number, number][];
    return (
      <div className="flex h-full flex-col overflow-hidden border-l border-dtour-surface bg-dtour-bg px-3 pb-3 pt-12 text-xs text-dtour-text">
        <div className="mb-2 shrink-0 font-semibold text-white truncate">{column}</div>
        <div className="flex flex-col gap-1.5 overflow-y-auto">
          {labels.map((label, i) => {
            const [r, g, b] = colors[i % colors.length]!;
            const dimmed = hasSelection && !legendSelection.has(i);
            return (
              <button
                key={label}
                type="button"
                className={`flex items-center gap-2 min-w-0 cursor-pointer rounded px-1 py-0.5 transition-opacity hover:bg-white/10 ${dimmed ? 'opacity-35' : 'opacity-100'}`}
                onClick={(e) => onSwatchClick(i, e)}
              >
                <span
                  className="shrink-0 w-3 h-3 rounded-sm"
                  style={{ backgroundColor: `rgb(${r},${g},${b})` }}
                />
                <span className="truncate">{label}</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  // Continuous (numeric column)
  const colIndex = metadata.columnNames.indexOf(column);
  if (colIndex === -1) return null;
  const min = metadata.mins[colIndex]!;
  const max = metadata.maxes[colIndex]!;

  // 13 stops from 25-entry cmap (step 2): labeled on even stopIdx, unlabeled in between
  const step = 2;
  const stops: { value: number; color: [number, number, number]; stopIdx: number }[] = [];
  const cmap = palette === 'magma' ? MAGMA_25 : VIRIDIS_25;
  for (let i = 0; i < cmap.length; i += step) {
    const t = i / (cmap.length - 1);
    stops.push({ value: min + t * (max - min), color: cmap[i]!, stopIdx: i / step });
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-dtour-bg px-3 pb-3 pt-12 text-xs text-dtour-text">
      <div className="mb-2 shrink-0 font-semibold text-white truncate">{column}</div>
      <div className="flex flex-col gap-0.5 overflow-y-auto">
        {stops.reverse().map(({ value, color: [r, g, b], stopIdx }) => {
          const dimmed = hasSelection && !legendSelection.has(stopIdx);
          const showLabel = stopIdx % 2 === 0;
          return (
            <button
              key={value}
              type="button"
              className={`flex items-center gap-2 min-w-0 cursor-pointer rounded px-1 py-0.5 transition-opacity hover:bg-white/10 ${dimmed ? 'opacity-35' : 'opacity-100'}`}
              onClick={(e) => onSwatchClick(stopIdx, e)}
            >
              <span
                className="shrink-0 w-3 h-3 rounded-sm"
                style={{ backgroundColor: `rgb(${r},${g},${b})` }}
              />
              {showLabel && <span className="truncate">{formatValue(value)}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
};

const formatValue = (v: number): string => {
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  return v.toPrecision(3);
};
