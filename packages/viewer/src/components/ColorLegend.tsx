import {
  COLORMAP_2D_INDEX,
  COLORMAP_2D_NAMES,
  GLASBEY_DARK,
  GLASBEY_LIGHT,
  MAGMA_25,
  OKABE_ITO,
  VIRIDIS_25,
  packColormap2DLut,
} from '@dtour/scatter';
import type { Colormap2DName } from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useRef } from 'react';
import { hexToRgb255 } from '../lib/color-utils.ts';
import {
  color2dColumnsAtom,
  color2dEnabledAtom,
  color2dMapAtom,
  colorMapAtom,
  legendClearGenAtom,
  legendSelectionAtom,
  metadataAtom,
  paletteAtom,
  pointColorAtom,
  resolvedThemeAtom,
} from '../state/atoms.ts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select.tsx';

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
  const [palette, setPalette] = useAtom(paletteAtom);
  const resolvedTheme = useAtomValue(resolvedThemeAtom);
  const rawColorMap = useAtomValue(colorMapAtom);
  const [legendSelection, setLegendSelection] = useAtom(legendSelectionAtom);
  const setClearGen = useSetAtom(legendClearGenAtom);
  const color2dEnabled = useAtomValue(color2dEnabledAtom);
  const color2dColumns = useAtomValue(color2dColumnsAtom);
  const [color2dMap, setColor2dMap] = useAtom(color2dMapAtom);

  const onSwatchClick = useCallback(
    (index: number, event: React.MouseEvent) => {
      handleSwatchClick(index, event, setLegendSelection, setClearGen);
    },
    [setLegendSelection, setClearGen],
  );

  // 2D color legend (only when both columns are selected)
  if (color2dEnabled && color2dColumns && color2dColumns[1]) {
    return (
      <Legend2D
        columnX={color2dColumns[0]}
        columnY={color2dColumns[1]}
        colormap={color2dMap}
        onColormapChange={setColor2dMap}
      />
    );
  }

  if (typeof pointColor !== 'string' || !metadata) return null;
  const column = pointColor;

  const isCategorical = metadata.categoricalColumnNames.includes(column);
  const hasSelection = legendSelection !== null;

  if (isCategorical) {
    const labels = metadata.categoricalLabels[column] ?? [];
    let colors: [number, number, number][];
    if (rawColorMap) {
      colors = labels.map((label) => {
        const v = rawColorMap[label];
        if (!v) return [128, 128, 128] as [number, number, number];
        const hex = typeof v === 'string' ? v : v[resolvedTheme];
        return hexToRgb255(hex);
      });
    } else {
      const glasbey = resolvedTheme === 'light' ? GLASBEY_LIGHT : GLASBEY_DARK;
      colors =
        labels.length <= OKABE_ITO.length
          ? OKABE_ITO
          : ([
              ...OKABE_ITO,
              ...Array(Math.ceil((labels.length - OKABE_ITO.length) / glasbey.length))
                .fill(undefined)
                .flatMap(() => glasbey),
            ] as [number, number, number][]);
    }
    return (
      <div className="flex h-full flex-col overflow-hidden bg-dtour-bg text-xs text-dtour-text">
        <div className="flex h-10 shrink-0 items-center border-b border-dtour-surface px-3 font-semibold text-dtour-highlight truncate">
          {column}
        </div>
        <div className="flex flex-col gap-1.5 overflow-y-auto px-3 pt-3 pb-3">
          {labels.map((label, i) => {
            const [r, g, b] = colors[i % colors.length]!;
            const dimmed = hasSelection && !legendSelection.has(i);
            return (
              <button
                key={label}
                type="button"
                className={`flex items-center gap-2 min-w-0 cursor-pointer rounded px-1 py-0.5 transition-opacity hover:bg-dtour-highlight/10 ${dimmed ? 'opacity-35' : 'opacity-100'}`}
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
  const baseCmap = palette === 'magma' ? MAGMA_25 : VIRIDIS_25;
  const cmap =
    resolvedTheme === 'light' ? ([...baseCmap].reverse() as [number, number, number][]) : baseCmap;
  for (let i = 0; i < cmap.length; i += step) {
    const t = i / (cmap.length - 1);
    stops.push({ value: min + t * (max - min), color: cmap[i]!, stopIdx: i / step });
  }

  return (
    <div className="flex h-full flex-col overflow-hidden bg-dtour-bg text-xs text-dtour-text">
      <div className="flex h-10 shrink-0 items-center border-b border-dtour-surface px-3 font-semibold text-dtour-highlight truncate">
        {column}
      </div>
      <div className="flex flex-col gap-0.5 overflow-y-auto px-3 pt-3 pb-3">
        {stops.reverse().map(({ value, color: [r, g, b], stopIdx }) => {
          const dimmed = hasSelection && !legendSelection.has(stopIdx);
          const showLabel = stopIdx % 2 === 0;
          return (
            <button
              key={value}
              type="button"
              className={`flex items-center gap-2 min-w-0 cursor-pointer rounded px-1 py-0.5 transition-opacity hover:bg-dtour-highlight/10 ${dimmed ? 'opacity-35' : 'opacity-100'}`}
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
      <div className="shrink-0 border-t border-dtour-surface px-3 py-2">
        <span className="text-[10px] text-dtour-text-muted">Color map</span>
        <Select value={palette} onValueChange={(v) => setPalette(v as typeof palette)}>
          <SelectTrigger className="mt-1 h-6 w-full text-[10px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="viridis">viridis</SelectItem>
            <SelectItem value="magma">magma</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};

const formatValue = (v: number): string => {
  if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
  return v.toPrecision(3);
};

// ─── 2D Color Legend ──────────────────────────────────────────────────────

// Oklab → sRGB conversion (mirrors the shader)
const oklab_to_srgb = (L: number, a: number, b: number): [number, number, number] => {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;
  const l = l_ ** 3;
  const m = m_ ** 3;
  const s = s_ ** 3;
  const r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const bCh = -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s;
  const gamma = (x: number) => Math.max(0, Math.min(1, x)) ** (1 / 2.2) * 255;
  return [gamma(r), gamma(g), gamma(bCh)];
};

const renderColormap2D = (ctx: CanvasRenderingContext2D, mapName: Colormap2DName, size: number) => {
  const mapIndex = COLORMAP_2D_INDEX[mapName] ?? 0;
  const imageData = ctx.createImageData(size, size);
  const data = imageData.data;

  if (mapName === 'oklab_polar') {
    // Procedural
    for (let y = 0; y < size; y++) {
      const v = 1 - y / (size - 1); // flip y so bottom=0
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1);
        const cx = u - 0.5;
        const cy = v - 0.5;
        const radius = Math.min(Math.sqrt(cx * cx + cy * cy) * 2, 1);
        const angle = Math.atan2(cy, cx);
        const chroma = radius * 0.25;
        const lightness = 0.8 - radius * 0.3;
        const [r, g, b] = oklab_to_srgb(
          lightness,
          chroma * Math.cos(angle),
          chroma * Math.sin(angle),
        );
        const idx = (y * size + x) * 4;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
        data[idx + 3] = 255;
      }
    }
  } else {
    // LUT-based
    const lut = packColormap2DLut(mapIndex);
    if (!lut) return;

    const interp = (base: number, t: number): number => {
      const pos = t * 15;
      const lo = Math.min(Math.floor(pos), 14);
      const frac = pos - lo;
      return lut[base + lo]! * (1 - frac) + lut[base + lo + 1]! * frac;
    };

    for (let y = 0; y < size; y++) {
      const v = 1 - y / (size - 1);
      for (let x = 0; x < size; x++) {
        const u = x / (size - 1);
        const r = interp(0, u) * interp(16, v);
        const g = interp(32, u) * interp(48, v);
        const b = interp(64, u) * interp(80, v) + interp(96, u) * interp(112, v);
        const idx = (y * size + x) * 4;
        data[idx] = Math.max(0, Math.min(255, r * 255));
        data[idx + 1] = Math.max(0, Math.min(255, g * 255));
        data[idx + 2] = Math.max(0, Math.min(255, b * 255));
        data[idx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
};

const Legend2D = ({
  columnX,
  columnY,
  colormap,
  onColormapChange,
}: {
  columnX: string;
  columnY: string;
  colormap: Colormap2DName;
  onColormapChange: (value: Colormap2DName) => void;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const SIZE = 150;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    renderColormap2D(ctx, colormap, SIZE);
  }, [colormap]);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-dtour-bg text-xs text-dtour-text">
      <div className="flex h-10 shrink-0 items-center capitalize border-b border-dtour-surface px-3 font-semibold text-dtour-highlight truncate">
        {colormap}
      </div>
      <div className="flex flex-col gap-1 p-2 text-[10px] text-dtour-text">
        <div className="flex gap-1.5">
          <span
            className="writing-mode-vertical rotate-180 text-dtour-text-muted flex items-center justify-center"
            style={{ writingMode: 'vertical-rl', height: `${SIZE}px` }}
          >
            {columnY}
          </span>
          <div className="flex flex-col items-center gap-0.5">
            <canvas
              ref={canvasRef}
              width={SIZE}
              height={SIZE}
              className="rounded-sm"
              style={{ width: SIZE, height: SIZE }}
            />
            <span className="text-dtour-text-muted">{columnX}</span>
          </div>
        </div>
      </div>
      <div className="shrink-0 border-t border-dtour-surface px-3 py-2">
        <span className="text-[10px] text-dtour-text-muted">Color map</span>
        <Select value={colormap} onValueChange={(v) => onColormapChange(v as Colormap2DName)}>
          <SelectTrigger className="mt-1 h-6 w-full text-[10px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {COLORMAP_2D_NAMES.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
};
