import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import { hexToRgb, hexToRgb255, isHexColor } from '../lib/color-utils.ts';
import {
  backgroundColorAtom,
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  colorMapAtom,
  guidedSuspendedAtom,
  legendClearGenAtom,
  legendSelectionAtom,
  metadataAtom,
  paletteAtom,
  pointColorAtom,
  resolvedThemeAtom,
  tourPositionAtom,
} from '../state/atoms.ts';
import { resolvedPointOpacityAtom, resolvedPointSizeAtom } from '../state/auto-style.ts';

/**
 * Bridge between Jotai atoms and a ScatterInstance.
 *
 * Subscribes to atom changes and forwards them as postMessage calls
 * to the GPU worker. Also subscribes to scatter status events and
 * writes metadata back into Jotai.
 */
export const useScatter = (scatter: ScatterInstance | null) => {
  const position = useAtomValue(tourPositionAtom);
  const pointSize = useAtomValue(resolvedPointSizeAtom);
  const opacity = useAtomValue(resolvedPointOpacityAtom);
  const color = useAtomValue(pointColorAtom);
  const guidedSuspended = useAtomValue(guidedSuspendedAtom);
  const panX = useAtomValue(cameraPanXAtom);
  const panY = useAtomValue(cameraPanYAtom);
  const zoom = useAtomValue(cameraZoomAtom);
  const backgroundColor = useAtomValue(backgroundColorAtom);
  const palette = useAtomValue(paletteAtom);
  const resolvedTheme = useAtomValue(resolvedThemeAtom);
  const rawColorMap = useAtomValue(colorMapAtom);
  const metadata = useAtomValue(metadataAtom);
  const setMetadata = useSetAtom(metadataAtom);
  const legendSelection = useAtomValue(legendSelectionAtom);
  const legendClearGen = useAtomValue(legendClearGenAtom);
  const setLegendSelection = useSetAtom(legendSelectionAtom);

  // Forward background color
  useEffect(() => {
    scatter?.setBackgroundColor(backgroundColor);
  }, [scatter, backgroundColor]);

  // Forward camera — registered first so the worker receives setCamera before
  // setTourPosition (which triggers a render). Otherwise the first render uses
  // the client's default zoom=1 instead of the atom value.
  useEffect(() => {
    scatter?.setCamera({ pan: [panX, panY], zoom });
  }, [scatter, panX, panY, zoom]);

  // Forward tour position (skipped when suspended after returning from manual/grand)
  useEffect(() => {
    if (guidedSuspended) return;
    scatter?.setTourPosition(position);
  }, [scatter, position, guidedSuspended]);

  // Forward point style (size + opacity + uniform color)
  useEffect(() => {
    if (!scatter) return;

    if (Array.isArray(color)) {
      // RGB tuple — uniform color; clear any per-point encoding
      scatter.clearColor();
      scatter.setStyle({ pointSize, opacity, color });
    } else if (isHexColor(color)) {
      // Hex string — parse to RGB uniform color
      scatter.clearColor();
      scatter.setStyle({ pointSize, opacity, color: hexToRgb(color) });
    } else {
      // Column name — encode per-point colors via data worker
      scatter.setStyle({ pointSize, opacity });
      // Resolve theme-aware colorMap to Record<string, [r,g,b]> for the scatter worker
      let resolvedColorMap: Record<string, [number, number, number]> | undefined;
      if (rawColorMap) {
        resolvedColorMap = {};
        for (const [label, value] of Object.entries(rawColorMap)) {
          const hex = typeof value === 'string' ? value : value[resolvedTheme];
          resolvedColorMap[label] = hexToRgb255(hex);
        }
      }
      scatter.encodeColor(color, palette, resolvedTheme, resolvedColorMap);
    }
  }, [scatter, pointSize, opacity, color, palette, resolvedTheme, rawColorMap]);

  // Forward legend selection → scatter.selectByColumn
  useEffect(() => {
    if (!scatter || !metadata || legendSelection === null || legendSelection.size === 0) return;

    // Determine the active color column
    if (typeof color !== 'string' || isHexColor(color)) return;
    const column = color;

    const isCategorical = metadata.categoricalColumnNames.includes(column);

    if (isCategorical) {
      scatter.selectByColumn(column, { labelIndices: Array.from(legendSelection) });
    } else {
      // Continuous: 13 stops (indices 0–12). Middle stops each cover range/12,
      // end stops (0 and 12) cover half that (range/24).
      const colIndex = metadata.columnNames.indexOf(column);
      if (colIndex === -1) return;
      const min = metadata.mins[colIndex]!;
      const max = metadata.maxes[colIndex]!;
      const range = max - min;
      const midWidth = range / 12;
      const endWidth = midWidth / 2;

      const ranges: number[] = [];
      for (const stopIdx of legendSelection) {
        const lo = stopIdx === 0 ? min : min + endWidth + (stopIdx - 1) * midWidth;
        const hi = stopIdx === 12 ? max : min + endWidth + stopIdx * midWidth;
        ranges.push(lo, hi);
      }

      scatter.selectByColumn(column, { valueRanges: new Float32Array(ranges) });
    }
  }, [scatter, legendSelection, color, metadata]);

  // Clear scatter selection when legend explicitly deselects (gen bumped by ColorLegend)
  useEffect(() => {
    if (!scatter || legendClearGen === 0) return;
    scatter.clearSelection();
  }, [scatter, legendClearGen]);

  // Reset legend selection and clear GPU selection mask when color column changes
  const prevColorRef = useRef(color);
  useEffect(() => {
    if (prevColorRef.current !== color) {
      prevColorRef.current = color;
      setLegendSelection(null);
      scatter?.clearSelection();
    }
  }, [scatter, color, setLegendSelection]);

  // Subscribe to scatter status events and update metadata atom.
  // Use a ref so the setMetadata closure never goes stale.
  const setMetadataRef = useRef(setMetadata);
  setMetadataRef.current = setMetadata;

  const setLegendSelectionRef = useRef(setLegendSelection);
  setLegendSelectionRef.current = setLegendSelection;

  useEffect(() => {
    if (!scatter) return;
    return scatter.subscribe((s: ScatterStatus) => {
      if (s.type === 'metadata') {
        setMetadataRef.current(s.metadata);
        setLegendSelectionRef.current(null);
      }
    });
  }, [scatter]);
};
