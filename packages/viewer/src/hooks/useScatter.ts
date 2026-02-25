import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import { hexToRgb, isHexColor } from '../lib/color-utils.ts';
import {
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  guidedSuspendedAtom,
  metadataAtom,
  pointColorAtom,
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
  const setMetadata = useSetAtom(metadataAtom);

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
      scatter.encodeColor(color);
    }
  }, [scatter, pointSize, opacity, color]);

  // Forward camera
  useEffect(() => {
    scatter?.setCamera({ pan: [panX, panY], zoom });
  }, [scatter, panX, panY, zoom]);

  // Subscribe to scatter status events and update metadata atom.
  // Use a ref so the setMetadata closure never goes stale.
  const setMetadataRef = useRef(setMetadata);
  setMetadataRef.current = setMetadata;

  useEffect(() => {
    if (!scatter) return;
    return scatter.subscribe((s: ScatterStatus) => {
      if (s.type === 'metadata') {
        setMetadataRef.current(s.metadata);
      }
    });
  }, [scatter]);
};
