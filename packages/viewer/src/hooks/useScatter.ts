import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import {
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  metadataAtom,
  pointColorAtom,
  pointOpacityAtom,
  pointSizeAtom,
  tourPositionAtom,
} from '../state/atoms.ts';

/**
 * Bridge between Jotai atoms and a ScatterInstance.
 *
 * Subscribes to atom changes and forwards them as postMessage calls
 * to the GPU worker. Also subscribes to scatter status events and
 * writes metadata back into Jotai.
 */
export const useScatter = (scatter: ScatterInstance | null) => {
  const position = useAtomValue(tourPositionAtom);
  const pointSize = useAtomValue(pointSizeAtom);
  const opacity = useAtomValue(pointOpacityAtom);
  const color = useAtomValue(pointColorAtom);
  const panX = useAtomValue(cameraPanXAtom);
  const panY = useAtomValue(cameraPanYAtom);
  const zoom = useAtomValue(cameraZoomAtom);
  const setMetadata = useSetAtom(metadataAtom);

  // Forward tour position
  useEffect(() => {
    scatter?.setTourPosition(position);
  }, [scatter, position]);

  // Forward point style
  useEffect(() => {
    scatter?.setStyle({ pointSize, opacity, color });
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
