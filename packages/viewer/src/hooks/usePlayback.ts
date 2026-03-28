import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue } from 'jotai';
import { useEffect } from 'react';
import { tourDirectionAtom, tourPlayingAtom, tourSpeedAtom } from '../state/atoms.ts';

/**
 * Delegates playback to the GPU worker's rAF loop.
 *
 * When playing, sends startPlayback to the scatter instance which runs
 * a requestAnimationFrame loop in the GPU worker — rendering directly
 * without main-thread involvement. Position updates are broadcast back
 * at ~30fps for UI sync (slider, atom).
 */
export const usePlayback = (scatter: ScatterInstance | null) => {
  const playing = useAtomValue(tourPlayingAtom);
  const speed = useAtomValue(tourSpeedAtom);
  const direction = useAtomValue(tourDirectionAtom);

  useEffect(() => {
    if (!scatter) return;
    if (playing) {
      scatter.startPlayback(speed, direction);
    } else {
      scatter.stopPlayback();
    }
    return () => scatter.stopPlayback();
  }, [scatter, playing, speed, direction]);
};
