import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue } from 'jotai';
import { useEffect } from 'react';
import {
  basisTransitioningAtom,
  tourDirectionAtom,
  tourPlayingAtom,
  tourSpeedAtom,
} from '../state/atoms.ts';

/**
 * Delegates playback to the GPU worker's rAF loop.
 *
 * When playing, sends startPlayback to the scatter instance which runs
 * a requestAnimationFrame loop in the GPU worker — rendering directly
 * without main-thread involvement. Position updates are broadcast back
 * at ~30fps for UI sync (slider, atom).
 *
 * Defers startPlayback while a basis-blend transition is active so the
 * worker's playback ticks don't clear directBasis and race the handoff.
 */
export const usePlayback = (scatter: ScatterInstance | null) => {
  const playing = useAtomValue(tourPlayingAtom);
  const basisTransitioning = useAtomValue(basisTransitioningAtom);
  const speed = useAtomValue(tourSpeedAtom);
  const direction = useAtomValue(tourDirectionAtom);

  useEffect(() => {
    if (!scatter) return;
    if (playing && !basisTransitioning) {
      scatter.startPlayback(speed, direction);
    } else {
      scatter.stopPlayback();
    }
    return () => scatter.stopPlayback();
  }, [scatter, playing, basisTransitioning, speed, direction]);
};
