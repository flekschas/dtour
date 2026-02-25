import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import {
  tourPlayingAtom,
  tourSuspendedAtom,
  viewModeAtom,
  zenExitTargetAtom,
} from '../state/atoms.ts';

const MODES = ['tour', 'manual', 'zen'] as const;

/**
 * Cycles view modes on Shift+Tab (tour → manual → zen → tour).
 *
 * Also manages tour suspension: pauses playback when leaving tour mode,
 * and sets `tourSuspended` when returning to tour so the current
 * projection is preserved until the user interacts with the slider.
 *
 * When exiting zen mode, defers the actual mode switch so the grand
 * tour animation can ease out first.
 */
export const useModeCycling = () => {
  const viewMode = useAtomValue(viewModeAtom);
  const setViewMode = useSetAtom(viewModeAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const setTourSuspended = useSetAtom(tourSuspendedAtom);
  const setZenExitTarget = useSetAtom(zenExitTargetAtom);

  // Use ref so the keydown handler always sees the latest viewMode
  // without needing to re-register the listener on every mode change.
  const viewModeRef = useRef(viewMode);
  viewModeRef.current = viewMode;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !e.shiftKey || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      e.preventDefault();

      const current = viewModeRef.current;
      const idx = MODES.indexOf(current);
      const next = MODES[(idx + 1) % MODES.length]!;

      if (current === 'zen') {
        // Exiting zen — request ease-out, defer mode switch
        // next is always 'tour' here (zen → tour in the cycle)
        setZenExitTarget(next as 'tour' | 'manual');
      } else {
        if (current === 'tour') {
          setPlaying(false);
        }
        if (next === 'tour') {
          setTourSuspended(true);
        }
        if (next === 'zen') {
          setZenExitTarget(null);
        }
        setViewMode(next);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setViewMode, setPlaying, setTourSuspended, setZenExitTarget]);
};
