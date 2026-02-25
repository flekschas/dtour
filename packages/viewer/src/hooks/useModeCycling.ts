import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import {
  grandExitTargetAtom,
  guidedSuspendedAtom,
  tourPlayingAtom,
  viewModeAtom,
} from '../state/atoms.ts';

const MODES = ['guided', 'manual', 'grand'] as const;

/**
 * Cycles view modes on Shift+Tab (guided → manual → grand → guided).
 *
 * Also manages guided suspension: pauses playback when leaving guided mode,
 * and sets `guidedSuspended` when returning to guided so the current
 * projection is preserved until the user interacts with the slider.
 *
 * When exiting grand mode, defers the actual mode switch so the grand
 * tour animation can ease out first.
 */
export const useModeCycling = () => {
  const viewMode = useAtomValue(viewModeAtom);
  const setViewMode = useSetAtom(viewModeAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);

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

      if (current === 'grand') {
        // Exiting grand — request ease-out, defer mode switch
        // next is always 'guided' here (grand → guided in the cycle)
        setGrandExitTarget(next as 'guided' | 'manual');
      } else {
        if (current === 'guided') {
          setPlaying(false);
        }
        if (next === 'guided') {
          setGuidedSuspended(true);
        }
        if (next === 'grand') {
          setGrandExitTarget(null);
        }
        setViewMode(next);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setViewMode, setPlaying, setGuidedSuspended, setGrandExitTarget]);
};
