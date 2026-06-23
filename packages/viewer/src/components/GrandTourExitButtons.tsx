import { CursorIcon, PathIcon } from '@phosphor-icons/react';
import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef, useState } from 'react';
import { cn } from '../lib/utils.ts';
import { grandExitTargetAtom, tourTraversalAtom } from '../state/atoms.ts';
import { Button } from './ui/button.tsx';

const VISIBLE_DURATION_MS = 3000;

/**
 * Translucent button group shown in grand tour mode, top-left corner.
 * Fades out after 3 seconds of inactivity, fades back in on mouse movement.
 * Lets users exit back to guided or manual mode without needing the toolbar
 * or the keyboard shortcut (Shift+Tab).
 */
export const GrandTourExitButtons = () => {
  const tourTraversal = useAtomValue(tourTraversalAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const [visible, setVisible] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (tourTraversal !== 'grand') return;

    // Delay by a frame so the CSS transition from opacity-0 → opacity-100 fires
    const rafId = requestAnimationFrame(() => setVisible(true));
    timerRef.current = setTimeout(() => setVisible(false), VISIBLE_DURATION_MS);

    const handleMouseMove = () => {
      setVisible(true);
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => setVisible(false), VISIBLE_DURATION_MS);
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => {
      cancelAnimationFrame(rafId);
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      window.removeEventListener('mousemove', handleMouseMove);
      // Reset so re-entering grand mode always starts from opacity-0 and
      // replays the fade-in transition.
      setVisible(false);
    };
  }, [tourTraversal]);

  if (tourTraversal !== 'grand') return null;

  return (
    <div
      className={cn(
        'absolute top-4 left-4 flex rounded-md overflow-hidden z-20',
        'transition-opacity duration-500 ease-out',
        visible ? 'opacity-100' : 'opacity-0 pointer-events-none',
      )}
    >
      <Button
        variant="ghost"
        size="sm"
        className="rounded-none bg-dtour-surface/60 hover:bg-dtour-surface backdrop-blur-sm"
        title="Switch to guided tour"
        onClick={() => setGrandExitTarget('guided')}
      >
        <PathIcon size={12} />
        <span>Guided</span>
      </Button>
      <div className="w-px self-stretch bg-dtour-border/60" />
      <Button
        variant="ghost"
        size="sm"
        className="rounded-none bg-dtour-surface/60 hover:bg-dtour-surface backdrop-blur-sm"
        title="Switch to manual exploration"
        onClick={() => setGrandExitTarget('manual')}
      >
        <CursorIcon size={12} />
        <span>Manual</span>
      </Button>
    </div>
  );
};
