import { useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useRef } from 'react';
import { animationGenAtom, tourPositionAtom } from '../state/atoms.ts';

/** 360° of travel = 1000ms base animation duration */
const MS_PER_FULL_ROTATION = 1000;
/** Minimum animation duration to keep it perceptible */
const MIN_ANIMATION_MS = 80;
/** Stretch factor applied to the base duration */
const DURATION_STRETCH = 1.5;

/** Ease-in-out cubic: slow start + slow end, fast middle. */
const easeInOutCubic = (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

/**
 * Shared hook for animating `tourPositionAtom` to a target value.
 *
 * Multiple components can call `useAnimatePosition()` independently.
 * A generation counter (`animationGenAtom`) ensures that when any
 * component starts a new animation or cancels, all other running
 * animations bail out on their next rAF tick.
 */
export const useAnimatePosition = () => {
  const store = useStore();
  const setPosition = useSetAtom(tourPositionAtom);
  const rafRef = useRef<number | null>(null);

  const cancelLocal = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  /** Cancel any running animation (from any component). */
  const cancelAnimation = useCallback(() => {
    store.set(animationGenAtom, (g) => g + 1);
    cancelLocal();
  }, [store, cancelLocal]);

  /**
   * Animate tour position from its current value to `target` along the
   * shortest arc on the [0, 1) circle, using ease-in-out cubic easing.
   */
  const animateTo = useCallback(
    (target: number) => {
      cancelLocal();

      // Claim a new generation — invalidates all other animations
      const gen = store.get(animationGenAtom) + 1;
      store.set(animationGenAtom, gen);

      // Read current position imperatively via the functional updater
      setPosition((current) => {
        // Shortest angular distance on [0,1) circle
        let delta = target - current;
        if (delta > 0.5) delta -= 1;
        if (delta < -0.5) delta += 1;

        const absDelta = Math.abs(delta);
        if (absDelta < 0.001) return target;

        const startPos = current;
        const startTime = performance.now();
        const durationMs =
          Math.max(MIN_ANIMATION_MS, absDelta * MS_PER_FULL_ROTATION) * DURATION_STRETCH;

        const tick = (now: number) => {
          // Bail if a newer animation or cancel has bumped the generation
          if (store.get(animationGenAtom) !== gen) return;

          const elapsed = now - startTime;
          const t = Math.min(1, elapsed / durationMs);
          const eased = easeInOutCubic(t);

          let pos = startPos + delta * eased;
          // Wrap to [0, 1)
          pos = pos - Math.floor(pos);

          setPosition(pos);

          if (t < 1) {
            rafRef.current = requestAnimationFrame(tick);
          } else {
            rafRef.current = null;
          }
        };

        rafRef.current = requestAnimationFrame(tick);

        // Return current unchanged — the rAF loop will drive updates
        return current;
      });
    },
    [store, setPosition, cancelLocal],
  );

  // Cleanup on unmount
  useEffect(() => cancelLocal, [cancelLocal]);

  return { animateTo, cancelAnimation };
};
