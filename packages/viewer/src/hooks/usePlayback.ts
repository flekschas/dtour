import { useAtomValue, useSetAtom } from 'jotai';
import { useEffect, useRef } from 'react';
import {
  tourDirectionAtom,
  tourPlayingAtom,
  tourPositionAtom,
  tourSpeedAtom,
} from '../state/atoms.ts';

/**
 * rAF loop that advances tour position when playing.
 *
 * Reads playing/speed/direction from atoms, writes position.
 * Automatically pauses when the tab is hidden (rAF stops firing).
 * Wraps at 0/1 for cyclic tour.
 */
export const usePlayback = () => {
  const playing = useAtomValue(tourPlayingAtom);
  const speed = useAtomValue(tourSpeedAtom);
  const direction = useAtomValue(tourDirectionAtom);
  const setPosition = useSetAtom(tourPositionAtom);

  // Use refs for values read inside rAF to avoid re-creating the effect
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const directionRef = useRef(direction);
  directionRef.current = direction;

  useEffect(() => {
    if (!playing) return;

    let prevTime: number | null = null;
    let rafId: number;

    const tick = (time: number) => {
      if (prevTime !== null) {
        const dt = (time - prevTime) / 1000; // seconds
        // Full tour cycle = 20s at speed=1
        const delta = (dt * speedRef.current * directionRef.current) / 20;
        setPosition((prev) => {
          let next = prev + delta;
          // Wrap cyclically
          next = next - Math.floor(next);
          return next;
        });
      }
      prevTime = time;
      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [playing, setPosition]);
};
