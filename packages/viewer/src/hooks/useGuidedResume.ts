import { interpolateAtPosition } from '@dtour/scatter';
import type { Metadata, ScatterInstance } from '@dtour/scatter';
import { useStore } from 'jotai';
import { useCallback, useEffect, useRef } from 'react';
import type { RefObject } from 'react';
import {
  basisTransitioningAtom,
  currentBasisAtom,
  guidedSuspendedAtom,
  tourPositionAtom,
  viewModeAtom,
} from '../state/atoms.ts';

const easeOut = (t: number): number => 1 - (1 - t) ** 3;

/**
 * Smoothly re-enters guided tour mode by blending the current direct basis
 * (from manual/grand manipulation) into the tour-interpolated basis over
 * `durationMs`.
 *
 * `guidedSuspendedAtom` stays true for the entire blend so nothing else
 * overwrites `currentBasisAtom` or calls `setTourPosition` during the
 * handoff. Both atoms are cleared atomically when the last frame fires.
 */
export const useGuidedResume = (
  scatterRef: RefObject<ScatterInstance | null>,
  resolvedViewsRef: RefObject<Float32Array[] | null>,
  arcLengthsRef: RefObject<Float32Array | null>,
  metadataRef: RefObject<Metadata | null>,
  positionRef: RefObject<number>,
) => {
  const store = useStore();
  const rafRef = useRef<number | null>(null);
  const activeRef = useRef(false);

  const cancelTransition = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (activeRef.current) {
      activeRef.current = false;
      store.set(basisTransitioningAtom, false);
      // Note: guidedSuspendedAtom is intentionally NOT cleared here.
      // The caller (handleDragStart, mode-switch, etc.) owns that state.
      // currentBasisAtom was updated on every tick, so any restart picks
      // up from the correct mid-blend visual position.
    }
  }, [store]);

  /**
   * Unsuspend guided mode and animate the projection from the current direct
   * basis to the tour basis over `durationMs`.
   *
   * @param durationMs   Blend duration (300ms for clicks, 150ms for wheel).
   * @param getPosition  Live position getter. Defaults to reading the atom.
   *                     Pass `() => positionRef.current` from the wheel handler
   *                     so the target tracks in-flight position updates.
   */
  const resumeWithTransition = useCallback(
    (durationMs: number, getPosition?: () => number) => {
      cancelTransition();

      // Already on the live tour — nothing to blend from.
      if (!store.get(guidedSuspendedAtom)) return;

      // Validate all prerequisites before touching state.
      const startBasis = store.get(currentBasisAtom);
      const rv = resolvedViewsRef.current;
      const al = arcLengthsRef.current;
      const meta = metadataRef.current;

      if (!startBasis || !rv || !al || !meta) {
        // Can't animate — unsuspend immediately so the user isn't stuck.
        store.set(guidedSuspendedAtom, false);
        return;
      }

      const dims = meta.dimCount;
      const start = new Float32Array(startBasis);
      store.set(basisTransitioningAtom, true);
      activeRef.current = true;
      // guidedSuspendedAtom intentionally stays true until handoff completes:
      // keeps currentBasisAtom and setTourPosition from being driven by tour
      // interpolation while the blend is still active.

      const readPos = getPosition ?? (() => store.get(tourPositionAtom));
      const startTime = performance.now();
      const scratch = new Float32Array(dims * 2);

      const tick = (now: number) => {
        if (!activeRef.current) return;

        const t = Math.min(1, (now - startTime) / durationMs);
        const eased = easeOut(t);

        interpolateAtPosition(scratch, rv, al, dims, readPos());

        for (let i = 0; i < scratch.length; i++) {
          scratch[i] = start[i]! + (scratch[i]! - start[i]!) * eased;
        }

        // Keep currentBasisAtom in sync so AxisOverlay (read-only mode)
        // tracks the blend — same contract as the DtourViewer effect that
        // normally owns this atom while guided is unsuspended.
        const snapshot = new Float32Array(scratch);
        store.set(currentBasisAtom, snapshot);
        // Transfer a separate copy to the GPU (postMessage transfers the buffer).
        scatterRef.current?.setDirectBasis(scratch.slice());

        if (t < 1) {
          rafRef.current = requestAnimationFrame(tick);
        } else {
          rafRef.current = null;
          activeRef.current = false;
          // Clear both atoms atomically on the final frame so useScatter's
          // setTourPosition effect fires with a basis that already matches.
          store.set(basisTransitioningAtom, false);
          store.set(guidedSuspendedAtom, false);
        }
      };

      rafRef.current = requestAnimationFrame(tick);
    },
    [store, scatterRef, resolvedViewsRef, arcLengthsRef, metadataRef, cancelTransition],
  );

  const isTransitioning = useCallback(() => activeRef.current, []);

  // Auto-cancel if the viewer leaves guided mode during the blend so the rAF
  // doesn't keep calling setDirectBasis after manual/grand takes over.
  useEffect(() => {
    return store.sub(viewModeAtom, () => {
      if (store.get(viewModeAtom) !== 'guided') cancelTransition();
    });
  }, [store, cancelTransition]);

  useEffect(() => cancelTransition, [cancelTransition]);

  return { resumeWithTransition, cancelTransition, isTransitioning };
};
