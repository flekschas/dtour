import type { Metadata, ScatterInstance } from '@dtour/scatter';
import { useAtomValue, useSetAtom, useStore } from 'jotai';
import { useEffect, useRef } from 'react';
import { gramSchmidt } from '../lib/gram-schmidt.ts';
import {
  activeIndicesAtom,
  currentBasisAtom,
  grandExitTargetAtom,
  guidedSuspendedAtom,
  tourSpeedAtom,
  viewModeAtom,
} from '../state/atoms.ts';

const EASE_DURATION = 0.5; // seconds

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Givens-rotation grand tour for grand mode.
 *
 * Generates random angular velocities for active dimension pairs and
 * applies rotations each frame via rAF. Sends basis to GPU via
 * `setDirectBasis` — much cheaper than `setBases`.
 *
 * Eases in over 500ms on entry and eases out over 500ms on exit.
 * During ease-out, `viewMode` stays 'grand' — the actual mode switch
 * happens only after the animation decelerates to zero.
 */
export const useGrandTour = (
  scatter: ScatterInstance | null,
  viewMode: 'guided' | 'manual' | 'grand',
  metadata: Metadata | null,
): void => {
  const speed = useAtomValue(tourSpeedAtom);
  const speedRef = useRef(speed);
  speedRef.current = speed;

  const scatterRef = useRef(scatter);
  scatterRef.current = scatter;

  const activeIndices = useAtomValue(activeIndicesAtom);

  // Read exit target via ref so the rAF closure always sees the latest
  // value without restarting the effect.
  const exitTarget = useAtomValue(grandExitTargetAtom);
  const exitTargetRef = useRef(exitTarget);
  exitTargetRef.current = exitTarget;

  const store = useStore();
  const setViewMode = useSetAtom(viewModeAtom);
  const setGrandExitTarget = useSetAtom(grandExitTargetAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);

  useEffect(() => {
    if (viewMode !== 'grand' || !metadata || metadata.dimCount < 2 || !scatter) return;
    if (activeIndices.length < 2) return;

    const dims = metadata.dimCount;

    // Build pairs only from active dimensions
    const pairs: [number, number][] = [];
    for (let a = 0; a < activeIndices.length; a++) {
      for (let b = a + 1; b < activeIndices.length; b++) {
        pairs.push([activeIndices[a]!, activeIndices[b]!]);
      }
    }
    const numPairs = pairs.length;

    // Generate random angular velocities for each active pair
    const omegas = new Float32Array(numPairs);
    for (let i = 0; i < numPairs; i++) {
      omegas[i] = (0.5 + Math.random()) * Math.PI * (Math.random() > 0.5 ? 1 : -1);
    }

    // Initialize basis from the current projection so the view doesn't jump
    const current = store.get(currentBasisAtom);
    const basis = new Float32Array(dims * 2);
    if (current && current.length === dims * 2) {
      basis.set(current);
    } else {
      basis[activeIndices[0]!] = 1;
      basis[dims + activeIndices[1]!] = 1;
    }

    // Zero out inactive dimensions and re-orthonormalize
    const activeSet = new Set(activeIndices);
    for (let d = 0; d < dims; d++) {
      if (!activeSet.has(d)) {
        basis[d] = 0;
        basis[dims + d] = 0;
      }
    }
    gramSchmidt(basis, dims);

    let prevTime: number | null = null;
    let rafId: number;
    let easeT = 0; // 0 = stopped, 1 = full speed

    const animate = (time: number) => {
      if (prevTime === null) {
        prevTime = time;
        rafId = requestAnimationFrame(animate);
        return;
      }

      const dt = Math.min((time - prevTime) * 0.001, 0.1); // seconds, clamped
      prevTime = time;

      const currentExitTarget = exitTargetRef.current;

      // Advance easeT toward target
      if (currentExitTarget === null) {
        easeT = Math.min(1, easeT + dt / EASE_DURATION);
      } else {
        easeT = Math.max(0, easeT - dt / EASE_DURATION);
      }

      const easeFactor = smoothstep(easeT);
      const currentSpeed = speedRef.current;

      // Apply Givens rotations for each active dimension pair
      for (let p = 0; p < numPairs; p++) {
        const [i, j] = pairs[p]!;
        const angle = omegas[p]! * dt * currentSpeed * easeFactor * 0.0375;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        // Rotate basis rows i and j for both columns
        for (let col = 0; col < 2; col++) {
          const offset = col * dims;
          const ai = basis[offset + i]!;
          const aj = basis[offset + j]!;
          basis[offset + i] = cos * ai - sin * aj;
          basis[offset + j] = sin * ai + cos * aj;
        }
      }

      // Givens rotations preserve orthonormality, no Gram-Schmidt needed
      scatterRef.current?.setDirectBasis(basis.slice());

      // Ease-out complete — perform the deferred mode switch
      if (currentExitTarget !== null && easeT <= 0) {
        cancelAnimationFrame(rafId);
        // Store final basis so the next mode can pick up where we left off
        store.set(currentBasisAtom, new Float32Array(basis));
        if (currentExitTarget === 'guided') {
          setGuidedSuspended(true);
        }
        setGrandExitTarget(null);
        setViewMode(currentExitTarget);
        return;
      }

      rafId = requestAnimationFrame(animate);
    };

    rafId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafId);
      // Store basis on cleanup so mode transitions always have the latest
      store.set(currentBasisAtom, new Float32Array(basis));
    };
  }, [
    viewMode,
    metadata,
    scatter,
    activeIndices,
    store,
    setViewMode,
    setGrandExitTarget,
    setGuidedSuspended,
  ]);
};
