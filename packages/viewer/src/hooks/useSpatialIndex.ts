import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue } from 'jotai';
import KDBush from 'kdbush';
import { useEffect, useRef } from 'react';
import { currentBasisAtom, metadataAtom, tourPlayingAtom, viewModeAtom } from '../state/atoms.ts';
import type { BuildResult } from '../workers/spatial-index.worker.ts';
import SpatialIndexWorkerFactory from '../workers/spatial-index.worker.ts?worker&inline';

/** Dynamic nodeSize — larger leaves = faster build, negligible query cost impact. */
const nodeSize = (n: number): number => {
  if (n <= 100_000) return 64;
  if (n <= 1_000_000) return 256;
  if (n <= 5_000_000) return 1024;
  if (n <= 10_000_000) return 2048;
  return 4096;
};

/** Dynamic debounce — scale with point count since build takes longer. */
const debounceMs = (n: number): number => {
  if (n <= 1_000_000) return 250;
  if (n <= 5_000_000) return 250 + ((n - 1_000_000) / 4_000_000) * 250; // 250→500 linearly
  return 500;
};

export type SpatialIndex = {
  index: KDBush;
  positions: Float32Array;
};

/**
 * Builds a kdbush spatial index over projected 2D positions in a Web Worker.
 *
 * - Rebuilds on debounce after the projected basis settles (covers guided
 *   scrub, manual axis drag, and mode switches).
 * - During playback and grand tour the index is invalidated — both drive
 *   continuous animation that would immediately stale any index.
 * - In-flight builds are discarded (via generation counter) when the
 *   projection changes before the build completes. The worker still finishes
 *   the stale build — true cancellation inside a Web Worker would require
 *   terminate + respawn or chunked processing, neither worth the complexity.
 * - Zoom/pan does NOT trigger rebuild — positions are in projection space,
 *   and queryNearest converts mouse coords to that space at query time.
 */
export const useSpatialIndex = (
  scatter: ScatterInstance | null,
): React.RefObject<SpatialIndex | null> => {
  const indexRef = useRef<SpatialIndex | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generationRef = useRef(0);
  const workerRef = useRef<Worker | null>(null);
  const metadata = useAtomValue(metadataAtom);
  const isPlaying = useAtomValue(tourPlayingAtom);
  const viewMode = useAtomValue(viewModeAtom);
  const currentBasis = useAtomValue(currentBasisAtom);

  // Projection is continuously animated during playback or grand tour
  const isAnimating = isPlaying || viewMode === 'grand';

  // Spin up worker once, tear down on unmount
  useEffect(() => {
    const worker = new SpatialIndexWorkerFactory();
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<BuildResult>) => {
      const { indexData, positions, generation } = e.data;
      // Discard stale results
      if (generation !== generationRef.current) return;
      indexRef.current = {
        index: KDBush.from(indexData),
        positions,
      };
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  // Invalidate index and cancel pending builds during animation
  useEffect(() => {
    if (isAnimating) {
      indexRef.current = null;
      generationRef.current++;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [isAnimating]);

  // Rebuild on debounce after projection settles.
  // currentBasis covers all modes: guided (interpolation), manual (axis drag),
  // grand (animation — caught by isAnimating guard above).
  // biome-ignore lint/correctness/useExhaustiveDependencies: currentBasis is a trigger — we re-run when the basis changes, but read positions from the worker
  useEffect(() => {
    if (isAnimating || !scatter || !metadata) return;

    // Invalidate stale index and bump generation to discard any in-flight build
    indexRef.current = null;
    generationRef.current++;

    const n = metadata.rowCount;
    if (n === 0) return;

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      timerRef.current = null;

      // Capture current generation before the async gap
      const gen = ++generationRef.current;

      scatter.getProjectedPositions().then((positions) => {
        // Check if still current after async readback
        if (gen !== generationRef.current || !workerRef.current) return;

        workerRef.current.postMessage({ positions, nodeSize: nodeSize(n), generation: gen }, [
          positions.buffer,
        ]);
      });
    }, debounceMs(n));

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [currentBasis, isAnimating, scatter, metadata]);

  return indexRef;
};
