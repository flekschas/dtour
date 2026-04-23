import type { ScatterInstance } from '@dtour/scatter';
import { useAtomValue, useStore } from 'jotai';
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
 * - Atom subscriptions use imperative store.sub() to avoid subscribing the
 *   parent component (DtourViewer) to 60fps currentBasisAtom updates.
 */
export const useSpatialIndex = (
  scatter: ScatterInstance | null,
): React.RefObject<SpatialIndex | null> => {
  const indexRef = useRef<SpatialIndex | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generationRef = useRef(0);
  const workerRef = useRef<Worker | null>(null);
  const store = useStore();
  // metadata is low-frequency (only changes on data load) — useAtomValue is fine here
  const metadata = useAtomValue(metadataAtom);

  // Keep mutable values in refs so subscription callbacks always see latest
  const scatterRef = useRef(scatter);
  scatterRef.current = scatter;
  const metadataRef = useRef(metadata);
  metadataRef.current = metadata;

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

  // Subscribe imperatively to avoid making DtourViewer re-render on every
  // currentBasisAtom update (which fires at 60fps during guided animation).
  useEffect(() => {
    const scheduleRebuild = () => {
      const isPlaying = store.get(tourPlayingAtom);
      const viewMode = store.get(viewModeAtom);
      const isAnimating = isPlaying || viewMode === 'grand';

      indexRef.current = null;
      generationRef.current++;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }

      const sc = scatterRef.current;
      const meta = metadataRef.current;
      if (isAnimating || !sc || !meta || meta.rowCount === 0) return;

      const n = meta.rowCount;
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        const gen = ++generationRef.current;
        sc.getProjectedPositions().then((positions) => {
          if (gen !== generationRef.current || !workerRef.current) return;
          workerRef.current.postMessage({ positions, nodeSize: nodeSize(n), generation: gen }, [
            positions.buffer,
          ]);
        });
      }, debounceMs(n));
    };

    const unsubBasis = store.sub(currentBasisAtom, scheduleRebuild);
    // Use scheduleRebuild (not a separate invalidate) so that the transition
    // from animating → idle also schedules a rebuild, not just a null-out.
    const unsubPlaying = store.sub(tourPlayingAtom, scheduleRebuild);
    const unsubViewMode = store.sub(viewModeAtom, scheduleRebuild);

    // Trigger initial build if basis is already set when the hook mounts
    const initialBasis = store.get(currentBasisAtom);
    if (initialBasis) scheduleRebuild();

    return () => {
      unsubBasis();
      unsubPlaying();
      unsubViewMode();
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [store]);

  return indexRef;
};
