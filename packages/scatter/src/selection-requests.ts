import type { MainToData } from './data/messages.ts';
import type { MainToGpu } from './gpu/messages.ts';

/**
 * Selection requests of a scatter instance. Each request gets an increasing id
 * that its `selectionResult` or `columnSelectionResult` repeats, so a result
 * whose id is below `latestSelectionId()` belongs to a replaced selection.
 * Selecting the same column values again without another selection request in
 * between changes nothing and returns the id of the earlier request.
 */
export const createSelectionRequests = (
  sendToData: (msg: MainToData, transfers?: Transferable[]) => void,
  sendToGpu: (msg: MainToGpu, transfers?: Transferable[]) => void,
) => {
  let latestId = 0;
  // The latest request's column and values when it selected by column
  let latestColumnKey: string | null = null;

  const nextId = (columnKey: string | null = null): number => {
    latestColumnKey = columnKey;
    latestId += 1;
    return latestId;
  };

  return {
    selectByColumn: (
      column: string,
      opts: { labelIndices?: number[]; valueRanges?: Float32Array },
    ): number => {
      // Clone valueRanges before transferring so the caller's buffer isn't detached
      const ranges = opts.valueRanges ? new Float32Array(opts.valueRanges) : undefined;
      const key = JSON.stringify([column, opts.labelIndices ?? null, ranges ? [...ranges] : null]);
      if (key === latestColumnKey) return latestId;
      const id = nextId(key);
      sendToData(
        {
          type: 'selectByColumn',
          id,
          column,
          labelIndices: opts.labelIndices,
          valueRanges: ranges,
        },
        ranges ? [ranges.buffer] : [],
      );
      return id;
    },
    setSelectionMask: (mask: Uint32Array): number => {
      const id = nextId();
      sendToGpu({ type: 'setSelectionMask', mask, id }, [mask.buffer]);
      return id;
    },
    lassoSelect: (polygon: Float32Array): number => {
      const id = nextId();
      sendToGpu({ type: 'lassoSelect', polygon, id }, [polygon.buffer]);
      return id;
    },
    clearSelection: (): number => {
      const id = nextId();
      sendToGpu({ type: 'clearSelectionMask', id });
      return id;
    },
    latestSelectionId: (): number => latestId,
    /** New data starts without a selection, so pending results become outdated. */
    reset: (): void => {
      nextId();
    },
  };
};
