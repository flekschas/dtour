import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtomValue, useStore } from 'jotai';
import { useEffect, useRef } from 'react';
import { legendSelectionAtom, linkAtom, metadataAtom, pointColorByAtom } from '../state/atoms.ts';

/**
 * A selection as linked views share it: the bit-packed selected rows, and for
 * a legend selection of a categorical column, its labels. A mask without set
 * bits is no selection.
 */
type LinkedSelection = { mask: Uint32Array; labels?: { column: string; values: string[] } };

type LinkMessage =
  // `to` addresses the reply to a view that asked to join
  | { type: 'selection'; rowCount: number; selection: LinkedSelection; to?: string }
  | { type: 'join'; rowCount: number; from: string };

const isEmpty = (mask: Uint32Array): boolean => mask.every((word) => word === 0);

/**
 * Shares this view's selections with the views in the same browser that have
 * the same `link` and number of rows, and applies theirs. Linked views must
 * show the same rows in the same order. A view that joins a link without a
 * selection adopts the selection of the linked views, unless it selects
 * something itself first.
 */
export const useLink = (scatter: ScatterInstance | null) => {
  const link = useAtomValue(linkAtom);
  const metadata = useAtomValue(metadataAtom);
  const store = useStore();
  // This view's selection, kept while the link changes so other views can adopt it
  const currentRef = useRef<LinkedSelection | null>(null);
  // The latest selection request whose result arrived. A newer request is pending.
  const settledIdRef = useRef(0);

  // New data starts without a selection
  // biome-ignore lint/correctness/useExhaustiveDependencies: metadata changes with each dataset
  useEffect(() => {
    currentRef.current = null;
    settledIdRef.current = scatter?.latestSelectionId() ?? 0;
  }, [metadata]);

  useEffect(() => {
    if (!scatter || !metadata) return;
    const { rowCount } = metadata;
    const viewId = crypto.randomUUID();
    const channel = link ? new BroadcastChannel(`dtour-link:${link}`) : null;
    const post = (message: LinkMessage) => channel?.postMessage(message);

    // The latest selection request made for a linked view, whose result must not be sent
    // back. Results of older requests are outdated anyway.
    let receivedId = 0;
    // The latest selection request when this view asked to join, or null when it doesn't adopt
    let joinedAt: number | null = null;

    const report = (selection: LinkedSelection, id: number) => {
      const received = id === receivedId;
      settledIdRef.current = Math.max(settledIdRef.current, id);
      if (id < scatter.latestSelectionId()) return;
      currentRef.current = isEmpty(selection.mask) ? null : selection;
      if (!received) post({ type: 'selection', rowCount, selection });
    };

    const unsubscribe = scatter.subscribe((status: ScatterStatus) => {
      if (status.type === 'selectionResult') {
        report({ mask: status.mask }, status.id);
      } else if (status.type === 'columnSelectionResult') {
        const labels = metadata.categoricalLabels[status.column];
        const values = labels && status.labelIndices?.map((i) => labels[i]!);
        report(
          { mask: status.mask, labels: values && { column: status.column, values } },
          status.id,
        );
      }
    });

    const apply = ({ mask, labels }: LinkedSelection) => {
      // A legend selection shows in the legend when this view is colored by its column
      if (labels && store.get(pointColorByAtom) === labels.column) {
        const wanted = new Set(labels.values);
        const labelIndices = (metadata.categoricalLabels[labels.column] ?? []).flatMap(
          (label, i) => (wanted.has(label) ? [i] : []),
        );
        if (labelIndices.length > 0) {
          receivedId = scatter.selectByColumn(labels.column, { labelIndices });
          store.set(legendSelectionAtom, new Set(labelIndices));
          return;
        }
      }
      store.set(legendSelectionAtom, null);
      receivedId = isEmpty(mask)
        ? scatter.clearSelection()
        : scatter.setSelectionMask(new Uint32Array(mask));
    };

    if (channel) {
      channel.onmessage = ({ data }: MessageEvent<LinkMessage>) => {
        if (data.rowCount !== rowCount) return;
        if (data.type === 'join') {
          const current = currentRef.current;
          if (current) post({ type: 'selection', rowCount, selection: current, to: data.from });
          return;
        }
        if (data.to !== undefined) {
          // Adopt one reply, unless this view selected something since it asked
          if (data.to !== viewId || joinedAt !== scatter.latestSelectionId()) return;
          joinedAt = null;
        }
        apply(data.selection);
      };
      // Adopt the linked views' selection only without a selection of its own, pending or shown
      const latestId = scatter.latestSelectionId();
      if (!currentRef.current && latestId === settledIdRef.current) joinedAt = latestId;
      post({ type: 'join', rowCount, from: viewId });
    }

    return () => {
      channel?.close();
      unsubscribe();
    };
  }, [link, scatter, metadata, store]);
};
