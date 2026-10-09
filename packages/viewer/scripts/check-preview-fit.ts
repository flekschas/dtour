/**
 * Regression checks for how many previews the gallery shows and which
 * keyframes they show. Run with `pnpm --filter @dtour/viewer check:preview-fit`.
 */
import assert from 'node:assert/strict';
import { createStore } from 'jotai';
import {
  computeGallerySizes,
  MAX_PREVIEW_COUNT,
  MIN_PREVIEW_SIZE,
  PREVIEW_SPACING,
} from '../src/layout/gallery-positions.ts';
import {
  galleryAreaAtom,
  keyframeDescriptionsAtom,
  metadataAtom,
  previewKeyframesAtom,
  previewLabelVisibilityAtom,
  resolvedPreviewCountAtom,
  resolvedPreviewLabelVisibilityAtom,
  resolvedPreviewScaleAtom,
  suppliedTourAtom,
} from '../src/state/atoms.ts';

type Store = ReturnType<typeof createStore>;

const TOOLBAR_OFFSETS = [40, 72]; // toolbar, toolbar + description bar

/** A store with a predefined tour of three columns, as DtourViewer sets it up. */
const tourStore = (keyframeCount: number) => {
  const store = createStore();
  store.set(metadataAtom, {
    columnNames: ['a', 'b', 'c'],
    categoricalColumnNames: [],
    categoricalLabels: {},
    rowCount: 1,
    dimCount: 3,
    mins: [0, 0, 0],
    maxes: [1, 1, 1],
    ranges: [1, 1, 1],
    means: [0, 0, 0],
  });
  const keyframes = Array.from(
    { length: keyframeCount },
    () => new Float32Array([1, 0, 0, 0, 1, 0]),
  );
  store.set(suppliedTourAtom, { keyframes, dimensions: undefined });
  store.set(
    keyframeDescriptionsAtom,
    Array.from({ length: keyframeCount }, (_, i) => `Step ${i + 1}`),
  );
  return store;
};

/** Publish a measured container the way DtourViewer does. */
const measure = (store: Store, width: number, height: number, toolbarOffset: number) =>
  store.set(galleryAreaAtom, { width, height: Math.max(0, height - toolbarOffset) });

/** Preview sizes as Gallery lays them out. */
const shownSizes = (store: Store) => {
  const area = store.get(galleryAreaAtom)!;
  const count = store.get(resolvedPreviewCountAtom);
  if (count === 0) return [];
  return computeGallerySizes(
    area.width - 2 * PREVIEW_SPACING,
    area.height - 2 * PREVIEW_SPACING,
    count,
    store.get(resolvedPreviewScaleAtom),
    store.get(resolvedPreviewLabelVisibilityAtom) === 'visible',
  ).sizes;
};

// Before the container is measured, every keyframe gets a preview
{
  const store = tourStore(32);
  assert.equal(store.get(resolvedPreviewCountAtom), 32);
}

// A measured gallery without room shows no previews
for (const toolbarOffset of TOOLBAR_OFFSETS) {
  for (const keyframeCount of [2, 32]) {
    const store = tourStore(keyframeCount);
    measure(store, 600, toolbarOffset, toolbarOffset);
    assert.equal(store.get(resolvedPreviewCountAtom), 0, `${keyframeCount} keyframes`);
    assert.deepEqual(store.get(previewKeyframesAtom), []);
  }
}

// Collapsing and revealing the viewer removes and restores the previews
for (const toolbarOffset of TOOLBAR_OFFSETS) {
  const store = tourStore(32);
  measure(store, 1280, 760, toolbarOffset);
  const expanded = store.get(resolvedPreviewCountAtom);
  assert.ok(expanded > 0);
  measure(store, 1280, toolbarOffset, toolbarOffset);
  assert.equal(store.get(resolvedPreviewCountAtom), 0);
  measure(store, 1280, 760, toolbarOffset);
  assert.equal(store.get(resolvedPreviewCountAtom), expanded);
}

// Every shown preview meets the minimum size, and the sampled keyframes are
// distinct and include the first and last keyframe
let galleries = 0;
for (const keyframeCount of [2, 5, 16, 17, 32, 37, 100]) {
  for (const width of [200, 320, 375, 600, 1000, 1920]) {
    for (const height of [40, 72, 120, 200, 400, 640, 1040]) {
      for (const toolbarOffset of TOOLBAR_OFFSETS) {
        for (const labels of ['auto', 'visible', 'interactive', 'hidden'] as const) {
          const store = tourStore(keyframeCount);
          store.set(previewLabelVisibilityAtom, labels);
          measure(store, width, height, toolbarOffset);
          const where = `${keyframeCount} keyframes, ${width}×${height}, offset ${toolbarOffset}, labels ${labels}`;

          const count = store.get(resolvedPreviewCountAtom);
          assert.ok(count <= Math.min(keyframeCount, MAX_PREVIEW_COUNT), where);
          for (const size of shownSizes(store)) assert.ok(size >= MIN_PREVIEW_SIZE, where);

          const keyframes = store.get(previewKeyframesAtom);
          assert.equal(keyframes.length, count, where);
          if (count > 0) {
            assert.equal(keyframes[0], 0, where);
            assert.equal(keyframes.at(-1), keyframeCount - 1, where);
            assert.ok(
              keyframes.every((k, i) => i === 0 || k > keyframes[i - 1]!),
              where,
            );
          }
          galleries++;
        }
      }
    }
  }
}

console.log(`Preview fit checks passed (${galleries} galleries).`);
