import type { WritableAtom } from 'jotai';
import { useStore } from 'jotai';
import { useEffect, useRef } from 'react';
import type { DtourSpec } from '../spec.ts';
import {
  backgroundColorAtom,
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  previewPaddingAtom,
  previewScaleAtom,
  pointColorAtom,
  pointOpacityAtom,
  pointSizeAtom,
  tourDirectionAtom,
  tourPlayingAtom,
  tourPositionAtom,
  tourSpeedAtom,
  previewCountAtom,
  viewModeAtom,
} from './atoms.ts';

// ---------------------------------------------------------------------------
// Spec ↔ Atom mapping
// ---------------------------------------------------------------------------

type SpecEntry<S, A> = {
  atom: WritableAtom<A, [A], void>;
  toAtom: (v: S) => A;
  fromAtom: (v: A) => S;
};

function identity<T>(v: T): T {
  return v;
}

function entry<T>(atom: WritableAtom<T, [T], void>): SpecEntry<T, T> {
  return { atom, toAtom: identity, fromAtom: identity };
}

const SPEC_ATOM_MAP = {
  tourPosition: entry(tourPositionAtom),
  tourPlaying: entry(tourPlayingAtom),
  tourSpeed: entry(tourSpeedAtom),
  tourDirection: {
    atom: tourDirectionAtom,
    toAtom: (v: 'forward' | 'backward') => (v === 'forward' ? 1 : -1) as 1 | -1,
    fromAtom: (v: 1 | -1): 'forward' | 'backward' => (v === 1 ? 'forward' : 'backward'),
  },
  previewCount: entry(previewCountAtom),
  previewScale: entry(previewScaleAtom),
  previewPadding: entry(previewPaddingAtom),
  pointSize: entry(pointSizeAtom),
  pointOpacity: entry(pointOpacityAtom),
  pointColor: entry(pointColorAtom),
  backgroundColor: entry(backgroundColorAtom),
  cameraPanX: entry(cameraPanXAtom),
  cameraPanY: entry(cameraPanYAtom),
  cameraZoom: entry(cameraZoomAtom),
  viewMode: entry(viewModeAtom),
} as const;

type SpecKey = keyof typeof SPEC_ATOM_MAP;
const SPEC_KEYS = Object.keys(SPEC_ATOM_MAP) as SpecKey[];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function shallowEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) return false;
    }
    return true;
  }
  return false;
}

/** Read all atoms from the store and build a full resolved spec. */
function readFullSpec(store: ReturnType<typeof useStore>): Required<DtourSpec> {
  const spec = {} as Record<string, unknown>;
  for (const key of SPEC_KEYS) {
    const mapping = SPEC_ATOM_MAP[key];
    // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
    const raw = store.get(mapping.atom as any);
    // biome-ignore lint/suspicious/noExplicitAny: generic transform
    spec[key] = (mapping as any).fromAtom(raw);
  }
  return spec as Required<DtourSpec>;
}

// ---------------------------------------------------------------------------
// Hook: initialize store from spec (called once before first render)
// ---------------------------------------------------------------------------

export function initStoreFromSpec(
  store: ReturnType<typeof useStore>,
  spec: DtourSpec | undefined,
): void {
  if (!spec) return;
  for (const key of SPEC_KEYS) {
    const value = spec[key];
    if (value !== undefined) {
      const mapping = SPEC_ATOM_MAP[key];
      // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
      store.set(mapping.atom as any, (mapping as any).toAtom(value));
    }
  }
}

// ---------------------------------------------------------------------------
// Hook: bidirectional spec ↔ atoms sync
// ---------------------------------------------------------------------------

const DEBOUNCE_MS = 250;

export function useSpecSync(
  spec: DtourSpec | undefined,
  onSpecChange: ((spec: Required<DtourSpec>) => void) | undefined,
): void {
  const store = useStore();
  const suppressRef = useRef(false);
  const onSpecChangeRef = useRef(onSpecChange);
  onSpecChangeRef.current = onSpecChange;

  // Inbound: spec prop → atoms
  useEffect(() => {
    if (!spec) return;
    suppressRef.current = true;

    for (const key of SPEC_KEYS) {
      const value = spec[key];
      if (value === undefined) continue;
      const mapping = SPEC_ATOM_MAP[key];
      // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
      const atomValue = (mapping as any).toAtom(value);
      // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
      const current = store.get(mapping.atom as any);
      if (!shallowEqual(atomValue, current)) {
        // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
        store.set(mapping.atom as any, atomValue);
      }
    }

    queueMicrotask(() => {
      suppressRef.current = false;
    });
  }, [spec, store]);

  // Outbound: atoms → onSpecChange (debounced)
  useEffect(() => {
    if (!onSpecChangeRef.current) return;

    let dirty = false;
    let timerId: ReturnType<typeof setTimeout> | null = null;

    const flush = () => {
      timerId = null;
      if (suppressRef.current || !dirty) return;
      dirty = false;
      onSpecChangeRef.current?.(readFullSpec(store));
    };

    const scheduleFlush = () => {
      dirty = true;
      if (timerId === null) {
        timerId = setTimeout(flush, DEBOUNCE_MS);
      }
    };

    const unsubs: (() => void)[] = [];
    for (const key of SPEC_KEYS) {
      const mapping = SPEC_ATOM_MAP[key];
      // biome-ignore lint/suspicious/noExplicitAny: generic atom mapping
      unsubs.push(store.sub(mapping.atom as any, scheduleFlush));
    }

    return () => {
      for (const unsub of unsubs) unsub();
      if (timerId !== null) clearTimeout(timerId);
    };
  }, [store]);
}
