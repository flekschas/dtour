import type { Metadata } from '@dtour/scatter';
import { atom } from 'jotai';

// ---------------------------------------------------------------------------
// Tour state — controls position and playback along the tour path
// ---------------------------------------------------------------------------

export const tourPositionAtom = atom(0);
export const tourPlayingAtom = atom(false);
export const tourSpeedAtom = atom(1);
export const tourDirectionAtom = atom<1 | -1>(1);

// ---------------------------------------------------------------------------
// View state — controls gallery layout and keyframe selection
// ---------------------------------------------------------------------------

export const viewCountAtom = atom<4 | 8 | 12 | 16>(4);
export const galleryPaddingAtom = atom(12);
export const selectedKeyframeAtom = atom<number | null>(null);

// ---------------------------------------------------------------------------
// Point style — visual appearance of scatter points
// ---------------------------------------------------------------------------

export const pointSizeAtom = atom(0.012);
export const pointOpacityAtom = atom(0.7);
export const pointColorAtom = atom<[number, number, number]>([0.25, 0.5, 0.9]);

// ---------------------------------------------------------------------------
// Camera state — 2D pan and zoom
// ---------------------------------------------------------------------------

export const cameraPanXAtom = atom(0);
export const cameraPanYAtom = atom(0);
export const cameraZoomAtom = atom(1);

// ---------------------------------------------------------------------------
// Read-only / derived — not exposed to AI setters
// ---------------------------------------------------------------------------

export const metadataAtom = atom<Metadata | null>(null);
