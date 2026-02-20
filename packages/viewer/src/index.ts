// @dtour/viewer — React UI for dtour: circular selector, preview gallery, tour controls.
import './styles.css';

export { Dtour } from './Dtour.tsx';
export type { DtourProps } from './Dtour.tsx';
export { CircularSelector } from './circular-range-selector.tsx';
export type { CircularSelectorProps } from './circular-range-selector.tsx';
export { createDefaultBases } from './bases.ts';
export { DtourToolbar } from './components/DtourToolbar.tsx';

// Jotai atoms — state groups for MCP-web integration
export {
  // Tour
  tourPositionAtom,
  tourPlayingAtom,
  tourSpeedAtom,
  tourDirectionAtom,
  // View
  viewCountAtom,
  galleryPaddingAtom,
  selectedKeyframeAtom,
  // Point style
  pointSizeAtom,
  pointOpacityAtom,
  pointColorAtom,
  // Camera
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  // Read-only
  metadataAtom,
} from './state/atoms.ts';
