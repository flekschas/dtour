// @dtour/viewer — React UI for dtour: circular selector, preview gallery, tour controls.
import './styles.css';

// Primary API — self-contained component with spec-driven state
export { Dtour } from './Dtour.tsx';
export type { DtourProps, DtourHandle } from './Dtour.tsx';
export type { DtourSpec, EmbeddedConfig, KeyframeLoading } from './spec.ts';
export { dtourSpecSchema, DTOUR_DEFAULTS, parseEmbeddedConfig } from './spec.ts';

// Portal container — for Shadow DOM isolation (e.g. anywidget/Marimo)
export { PortalContainerContext } from './portal-container.tsx';

// Advanced composable API — for users who need granular control with their own Provider
export { DtourViewer } from './DtourViewer.tsx';
export type { DtourViewerProps } from './DtourViewer.tsx';
export { DtourToolbar } from './components/DtourToolbar.tsx';
export { CircularSlider } from './components/CircularSlider.tsx';
export type { CircularSliderProps, CircularSliderHandle } from './components/CircularSlider.tsx';
export { createDefaultViews } from './views.ts';

// Radial chart — quality metrics visualization
export { RadialChart, parseMetrics } from './radial-chart/index.ts';
export type { RadialTrackConfig, ParsedTrack, RadialChartProps } from './radial-chart/index.ts';

// Jotai atoms — for advanced users composing with DtourViewer + own Provider
export {
  // Tour
  tourPositionAtom,
  tourPlayingAtom,
  tourSpeedAtom,
  tourDirectionAtom,
  tourSliderSpacingAtom,
  arcLengthsAtom,
  // Preview
  previewCountAtom,
  previewPaddingAtom,
  selectedKeyframeAtom,
  currentKeyframeAtom,
  hoveredKeyframeAtom,
  // Point style
  pointSizeAtom,
  pointOpacityAtom,
  pointColorAtom,
  pointColorByAtom,
  colorMapAtom,
  minPointSizeAtom,
  // Camera
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  // Tour traversal
  tourTraversalAtom,
  // Legend
  showLegendAtom,
  // Axes
  showAxesAtom,
  // Keyframe numbers
  showKeyframeNumbersAtom,
  // Keyframe loadings & descriptions
  showKeyframeLoadingsAtom,
  keyframeLoadingsAtom,
  keyframeDescriptionsAtom,
  tourFamilyAtom,
  legendVisibleAtom,
  // Predefined tour
  predefinedTourAtom,
  // Theme
  themeModeAtom,
  resolvedThemeAtom,
  // Read-only
  metadataAtom,
  embeddedConfigAtom,
} from './state/atoms.ts';
