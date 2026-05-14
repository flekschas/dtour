// @dtour/viewer — React UI for dtour: circular selector, preview gallery, tour controls.
import './styles.css';

export type { CircularSliderHandle, CircularSliderProps } from './components/CircularSlider.tsx';
export { CircularSlider } from './components/CircularSlider.tsx';
export { DtourToolbar } from './components/DtourToolbar.tsx';
export type { DtourHandle, DtourProps } from './Dtour.tsx';
// Primary API — self-contained component with spec-driven state
export { Dtour } from './Dtour.tsx';
export type { DtourViewerProps } from './DtourViewer.tsx';
// Advanced composable API — for users who need granular control with their own Provider
export { DtourViewer } from './DtourViewer.tsx';
// Portal container — for Shadow DOM isolation (e.g. anywidget/Marimo)
export { PortalContainerContext } from './portal-container.tsx';
export type { ParsedTrack, RadialChartProps, RadialTrackConfig } from './radial-chart/index.ts';
// Radial chart — quality metrics visualization
export { parseMetrics, RadialChart } from './radial-chart/index.ts';
export type { DtourSpec, EmbeddedConfig, KeyframeLoading } from './spec.ts';
export { DTOUR_DEFAULTS, dtourSpecSchema, parseEmbeddedConfig } from './spec.ts';
// Jotai atoms — for advanced users composing with DtourViewer + own Provider
export {
  arcLengthsAtom,
  // Camera
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  colorMapAtom,
  currentKeyframeAtom,
  embeddedConfigAtom,
  hoveredKeyframeAtom,
  keyframeDescriptionsAtom,
  keyframeLoadingsAtom,
  legendVisibleAtom,
  // Read-only
  metadataAtom,
  minPointSizeAtom,
  panZoomModeAtom,
  pointColorAtom,
  pointColorByAtom,
  pointOpacityAtom,
  // Point style
  pointSizeAtom,
  // Predefined tour
  predefinedTourAtom,
  // Preview
  previewCountAtom,
  previewPaddingAtom,
  resolvedThemeAtom,
  selectedKeyframeAtom,
  // Axes
  showAxesAtom,
  // Keyframe loadings & descriptions
  showKeyframeLoadingsAtom,
  // Keyframe numbers
  showKeyframeNumbersAtom,
  // Legend
  showLegendAtom,
  // Slider visibility
  sliderVisibilityAtom,
  // Theme
  themeModeAtom,
  tourDirectionAtom,
  tourFamilyAtom,
  tourPlayingAtom,
  // Tour
  tourPositionAtom,
  tourSliderSpacingAtom,
  tourSpeedAtom,
  // Tour traversal
  tourTraversalAtom,
} from './state/atoms.ts';
export { createDefaultViews } from './views.ts';
