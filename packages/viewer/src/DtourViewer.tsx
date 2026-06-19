import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import {
  computeArcLengths,
  createScatter,
  createScatterWebGL,
  detectBackend,
  GLASBEY_DARK,
  GLASBEY_LIGHT,
  interpolateAtPosition,
  OKABE_ITO,
} from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AxisOverlayHandle } from './components/AxisOverlay.tsx';
import { AxisOverlay } from './components/AxisOverlay.tsx';
import type { CircularSliderHandle } from './components/CircularSlider.tsx';
import { CircularSlider } from './components/CircularSlider.tsx';
import { Gallery } from './components/Gallery.tsx';
import { LassoOverlay } from './components/LassoOverlay.tsx';
import { RevertCameraButton } from './components/RevertCameraButton.tsx';
import { useAnimatePosition } from './hooks/useAnimatePosition.ts';
import { useGrandTour } from './hooks/useGrandTour.ts';
import { useGuidedResume } from './hooks/useGuidedResume.ts';
import { usePlayback } from './hooks/usePlayback.ts';
import { useScatter } from './hooks/useScatter.ts';
import { useSpatialIndex } from './hooks/useSpatialIndex.ts';
import { computeSelectorSize } from './layout/selector-size.ts';
import {
  arcballQuat,
  IDENTITY_QUAT,
  isIdentityQuat,
  multiplyQuat,
  projectToSphere,
  type Quat,
  quatToMat3,
  slerp,
} from './lib/arcball.ts';
import { tourToVisual, visualToTour } from './lib/position-remap.ts';
import { throttleAndDebounce } from './lib/throttle-debounce.ts';
import { parseMetrics } from './radial-chart/parse-metrics.ts';
import { RadialChart } from './radial-chart/RadialChart.tsx';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import {
  activeColumnsAtom,
  activeIndicesAtom,
  animationGenAtom,
  arcLengthsAtom,
  cameraPanXAtom,
  cameraPanYAtom,
  cameraZoomAtom,
  canvasSizeAtom,
  colorMapAtom,
  currentBasisAtom,
  currentKeyframeAtom,
  embeddedConfigAtom,
  guidedSuspendedAtom,
  hoveredKeyframeAtom,
  is3dRotatedAtom,
  keyframeDescriptionsAtom,
  keyframeLoadingsAtom,
  legendSelectionAtom,
  metadataAtom,
  panZoomModeAtom,
  pointColorByAtom,
  predefinedTourAtom,
  previewCentersAtom,
  previewCountAtom,
  previewScaleAtom,
  resolvedThemeAtom,
  resumeGuidedAtom,
  showAxesAtom,
  showKeyframeLoadingsAtom,
  sliderVisibilityAtom,
  tourByAtom,
  tourFamilyAtom,
  tourPlayingAtom,
  tourPositionAtom,
  tourSliderSpacingAtom,
  tourTraversalAtom,
} from './state/atoms.ts';
import { createDefaultViews, createPCAViews, expandBases } from './views.ts';

export type DtourViewerProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer | undefined;
  /** Tour keyframe views (p×2 column-major). Auto-generated if omitted. */
  views?: Float32Array[] | undefined;
  /** Arrow IPC ArrayBuffer with per-view quality metrics. */
  metrics?: ArrayBuffer | undefined;
  /** Track configuration for radial bar charts. */
  metricTracks?: RadialTrackConfig[] | undefined;
  /** Global bar width override for radial charts ('full' or px). */
  metricBarWidth?: 'full' | number | undefined;
  /** Called on every status event from the renderer. */
  onStatus?: ((status: ScatterStatus) => void) | undefined;
  /** Height in px of an overlay toolbar above the canvas. The shader shifts
   *  and scales content to center it in the visible area below the toolbar.
   *  Animates smoothly to 0 in zen mode. Default 0. */
  toolbarHeight?: number | undefined;
  /** Called when the scatter instance is created (or null on destroy). */
  onScatterReady?: ((scatter: ScatterInstance | null) => void) | undefined;
  /** Rendering backend. Resolved once on mount — changing after mount has no
   *  effect. 'auto' (the default) probes for WebGPU support (incl. the
   *  float32-blendable feature) and falls back to the WebGL2 backend otherwise. */
  backend?: 'webgpu' | 'webgl' | 'auto' | undefined;
};

const PREVIEW_INITIAL_SIZE = 2; // Placeholder; real size set by ResizeObserver in Gallery

const INSET_ANIMATION_MS = 300;
const SELECTOR_PADDING = 16;

export const DtourViewer = ({
  data,
  views,
  metrics,
  metricTracks,
  metricBarWidth,
  onStatus,
  toolbarHeight = 0,
  onScatterReady,
  backend = 'auto',
}: DtourViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Resolve 'auto' to a concrete backend via async feature detection. When an
  // explicit backend is requested we use it directly (no detection). The
  // scatter lifecycle effect below is gated on this becoming non-null.
  const [resolvedBackend, setResolvedBackend] = useState<'webgpu' | 'webgl' | null>(
    backend === 'webgpu' || backend === 'webgl' ? backend : null,
  );
  useEffect(() => {
    if (resolvedBackend) return;
    let cancelled = false;
    detectBackend().then((b) => {
      if (!cancelled) setResolvedBackend(b);
    });
    return () => {
      cancelled = true;
    };
  }, [resolvedBackend]);
  const onScatterReadyRef = useRef(onScatterReady);
  onScatterReadyRef.current = onScatterReady;
  const [scatter, setScatter] = useState<ScatterInstance | null>(null);
  const scatterRef = useRef<ScatterInstance | null>(null);
  const [previewCanvases, setPreviewCanvases] = useState<HTMLCanvasElement[]>([]);
  const [position, setPosition] = useAtom(tourPositionAtom);
  const metadata = useAtomValue(metadataAtom);
  const embeddedConfig = useAtomValue(embeddedConfigAtom);
  const previewCount = useAtomValue(previewCountAtom);
  const previewScale = useAtomValue(previewScaleAtom);
  const [tourTraversal, setTourTraversal] = useAtom(tourTraversalAtom);
  const [guidedSuspended, setGuidedSuspended] = useAtom(guidedSuspendedAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const setCanvasSize = useSetAtom(canvasSizeAtom);
  const store = useStore();
  const currentKeyframe = useAtomValue(currentKeyframeAtom);
  const hoveredKeyframe = useAtomValue(hoveredKeyframeAtom);
  const previewCenters = useAtomValue(previewCentersAtom);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const activeIndices = useAtomValue(activeIndicesAtom);
  const setActiveColumns = useSetAtom(activeColumnsAtom);
  const lastDataRef = useRef<ArrayBuffer | undefined>(undefined);
  const prevDimCountRef = useRef<number | null>(null);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  const sliderRef = useRef<CircularSliderHandle>(null);
  const axisOverlayRef = useRef<AxisOverlayHandle>(null);
  const positionRef = useRef(position);

  // Sync positionRef with atom value (overwritten by direct drives within ~33ms)
  useEffect(() => {
    positionRef.current = position;
  }, [position]);

  // Throttle+debounce atom write — fires at most every 100ms during playback
  // (throttle) AND once more after the last tick (debounce), so Gallery's
  // currentKeyframe highlight updates during playback (~10fps) while also
  // guaranteeing the final position is flushed.
  const positionFlushRef = useRef(throttleAndDebounce((pos: number) => setPosition(pos), 100));

  const schedulePositionFlush = useCallback(() => {
    positionFlushRef.current(positionRef.current);
  }, []);

  // Stable ref so the scatter subscribe callback can access the latest flush fn
  const scheduleFlushRef = useRef(schedulePositionFlush);
  scheduleFlushRef.current = schedulePositionFlush;

  // Delegate playback rAF to the GPU worker
  usePlayback(scatter);

  // Flush position to atom immediately when playback stops
  const playing = useAtomValue(tourPlayingAtom);
  const prevPlayingRef = useRef(false);
  useEffect(() => {
    if (prevPlayingRef.current && !playing) {
      positionFlushRef.current.cancel();
      positionFlushRef.current.reset();
      setPosition(positionRef.current);
    }
    prevPlayingRef.current = playing;
  }, [playing, setPosition]);

  const setCurrentBasis = useSetAtom(currentBasisAtom);
  const tourBy = useAtomValue(tourByAtom);
  const tourFamily = useAtomValue(tourFamilyAtom);
  const [pcaResult, setPcaResult] = useState<{
    eigenvectors: Float32Array[];
    numDims: number;
  } | null>(null);

  const showAxes = useAtomValue(showAxesAtom);
  const sliderVisibility = useAtomValue(sliderVisibilityAtom);
  const spacingMode = useAtomValue(tourSliderSpacingAtom);
  const setArcLengthsAtom_ = useSetAtom(arcLengthsAtom);
  const isGuidedMode = tourTraversal === 'guided';
  const keyframeLoadings = useAtomValue(keyframeLoadingsAtom);
  const keyframeDescriptions = useAtomValue(keyframeDescriptionsAtom);
  const showKeyframeLoadings = useAtomValue(showKeyframeLoadingsAtom);
  const loadingsVisible =
    showKeyframeLoadings && keyframeLoadings !== null && keyframeLoadings.length > 0;
  const descriptionsVisible =
    !loadingsVisible && keyframeDescriptions !== null && Array.isArray(keyframeDescriptions);
  const showBarSpace = loadingsVisible || descriptionsVisible;

  // Resolve views (from props or auto-generated) and precompute arc lengths
  // so we can track the current tour basis on the main thread.
  // Embedded tour views from Parquet metadata.
  const embeddedViews =
    embeddedConfig?.tour && metadata && embeddedConfig.tour.keyframes.length > 0
      ? embeddedConfig.tour.keyframes
      : null;

  // Track whether the active tour is predefined (externally computed) vs auto-generated.
  // Predefined tours lock down column toggles, preview count, and the Dims/PCA selector.
  const setPredefinedTour = useSetAtom(predefinedTourAtom);
  useEffect(() => {
    const predefinedViews = views ?? embeddedViews;
    if (predefinedViews && predefinedViews.length > 0 && metadata) {
      // Resolve dimensions: use tour.dimensions if available, fall back
      // to the first nDims column names (derived from basis matrix size).
      const tourNDims = predefinedViews[0]!.length / 2;
      const dims = embeddedConfig?.tour?.dimensions ?? metadata.columnNames.slice(0, tourNDims);
      setPredefinedTour({ dimensions: dims, keyframeCount: predefinedViews.length });
    } else {
      setPredefinedTour(null);
    }
  }, [views, embeddedViews, metadata, embeddedConfig, setPredefinedTour]);

  // Effective preview count: predefined tours use their own view count,
  // auto-generated tours use the user-configurable previewCount atom.
  const predefinedTour = useAtomValue(predefinedTourAtom);
  const effectivePreviewCount = predefinedTour?.keyframeCount ?? previewCount;

  const { resolvedViews, arcLengths } = useMemo(() => {
    if (!metadata || metadata.dimCount < 2) return { resolvedViews: null, arcLengths: null };
    if (activeIndices.length < 2) return { resolvedViews: null, arcLengths: null };
    const dims = metadata.dimCount;
    let rb: Float32Array[];
    if (tourBy === 'pca' && pcaResult && pcaResult.eigenvectors.length >= 2) {
      rb = createPCAViews(pcaResult.eigenvectors, dims, pcaResult.numDims, previewCount);
    } else if (views && views.length > 0) {
      const tourNDims = views[0]!.length / 2;
      if (tourNDims === dims) {
        rb = views.map((b) => new Float32Array(b));
      } else if (tourNDims < dims) {
        const tourDims = metadata.columnNames.slice(0, tourNDims);
        rb = expandBases(views, tourDims, metadata.columnNames, dims);
      } else {
        rb = createDefaultViews(dims, previewCount, activeIndices);
      }
    } else if (embeddedViews) {
      const tourNDims = embeddedViews[0]!.length / 2;
      if (tourNDims === dims) {
        rb = embeddedViews.map((b) => new Float32Array(b));
      } else if (tourNDims < dims) {
        const tourDims =
          embeddedConfig?.tour?.dimensions ?? metadata.columnNames.slice(0, tourNDims);
        rb = expandBases(embeddedViews, tourDims, metadata.columnNames, dims);
      } else {
        rb = createDefaultViews(dims, previewCount, activeIndices);
      }
    } else {
      rb = createDefaultViews(dims, previewCount, activeIndices);
    }
    return {
      resolvedViews: rb,
      arcLengths: computeArcLengths(rb, dims, tourFamily !== 'sequential'),
    };
  }, [
    views,
    embeddedViews,
    embeddedConfig,
    metadata,
    previewCount,
    activeIndices,
    tourBy,
    tourFamily,
    pcaResult,
  ]);

  // Sync arcLengths atom so Gallery and other components can access it
  useEffect(() => {
    setArcLengthsAtom_(arcLengths);
  }, [arcLengths, setArcLengthsAtom_]);

  // Refs for spacing mode and arcLengths so the scatter subscribe callback
  // (created once in the init effect) can access the latest values.
  const spacingModeRef = useRef(spacingMode);
  spacingModeRef.current = spacingMode;
  const arcLengthsRef = useRef(arcLengths);
  arcLengthsRef.current = arcLengths;
  const resolvedViewsRef = useRef(resolvedViews);
  resolvedViewsRef.current = resolvedViews;
  const metadataRef = useRef(metadata);
  metadataRef.current = metadata;
  const orthonormalize = tourFamily !== 'sequential';
  const orthonormalizeRef = useRef(orthonormalize);
  orthonormalizeRef.current = orthonormalize;
  // Pre-allocated scratch buffer for imperative basis interpolation
  const basisScratchRef = useRef(new Float32Array(0));

  // Imperative axis overlay update — compute the interpolated basis at
  // a given tour position and push directly to SVG via setBasis.
  // Called from playbackTick, slider drag, and wheel scrub.
  const updateAxesImperative = useCallback((tourPos: number) => {
    const rv = resolvedViewsRef.current;
    const al = arcLengthsRef.current;
    const meta = metadataRef.current;
    if (!rv || !al || !meta || !axisOverlayRef.current) return;
    const p = meta.dimCount;
    if (basisScratchRef.current.length !== p * 2) {
      basisScratchRef.current = new Float32Array(p * 2);
    }
    interpolateAtPosition(basisScratchRef.current, rv, al, p, tourPos, orthonormalizeRef.current);
    axisOverlayRef.current.setBasis(basisScratchRef.current);
  }, []);
  const updateAxesRef = useRef(updateAxesImperative);
  updateAxesRef.current = updateAxesImperative;

  // Keep currentBasisAtom in sync with the tour interpolation so other
  // modes (manual, grand) can initialize from the current projection.
  // Only update in guided mode — in manual/grand the atom is owned by
  // AxisOverlay / useGrandTour respectively.
  // Skip when guidedSuspended: the GPU is still showing a directBasis
  // from the previous mode (grand/manual), not the tour interpolation,
  // so overwriting currentBasisAtom here would cause a jump on re-entry.
  useEffect(() => {
    if (tourTraversal !== 'guided') return;
    if (guidedSuspended) return;
    if (!resolvedViews || !arcLengths || !metadata) return;
    const dims = metadata.dimCount;
    const out = new Float32Array(dims * 2);
    interpolateAtPosition(out, resolvedViews, arcLengths, dims, position, orthonormalize);
    setCurrentBasis(out);
  }, [
    tourTraversal,
    guidedSuspended,
    resolvedViews,
    arcLengths,
    metadata,
    position,
    orthonormalize,
    setCurrentBasis,
  ]);

  // Parse metrics Arrow IPC into renderable tracks
  const parsedTracks = useMemo(() => {
    if (!metrics) return [];
    return parseMetrics(metrics, metricTracks, metricBarWidth);
  }, [metrics, metricTracks, metricBarWidth]);

  // Override confusion track color: highlight by default, label palette color on single selection
  const legendSelection = useAtomValue(legendSelectionAtom);
  const pointColorBy = useAtomValue(pointColorByAtom);
  const resolvedTheme = useAtomValue(resolvedThemeAtom);
  const rawColorMap = useAtomValue(colorMapAtom);

  const coloredTracks = useMemo(() => {
    if (parsedTracks.length === 0) return parsedTracks;
    const confusionIdx = parsedTracks.findIndex((t) => t.label === 'confusion');
    if (confusionIdx === -1) return parsedTracks;

    let confusionColor = resolvedTheme === 'light' ? '#000000' : '#ffffff';

    if (
      legendSelection &&
      legendSelection.size === 1 &&
      pointColorBy &&
      metadata?.categoricalColumnNames.includes(pointColorBy)
    ) {
      const selectedIndex = legendSelection.values().next().value as number;
      const labels = metadata.categoricalLabels[pointColorBy] ?? [];
      const selectedLabel = labels[selectedIndex];

      if (selectedLabel && rawColorMap?.[selectedLabel]) {
        const v = rawColorMap[selectedLabel]!;
        confusionColor = typeof v === 'string' ? v : v[resolvedTheme];
      } else {
        const glasbey = resolvedTheme === 'light' ? GLASBEY_LIGHT : GLASBEY_DARK;
        const colors =
          labels.length <= OKABE_ITO.length
            ? OKABE_ITO
            : ([...OKABE_ITO, ...glasbey] as [number, number, number][]);
        const [r, g, b] = colors[selectedIndex % colors.length]!;
        confusionColor = `rgb(${r},${g},${b})`;
      }
    }

    const currentTrack = parsedTracks[confusionIdx]!;
    if (currentTrack.color === confusionColor) return parsedTracks;

    const result = [...parsedTracks];
    result[confusionIdx] = { ...currentTrack, color: confusionColor };
    return result;
  }, [parsedTracks, legendSelection, pointColorBy, metadata, resolvedTheme, rawColorMap]);

  // Bridge Jotai atoms (style, camera) → scatter instance
  useScatter(scatter);

  // Spatial index for hover picking
  const spatialIndexRef = useSpatialIndex(scatter);

  const isToolbarVisible = toolbarHeight > 0 && tourTraversal !== 'grand';
  const effectiveToolbarHeight = isToolbarVisible ? toolbarHeight : 0;

  // Animate camera inset when the toolbar appears/disappears (grand toggle).
  // The shader shifts + scales content to center it below the toolbar.
  // We also track the current pixel offset for positioning overlays.
  const [overlayOffsetY, setOverlayOffsetY] = useState(isToolbarVisible ? toolbarHeight : 0);
  const overlayOffsetRef = useRef(overlayOffsetY);
  overlayOffsetRef.current = overlayOffsetY;
  const insetAnimRef = useRef<number | null>(null);

  useEffect(() => {
    if (!scatter || containerSize.height === 0) return;

    const targetT = tourTraversal === 'grand' || toolbarHeight === 0 ? 0 : 1;
    const h = containerSize.height;
    const t = toolbarHeight;

    // Current inset factor: derive from current overlayOffsetY via ref
    const startT = t > 0 ? overlayOffsetRef.current / t : 0;
    if (Math.abs(startT - targetT) < 0.001) {
      // Already at target — just ensure shader is in sync
      const insetOffsetY = (-targetT * t) / h;
      const insetZoom = 1 - (targetT * t) / h;
      scatter.setCamera({ insetOffsetY, insetZoom } as Parameters<typeof scatter.setCamera>[0]);
      return;
    }

    const startTime = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / INSET_ANIMATION_MS);
      // ease-in-out cubic
      const eased =
        progress < 0.5 ? 4 * progress * progress * progress : 1 - (-2 * progress + 2) ** 3 / 2;

      const currentT = startT + (targetT - startT) * eased;

      // Shader inset: shift down and scale to fit visible area
      const insetOffsetY = (-currentT * t) / h;
      const insetZoom = 1 - (currentT * t) / h;
      scatter.setCamera({ insetOffsetY, insetZoom } as Parameters<typeof scatter.setCamera>[0]);

      // Overlay pixel offset
      setOverlayOffsetY(currentT * t);

      if (progress < 1) {
        insetAnimRef.current = requestAnimationFrame(tick);
      } else {
        insetAnimRef.current = null;
      }
    };

    if (insetAnimRef.current !== null) cancelAnimationFrame(insetAnimRef.current);
    insetAnimRef.current = requestAnimationFrame(tick);

    return () => {
      if (insetAnimRef.current !== null) {
        cancelAnimationFrame(insetAnimRef.current);
        insetAnimRef.current = null;
      }
    };
  }, [scatter, tourTraversal, toolbarHeight, containerSize.height]);

  // Grand mode: Givens-rotation grand tour
  useGrandTour(scatter, tourTraversal, metadata);

  // Largest selector diameter that doesn't overlap any gallery preview
  const selectorSize = useMemo(
    () =>
      computeSelectorSize(
        containerSize.width,
        containerSize.height - effectiveToolbarHeight,
        effectivePreviewCount,
        0,
        SELECTOR_PADDING,
        previewScale,
        coloredTracks.length,
        showBarSpace,
      ),
    [
      containerSize.width,
      containerSize.height,
      effectivePreviewCount,
      effectiveToolbarHeight,
      previewScale,
      coloredTracks.length,
      showBarSpace,
    ],
  );

  // Effect A — Scatter lifecycle: create main canvas + scatter instance.
  // Runs once, after backend detection resolves (resolvedBackend goes
  // null→concrete exactly once and never changes after, so this fires a single
  // time). store and setCanvasSize are stable singletons.
  // NOTE: This effect is NOT StrictMode-safe. transferControlToOffscreen()
  // and ArrayBuffer transfers are one-shot ownership operations that cannot
  // survive StrictMode's mount→cleanup→remount cycle. Consumers must either
  // avoid StrictMode or accept a one-time dev-mode data copy.
  // biome-ignore lint/correctness/useExhaustiveDependencies: gated on resolvedBackend; other captured values are stable or refs
  useEffect(() => {
    if (!resolvedBackend) return;
    const container = containerRef.current;
    if (!container) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = container.getBoundingClientRect();

    // Main canvas — fills container
    const mainCanvas = document.createElement('canvas');
    mainCanvas.width = Math.round(rect.width * dpr);
    mainCanvas.height = Math.round(rect.height * dpr);
    mainCanvas.style.width = '100%';
    mainCanvas.style.height = '100%';
    mainCanvas.style.display = 'block';
    container.prepend(mainCanvas);

    const factory = resolvedBackend === 'webgl' ? createScatterWebGL : createScatter;
    const instance = factory({
      canvas: mainCanvas,
      zoom: store.get(cameraZoomAtom),
    });
    scatterRef.current = instance;
    setScatter(instance);
    onScatterReadyRef.current?.(instance);
    // Expose scatter instance for dev tools and benchmark automation
    if (import.meta.env.DEV || (globalThis as Record<string, unknown>).__dtourBenchmarkMode) {
      (globalThis as Record<string, unknown>).scatter = instance;
    }

    instance.subscribe((s: ScatterStatus) => {
      onStatusRef.current?.(s);
      if (s.type === 'pcaResult') {
        setPcaResult({ eigenvectors: s.eigenvectors, numDims: s.numDims });
      }
      if (s.type === 'metadata') {
        // Data reload: worker resets 3D state, so clear viewer-side refs too
        if (is3dEnabledRef.current) {
          is3dEnabledRef.current = false;
          quatRef.current = IDENTITY_QUAT;
          residualPCRef.current = null;
          axisOverlayRef.current?.clearRotation3d();
          store.set(is3dRotatedAtom, false);
        }
      }
      if (s.type === 'residualPC') {
        residualPCRef.current = s.residualPC;
        // If already rotating, immediately update axis overlay so the first
        // drag frame isn't missed while waiting for the next pointermove.
        if (!isIdentityQuat(quatRef.current)) {
          axisOverlayRef.current?.setRotation3d(s.residualPC, quatToMat3(quatRef.current));
        }
      }
      if (s.type === 'playbackTick') {
        positionRef.current = s.position;
        const al = arcLengthsRef.current;
        const visual =
          spacingModeRef.current === 'equal' && al ? tourToVisual(s.position, al) : s.position;
        sliderRef.current?.setPosition(visual);
        scheduleFlushRef.current();
        updateAxesRef.current(s.position);
      }
    });

    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const curDpr = window.devicePixelRatio || 1;
      instance.resize(0, Math.round(width * curDpr), Math.round(height * curDpr), curDpr);
      setContainerSize({ width, height });
      setCanvasSize({ width, height });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      instance.destroy();
      scatterRef.current = null;
      setScatter(null);
      onScatterReadyRef.current?.(null);
      mainCanvas.remove();
    };
  }, [resolvedBackend]);

  // Effect B — Preview canvas lifecycle: add/remove preview canvases dynamically.
  // Uses effectivePreviewCount so predefined tours create the right number of canvases.
  useEffect(() => {
    if (!scatter) return;

    const previews: HTMLCanvasElement[] = [];
    for (let i = 0; i < effectivePreviewCount; i++) {
      const c = document.createElement('canvas');
      c.width = PREVIEW_INITIAL_SIZE;
      c.height = PREVIEW_INITIAL_SIZE;
      c.style.width = '100%';
      c.style.height = '100%';
      c.style.display = 'block';
      c.style.borderRadius = '2px';
      previews.push(c);
      scatter.addPreviewCanvas(i, c);
    }
    setPreviewCanvases(previews);

    // Observe each preview canvas for CSS layout changes and resize
    // the OffscreenCanvas backing store to match (DPR-aware).
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const curDpr = window.devicePixelRatio || 1;
        const canvas = entry.target as HTMLCanvasElement;
        const idx = previews.indexOf(canvas);
        if (idx < 0) continue;
        const { width, height } = entry.contentRect;
        const pw = Math.round(width * curDpr);
        const ph = Math.round(height * curDpr);
        if (pw < 1 || ph < 1) continue;
        scatter.resizePreview(idx, pw, ph);
      }
    });
    for (const c of previews) ro.observe(c);

    scatter.render();

    return () => {
      ro.disconnect();
      for (let i = 0; i < previews.length; i++) {
        scatter.removePreviewCanvas(i);
        previews[i]!.remove();
      }
      setPreviewCanvases([]);
    };
  }, [scatter, effectivePreviewCount]);

  // Reset active columns and PCA results when a new dataset loads (different dim count)
  useEffect(() => {
    if (!metadata) return;
    if (prevDimCountRef.current !== null && prevDimCountRef.current !== metadata.dimCount) {
      setActiveColumns(null);
      setPcaResult(null);
    }
    prevDimCountRef.current = metadata.dimCount;
  }, [metadata, setActiveColumns]);

  // Send data when it changes
  useEffect(() => {
    if (!data || !scatter || data === lastDataRef.current) return;
    if (data.byteLength === 0) return; // already transferred (detached)
    lastDataRef.current = data;
    scatter.loadData(data);
  }, [data, scatter]);

  // Force guided mode when tourBy is 'parameter' (no manual/grand for parameter tours)
  useEffect(() => {
    if (tourBy === 'parameter' && tourTraversal !== 'guided') {
      setTourTraversal('guided');
    }
  }, [tourBy, tourTraversal, setTourTraversal]);

  // Trigger PCA computation when tourBy is 'pca' and data is loaded
  useEffect(() => {
    if (tourBy !== 'pca' || !metadata || metadata.dimCount < 2 || !scatter) return;
    scatter.computePCA();
  }, [tourBy, metadata, scatter]);

  // Set views when available (from props, PCA, embedded, or auto-generated from metadata).
  // Predefined tours (views prop, embeddedViews) are used as-is — column selection
  // only affects auto-generated views (createDefaultViews).
  useEffect(() => {
    if (!scatter || !metadata || metadata.dimCount < 2) return;
    if (activeIndices.length < 2) return;
    const dims = metadata.dimCount;
    let bases: Float32Array[];
    if (tourBy === 'pca' && pcaResult && pcaResult.eigenvectors.length >= 2) {
      bases = createPCAViews(pcaResult.eigenvectors, dims, pcaResult.numDims, previewCount);
    } else if (views && views.length > 0) {
      const tourNDims = views[0]!.length / 2;
      if (tourNDims === dims) {
        bases = views.map((b) => new Float32Array(b));
      } else if (tourNDims < dims) {
        const tourDims = metadata.columnNames.slice(0, tourNDims);
        bases = expandBases(views, tourDims, metadata.columnNames, dims);
      } else {
        bases = createDefaultViews(dims, previewCount, activeIndices);
      }
    } else if (embeddedViews) {
      const tourNDims = embeddedViews[0]!.length / 2;
      if (tourNDims === dims) {
        bases = embeddedViews.map((b) => new Float32Array(b));
      } else if (tourNDims < dims) {
        const tourDims =
          embeddedConfig?.tour?.dimensions ?? metadata.columnNames.slice(0, tourNDims);
        bases = expandBases(embeddedViews, tourDims, metadata.columnNames, dims);
      } else {
        bases = createDefaultViews(dims, previewCount, activeIndices);
      }
    } else {
      bases = createDefaultViews(dims, previewCount, activeIndices);
    }
    scatter.setBases(bases, tourFamily);
    scatter.render();
  }, [
    scatter,
    views,
    embeddedViews,
    embeddedConfig,
    metadata,
    previewCount,
    activeIndices,
    tourBy,
    tourFamily,
    pcaResult,
  ]);

  const { animateTo, cancelAnimation } = useAnimatePosition();
  const { resumeWithTransition, cancelTransition, isTransitioning } = useGuidedResume(
    scatterRef,
    resolvedViewsRef,
    arcLengthsRef,
    metadataRef,
  );
  // Keep stable refs so the imperative wheel handler always calls the latest version.
  const resumeWithTransitionRef = useRef(resumeWithTransition);
  resumeWithTransitionRef.current = resumeWithTransition;
  const isTransitioningRef = useRef(isTransitioning);
  isTransitioningRef.current = isTransitioning;

  // Publish resumeWithTransition to the store so sibling components (DtourToolbar)
  // can call it without holding scatter refs.
  useEffect(() => {
    store.set(resumeGuidedAtom, { fn: resumeWithTransition });
    return () => store.set(resumeGuidedAtom, null);
  }, [store, resumeWithTransition]);

  // Slider click → animated seek to the clicked position.
  // The slider reports a visual position; convert to tour position for the GPU.
  const handlePositionSeek = useCallback(
    (visualPos: number) => {
      resumeWithTransition(300);
      setPlaying(false);
      const tourPos =
        spacingMode === 'equal' && arcLengths ? visualToTour(visualPos, arcLengths) : visualPos;
      animateTo(tourPos);
    },
    [resumeWithTransition, setPlaying, animateTo, spacingMode, arcLengths],
  );

  // Slider drag start → cancel any basis transition and animation, then switch
  // to immediate updates (no transition: user is taking direct control).
  const handleDragStart = useCallback(() => {
    cancelAnimation();
    cancelTransition();
    setGuidedSuspended(false);
  }, [cancelAnimation, cancelTransition, setGuidedSuspended]);

  // Slider drag move → send directly to GPU, update slider imperatively,
  // debounce atom write to minimize React re-renders during drag.
  // The slider reports a visual position; convert to tour position for the GPU.
  const handlePositionChange = useCallback(
    (visualPos: number) => {
      setGuidedSuspended(false);
      const tourPos =
        spacingMode === 'equal' && arcLengths ? visualToTour(visualPos, arcLengths) : visualPos;
      scatterRef.current?.setTourPosition(tourPos);
      sliderRef.current?.setPosition(visualPos);
      positionRef.current = tourPos;
      updateAxesImperative(tourPos);
      schedulePositionFlush();
    },
    [setGuidedSuspended, schedulePositionFlush, spacingMode, arcLengths, updateAxesImperative],
  );

  // Wheel → zoom-about-cursor or tour scrub, depending on mode and Shift key.
  // Normal guided: scroll = tour scrub, Shift+scroll = zoom.
  // Pan/zoom guided (or manual/grand): scroll = zoom, Shift+scroll = tour scrub.
  // Imperative listener with { passive: false } so preventDefault() works.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handler = (e: WheelEvent) => {
      const traversal = store.get(tourTraversalAtom);
      const isPanZoom = traversal !== 'guided' || store.get(panZoomModeAtom);
      const wantsZoom = isPanZoom ? !e.shiftKey : e.shiftKey;

      if (wantsZoom) {
        e.preventDefault();

        // Zoom about cursor: keep the point under the cursor fixed on screen
        const rect = container.getBoundingClientRect();
        const canvasH = rect.height;
        const aspect = rect.width / canvasH || 1;
        const ofsY = overlayOffsetRef.current;
        const iz = canvasH > 0 ? 1 - ofsY / canvasH : 1;
        const iy = canvasH > 0 ? -ofsY / canvasH : 0;

        const oldZoom = store.get(cameraZoomAtom);
        const oldPanX = store.get(cameraPanXAtom);
        const oldPanY = store.get(cameraPanYAtom);

        // Cursor → NDC → projection space
        const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const ndcY = -(((e.clientY - rect.top) / canvasH) * 2 - 1);
        const zoomIz = oldZoom * iz;
        const cursorProjX = (ndcX * aspect) / zoomIz - oldPanX;
        const cursorProjY = (ndcY - iy) / zoomIz - oldPanY;

        const factor = 1 - (e.deltaX || e.deltaY) * 0.002;
        const newZoom = Math.min(4, Math.max(0.25, oldZoom * factor));

        // Adjust pan so cursor stays at same screen position
        const ratio = newZoom / oldZoom;
        const newPanX = (cursorProjX + oldPanX) / ratio - cursorProjX;
        const newPanY = (cursorProjY + oldPanY) / ratio - cursorProjY;

        store.set(cameraZoomAtom, newZoom);
        store.set(cameraPanXAtom, newPanX);
        store.set(cameraPanYAtom, newPanY);
        return;
      }

      // Tour scrub — only in guided mode
      if (store.get(tourTraversalAtom) !== 'guided') return;
      e.preventDefault();
      // Cancel any running position animation before wheel scrub
      store.set(animationGenAtom, (g) => g + 1);
      store.set(tourPlayingAtom, false);
      const wasSuspended = store.get(guidedSuspendedAtom);
      // Update position + slider imperatively (always, even during transition).
      // In equal mode, scrub in visual space for perceptual uniformity.
      const al = arcLengthsRef.current;
      const mode = spacingModeRef.current;
      if (mode === 'equal' && al) {
        const curVisual = tourToVisual(positionRef.current, al);
        let nextVisual = curVisual + e.deltaY * 0.002;
        nextVisual = nextVisual - Math.floor(nextVisual);
        const nextTour = visualToTour(nextVisual, al);
        positionRef.current = nextTour;
        sliderRef.current?.setPosition(nextVisual);
        updateAxesRef.current(nextTour);
      } else {
        let next = positionRef.current + e.deltaY * 0.002;
        next = next - Math.floor(next);
        positionRef.current = next;
        sliderRef.current?.setPosition(next);
        updateAxesRef.current(next);
      }
      if (wasSuspended && !isTransitioningRef.current()) {
        resumeWithTransitionRef.current(150, () => positionRef.current);
      } else if (!wasSuspended && !isTransitioningRef.current()) {
        scatterRef.current?.setTourPosition(positionRef.current);
      }
      scheduleFlushRef.current();
    };
    container.addEventListener('wheel', handler, { passive: false });
    return () => container.removeEventListener('wheel', handler);
  }, [store]);

  // ─── 3D camera rotation (manual mode only) ──────────────────────────────
  const setIs3dRotated = useSetAtom(is3dRotatedAtom);
  const is3dRotated = useAtomValue(is3dRotatedAtom);
  const quatRef = useRef<Quat>(IDENTITY_QUAT);
  const is3dEnabledRef = useRef(false);
  const revertAnimRef = useRef<number | null>(null);
  const residualPCRef = useRef<Float32Array | null>(null);
  const backendRef = useRef(resolvedBackend);
  backendRef.current = resolvedBackend;
  const effectiveToolbarHeightRef = useRef(effectiveToolbarHeight);
  effectiveToolbarHeightRef.current = effectiveToolbarHeight;

  // Shift+drag arcball rotation
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let dragging = false;
    let lastSphere: [number, number, number] | null = null;

    const toNdc = (e: PointerEvent): [number, number] => {
      const rect = container.getBoundingClientRect();
      // Map relative to the visible content area (below toolbar)
      const t = effectiveToolbarHeightRef.current;
      const visibleH = rect.height - t;
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -(((e.clientY - rect.top - t) / visibleH) * 2 - 1);
      return [x, y];
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      // Shift+drag to enter 3D; once active, plain drag also rotates
      if (!e.shiftKey && !store.get(is3dRotatedAtom)) return;
      if (store.get(tourTraversalAtom) !== 'manual') return;
      if (!store.get(metadataAtom)) return; // data not loaded yet
      if (backendRef.current !== 'webgpu') return; // 3D requires WebGPU
      // Let clicks on buttons (revert, toolbar, etc.) pass through
      if ((e.target as Element).closest('button, a')) return;
      e.preventDefault();
      e.stopPropagation();
      dragging = true;
      container.setPointerCapture(e.pointerId);
      const [nx, ny] = toNdc(e);
      lastSphere = projectToSphere(nx, ny);

      // Enable 3D on first rotation
      if (!is3dEnabledRef.current) {
        scatterRef.current?.enable3d();
        is3dEnabledRef.current = true;
      }
    };

    const onMove = (e: PointerEvent) => {
      if (!dragging || !lastSphere) return;
      e.preventDefault();
      const [nx, ny] = toNdc(e);
      const curSphere = projectToSphere(nx, ny);
      const delta = arcballQuat(lastSphere, curSphere);
      quatRef.current = multiplyQuat(delta, quatRef.current);
      lastSphere = curSphere;

      const mat = quatToMat3(quatRef.current);
      scatterRef.current?.set3dRotation(mat);

      if (residualPCRef.current) {
        axisOverlayRef.current?.setRotation3d(residualPCRef.current, mat);
      }

      if (!store.get(is3dRotatedAtom)) {
        store.set(is3dRotatedAtom, true);
      }
    };

    const endDrag = () => {
      dragging = false;
      lastSphere = null;
    };

    const onUp = (e: PointerEvent) => {
      if (!dragging) return;
      endDrag();
      container.releasePointerCapture(e.pointerId);
    };

    container.addEventListener('pointerdown', onDown);
    container.addEventListener('pointermove', onMove);
    container.addEventListener('pointerup', onUp);
    container.addEventListener('pointercancel', endDrag);
    container.addEventListener('lostpointercapture', endDrag);
    return () => {
      container.removeEventListener('pointerdown', onDown);
      container.removeEventListener('pointermove', onMove);
      container.removeEventListener('pointerup', onUp);
      container.removeEventListener('pointercancel', endDrag);
      container.removeEventListener('lostpointercapture', endDrag);
    };
  }, [store]);

  // Slerp revert animation
  const revertCamera = useCallback(() => {
    if (revertAnimRef.current !== null) cancelAnimationFrame(revertAnimRef.current);
    const startQuat: Quat = [...quatRef.current];
    if (isIdentityQuat(startQuat)) {
      // Already at identity — just disable
      scatterRef.current?.disable3d();
      is3dEnabledRef.current = false;
      quatRef.current = IDENTITY_QUAT;
      residualPCRef.current = null;
      axisOverlayRef.current?.clearRotation3d();
      setIs3dRotated(false);
      return;
    }
    const duration = 300;
    const startTime = performance.now();
    const tick = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // Ease-out cubic
      const eased = 1 - (1 - progress) ** 3;
      const q = slerp(startQuat, IDENTITY_QUAT, eased);
      quatRef.current = q;
      const mat = quatToMat3(q);
      scatterRef.current?.set3dRotation(mat);
      if (residualPCRef.current) {
        axisOverlayRef.current?.setRotation3d(residualPCRef.current, mat);
      }
      if (progress < 1) {
        revertAnimRef.current = requestAnimationFrame(tick);
      } else {
        revertAnimRef.current = null;
        quatRef.current = IDENTITY_QUAT;
        scatterRef.current?.disable3d();
        is3dEnabledRef.current = false;
        residualPCRef.current = null;
        axisOverlayRef.current?.clearRotation3d();
        setIs3dRotated(false);
      }
    };
    revertAnimRef.current = requestAnimationFrame(tick);
  }, [setIs3dRotated]);

  // Escape key to revert
  useEffect(() => {
    if (!is3dRotated) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        revertCamera();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [is3dRotated, revertCamera]);

  // Reset 3D state when leaving manual mode
  useEffect(() => {
    if (tourTraversal === 'manual') return;
    if (is3dEnabledRef.current) {
      scatterRef.current?.disable3d();
      is3dEnabledRef.current = false;
      quatRef.current = IDENTITY_QUAT;
      residualPCRef.current = null;
      axisOverlayRef.current?.clearRotation3d();
      setIs3dRotated(false);
    }
  }, [tourTraversal, setIs3dRotated]);

  const tickCount = views?.length ?? embeddedViews?.length ?? previewCount;
  const hasData = !!data && !!metadata;

  if (import.meta.env.DEV && views && views.length !== previewCount) {
    console.warn(
      `[dtour] views.length (${views.length}) differs from previewCount (${previewCount}). Selector ticks and radial bars reflect views count; preview gallery reflects previewCount. Set previewCount to match views.length for full alignment.`,
    );
  }

  const overlayHeight = containerSize.height - overlayOffsetY;
  // Compute camera inset values for hover coordinate transform
  const cameraInsetOffsetY = containerSize.height > 0 ? -overlayOffsetY / containerSize.height : 0;
  const cameraInsetZoom = containerSize.height > 0 ? 1 - overlayOffsetY / containerSize.height : 1;

  return (
    <div ref={containerRef} className="w-full h-full relative bg-dtour-bg">
      {/* Overlay wrapper — positioned below the toolbar so overlays
          are visually centered in the area below the toolbar. */}
      <div className="absolute left-0 right-0 bottom-0" style={{ top: `${overlayOffsetY}px` }}>
        {/* Preview gallery — only in guided mode */}
        {isGuidedMode && hasData && containerSize.width > 0 && previewCanvases.length > 0 && (
          <Gallery
            previewCanvases={previewCanvases}
            containerWidth={containerSize.width}
            containerHeight={overlayHeight}
            toolbarHeight={0}
            onResumeGuided={resumeWithTransition}
          />
        )}

        {/* Lasso selection overlay — available in all modes (disabled during 3D rotation) */}
        {hasData && containerSize.width > 0 && !is3dRotated && (
          <LassoOverlay
            scatter={scatter}
            width={containerSize.width}
            height={overlayHeight}
            offsetY={overlayOffsetY}
            spatialIndex={spatialIndexRef}
            insetOffsetY={cameraInsetOffsetY}
            insetZoom={cameraInsetZoom}
          />
        )}

        {/* Axis overlay — interactive in manual mode (disabled during 3D rotation),
            read-only in guided when enabled */}
        {(tourTraversal === 'manual' || (isGuidedMode && showAxes)) &&
          hasData &&
          containerSize.width > 0 && (
            <AxisOverlay
              ref={axisOverlayRef}
              scatter={scatter}
              width={containerSize.width}
              height={overlayHeight}
              readOnly={isGuidedMode || is3dRotated}
            />
          )}

        {/* Revert camera button — shown when 3D camera is rotated in manual mode */}
        {tourTraversal === 'manual' && <RevertCameraButton onRevert={revertCamera} />}

        {/* Circular selector + radial chart overlay — only in guided mode, above lasso */}
        {isGuidedMode && hasData && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            {/* Radial chart — behind selector */}
            {coloredTracks.length > 0 && (
              <div className="absolute">
                <RadialChart
                  tracks={coloredTracks}
                  keyframeCount={tickCount}
                  size={selectorSize}
                  innerRadius={selectorSize * 0.4}
                  arcLengths={arcLengths}
                  spacingMode={spacingMode}
                />
              </div>
            )}
            {/* Selector — on top for drag interaction */}
            <div className="pointer-events-none relative z-10">
              <CircularSlider
                ref={sliderRef}
                value={
                  spacingMode === 'equal' && arcLengths
                    ? tourToVisual(position, arcLengths)
                    : position
                }
                onChange={handlePositionChange}
                onSeek={handlePositionSeek}
                onDragStart={handleDragStart}
                tickCount={tickCount}
                size={selectorSize}
                arcLengths={arcLengths}
                spacingMode={spacingMode}
                currentKeyframe={currentKeyframe}
                hoveredKeyframe={hoveredKeyframe}
                previewCenters={previewCenters}
                showOriginDot={showAxes}
                visibility={sliderVisibility}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
