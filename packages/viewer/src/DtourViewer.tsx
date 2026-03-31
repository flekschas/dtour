import {
  GLASBEY_DARK,
  GLASBEY_LIGHT,
  OKABE_ITO,
  computeArcLengths,
  createScatter,
  createScatterWebGL,
  interpolateAtPosition,
} from '@dtour/scatter';
import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AxisOverlay } from './components/AxisOverlay.tsx';
import { CircularSlider } from './components/CircularSlider.tsx';
import type { CircularSliderHandle } from './components/CircularSlider.tsx';
import { Gallery } from './components/Gallery.tsx';
import { LassoOverlay } from './components/LassoOverlay.tsx';
import { useAnimatePosition } from './hooks/useAnimatePosition.ts';
import { useGrandTour } from './hooks/useGrandTour.ts';
import { usePlayback } from './hooks/usePlayback.ts';
import { useScatter } from './hooks/useScatter.ts';
import { computeSelectorSize } from './layout/selector-size.ts';
import { RadialChart } from './radial-chart/RadialChart.tsx';
import { parseMetrics } from './radial-chart/parse-metrics.ts';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import {
  activeColumnsAtom,
  activeIndicesAtom,
  animationGenAtom,
  cameraZoomAtom,
  canvasSizeAtom,
  currentBasisAtom,
  embeddedConfigAtom,
  guidedSuspendedAtom,
  legendSelectionAtom,
  metadataAtom,
  pointColorAtom,
  previewCountAtom,
  previewScaleAtom,
  resolvedThemeAtom,
  tourByAtom,
  tourPlayingAtom,
  tourPositionAtom,
  viewModeAtom,
} from './state/atoms.ts';
import { createDefaultViews, createPCAViews } from './views.ts';

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
  /** Rendering backend. Default 'webgpu'. */
  backend?: 'webgpu' | 'webgl' | undefined;
};

const PREVIEW_PHYSICAL_SIZE = 300; // Physical pixels per preview canvas

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
  backend = 'webgpu',
}: DtourViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const onScatterReadyRef = useRef(onScatterReady);
  onScatterReadyRef.current = onScatterReady;
  const [scatter, setScatter] = useState<ScatterInstance | null>(null);
  const scatterRef = useRef<ScatterInstance | null>(null);
  const previewCanvasesRef = useRef<HTMLCanvasElement[]>([]);
  const [position, setPosition] = useAtom(tourPositionAtom);
  const metadata = useAtomValue(metadataAtom);
  const embeddedConfig = useAtomValue(embeddedConfigAtom);
  const previewCount = useAtomValue(previewCountAtom);
  const previewScale = useAtomValue(previewScaleAtom);
  const viewMode = useAtomValue(viewModeAtom);
  const [guidedSuspended, setGuidedSuspended] = useAtom(guidedSuspendedAtom);
  const setPlaying = useSetAtom(tourPlayingAtom);
  const setCanvasSize = useSetAtom(canvasSizeAtom);
  const store = useStore();
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const activeIndices = useAtomValue(activeIndicesAtom);
  const setActiveColumns = useSetAtom(activeColumnsAtom);
  const lastDataRef = useRef<ArrayBuffer | undefined>(undefined);
  const prevDimCountRef = useRef<number | null>(null);
  const dataRef = useRef(data);
  dataRef.current = data;
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  const sliderRef = useRef<CircularSliderHandle>(null);
  const positionRef = useRef(position);
  const positionFlushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync positionRef with atom value (overwritten by direct drives within ~33ms)
  useEffect(() => {
    positionRef.current = position;
  }, [position]);

  // Debounced atom write — batches rapid position updates (~10fps atom writes)
  const schedulePositionFlush = useCallback(() => {
    if (positionFlushTimer.current !== null) {
      clearTimeout(positionFlushTimer.current);
    }
    positionFlushTimer.current = setTimeout(() => {
      positionFlushTimer.current = null;
      setPosition(positionRef.current);
    }, 100);
  }, [setPosition]);

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
      if (positionFlushTimer.current !== null) {
        clearTimeout(positionFlushTimer.current);
        positionFlushTimer.current = null;
      }
      setPosition(positionRef.current);
    }
    prevPlayingRef.current = playing;
  }, [playing, setPosition]);

  const setCurrentBasis = useSetAtom(currentBasisAtom);
  const tourBy = useAtomValue(tourByAtom);
  const [pcaResult, setPcaResult] = useState<{
    eigenvectors: Float32Array[];
    numDims: number;
  } | null>(null);

  const isGuidedMode = viewMode === 'guided';

  // Resolve views (from props or auto-generated) and precompute arc lengths
  // so we can track the current tour basis on the main thread.
  // Embedded tour views from Parquet metadata (only when nDims matches the dataset)
  const embeddedViews =
    embeddedConfig?.tour && metadata && embeddedConfig.tour.nDims === metadata.dimCount
      ? embeddedConfig.tour.views
      : null;

  const { resolvedViews, arcLengths } = useMemo(() => {
    if (!metadata || metadata.dimCount < 2) return { resolvedViews: null, arcLengths: null };
    if (activeIndices.length < 2) return { resolvedViews: null, arcLengths: null };
    const dims = metadata.dimCount;
    let rb: Float32Array[];
    if (tourBy === 'pca' && pcaResult && pcaResult.eigenvectors.length >= 2) {
      rb = createPCAViews(pcaResult.eigenvectors, dims, pcaResult.numDims, previewCount);
    } else if (views && views.length > 0) {
      rb = views.map((b) => new Float32Array(b));
    } else if (!views && embeddedViews) {
      rb = embeddedViews.map((b) => new Float32Array(b));
    } else {
      rb = createDefaultViews(dims, previewCount, activeIndices);
    }
    return { resolvedViews: rb, arcLengths: computeArcLengths(rb, dims) };
  }, [views, embeddedViews, metadata, previewCount, activeIndices, tourBy, pcaResult]);

  // Keep currentBasisAtom in sync with the tour interpolation so other
  // modes (manual, grand) can initialize from the current projection.
  // Only update in guided mode — in manual/grand the atom is owned by
  // AxisOverlay / useGrandTour respectively.
  // Skip when guidedSuspended: the GPU is still showing a directBasis
  // from the previous mode (grand/manual), not the tour interpolation,
  // so overwriting currentBasisAtom here would cause a jump on re-entry.
  useEffect(() => {
    if (viewMode !== 'guided') return;
    if (guidedSuspended) return;
    if (!resolvedViews || !arcLengths || !metadata) return;
    const dims = metadata.dimCount;
    const out = new Float32Array(dims * 2);
    interpolateAtPosition(out, resolvedViews, arcLengths, dims, position);
    setCurrentBasis(out);
  }, [viewMode, guidedSuspended, resolvedViews, arcLengths, metadata, position, setCurrentBasis]);

  // Parse metrics Arrow IPC into renderable tracks
  const parsedTracks = useMemo(() => {
    if (!metrics) return [];
    return parseMetrics(metrics, metricTracks, metricBarWidth);
  }, [metrics, metricTracks, metricBarWidth]);

  // Override confusion track color: highlight by default, label palette color on single selection
  const legendSelection = useAtomValue(legendSelectionAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const resolvedTheme = useAtomValue(resolvedThemeAtom);

  const coloredTracks = useMemo(() => {
    if (parsedTracks.length === 0) return parsedTracks;
    const confusionIdx = parsedTracks.findIndex((t) => t.label === 'confusion');
    if (confusionIdx === -1) return parsedTracks;

    let confusionColor = resolvedTheme === 'light' ? '#000000' : '#ffffff';

    if (
      legendSelection &&
      legendSelection.size === 1 &&
      typeof pointColor === 'string' &&
      metadata?.categoricalColumnNames.includes(pointColor)
    ) {
      const selectedIndex = legendSelection.values().next().value as number;
      const labels = metadata.categoricalLabels[pointColor] ?? [];
      const glasbey = resolvedTheme === 'light' ? GLASBEY_LIGHT : GLASBEY_DARK;
      const colors =
        labels.length <= OKABE_ITO.length
          ? OKABE_ITO
          : ([...OKABE_ITO, ...glasbey] as [number, number, number][]);
      const [r, g, b] = colors[selectedIndex % colors.length]!;
      confusionColor = `rgb(${r},${g},${b})`;
    }

    const currentTrack = parsedTracks[confusionIdx]!;
    if (currentTrack.color === confusionColor) return parsedTracks;

    const result = [...parsedTracks];
    result[confusionIdx] = { ...currentTrack, color: confusionColor };
    return result;
  }, [parsedTracks, legendSelection, pointColor, metadata, resolvedTheme]);

  // Bridge Jotai atoms (style, camera) → scatter instance
  useScatter(scatter);

  const isToolbarVisible = toolbarHeight > 0 && viewMode !== 'grand';

  // Animate camera inset when the toolbar appears/disappears (grand toggle).
  // The shader shifts + scales content to center it below the toolbar.
  // We also track the current pixel offset for positioning overlays.
  const [overlayOffsetY, setOverlayOffsetY] = useState(isToolbarVisible ? toolbarHeight / 2 : 0);
  const overlayOffsetRef = useRef(overlayOffsetY);
  overlayOffsetRef.current = overlayOffsetY;
  const insetAnimRef = useRef<number | null>(null);

  useEffect(() => {
    if (!scatter || containerSize.height === 0) return;

    const targetT = viewMode === 'grand' || toolbarHeight === 0 ? 0 : 1;
    const h = containerSize.height;
    const t = toolbarHeight;

    // Current inset factor: derive from current overlayOffsetY via ref
    const startT = t > 0 ? overlayOffsetRef.current / (t / 2) : 0;
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
      setOverlayOffsetY((currentT * t) / 2);

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
  }, [scatter, viewMode, toolbarHeight, containerSize.height]);

  // Grand mode: Givens-rotation grand tour
  useGrandTour(scatter, viewMode, metadata);

  // Largest selector diameter that doesn't overlap any gallery preview
  const selectorSize = useMemo(
    () =>
      computeSelectorSize(
        containerSize.width,
        containerSize.height,
        previewCount,
        isToolbarVisible,
        SELECTOR_PADDING,
        previewScale,
        coloredTracks.length,
      ),
    [
      containerSize.width,
      containerSize.height,
      previewCount,
      isToolbarVisible,
      previewScale,
      coloredTracks.length,
    ],
  );

  // Initialize scatter — create main + preview canvases imperatively
  // (transferControlToOffscreen can only be called once per canvas).
  useEffect(() => {
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

    // Preview canvases — one per tour keyframe view
    const previews: HTMLCanvasElement[] = [];
    for (let i = 0; i < previewCount; i++) {
      const c = document.createElement('canvas');
      c.width = PREVIEW_PHYSICAL_SIZE;
      c.height = PREVIEW_PHYSICAL_SIZE;
      c.style.width = '100%';
      c.style.height = '100%';
      c.style.display = 'block';
      c.style.borderRadius = '2px';
      previews.push(c);
    }
    previewCanvasesRef.current = previews;

    const factory = backend === 'webgl' ? createScatterWebGL : createScatter;
    const instance = factory({
      canvases: [mainCanvas, ...previews],
      zoom: store.get(cameraZoomAtom),
    });
    scatterRef.current = instance;
    setScatter(instance);
    onScatterReadyRef.current?.(instance);
    (globalThis as Record<string, unknown>).scatter = instance;

    instance.subscribe((s: ScatterStatus) => {
      onStatusRef.current?.(s);
      if (s.type === 'pcaResult') {
        setPcaResult({ eigenvectors: s.eigenvectors, numDims: s.numDims });
      }
      if (s.type === 'playbackTick') {
        positionRef.current = s.position;
        sliderRef.current?.setPosition(s.position);
        scheduleFlushRef.current();
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

    // Re-send data to the new scatter instance (e.g. after previewCount change
    // or HMR where the scatter is recreated but data hasn't changed).
    if (dataRef.current) {
      instance.loadData(dataRef.current.slice(0));
      lastDataRef.current = dataRef.current;
    } else {
      lastDataRef.current = undefined;
    }

    return () => {
      ro.disconnect();
      instance.destroy();
      scatterRef.current = null;
      setScatter(null);
      onScatterReadyRef.current?.(null);
      mainCanvas.remove();
      for (const c of previews) c.remove();
      previewCanvasesRef.current = [];
    };
  }, [previewCount, setCanvasSize, store, backend]);

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
    lastDataRef.current = data;
    scatter.loadData(data.slice(0));
  }, [data, scatter]);

  // Trigger PCA computation when tourBy is 'pca' and data is loaded
  useEffect(() => {
    if (tourBy !== 'pca' || !metadata || metadata.dimCount < 2 || !scatter) return;
    scatter.computePCA();
  }, [tourBy, metadata, scatter]);

  // Set views when available (from props, PCA, embedded, or auto-generated from metadata)
  useEffect(() => {
    if (!scatter) return;
    if (tourBy === 'pca' && pcaResult && pcaResult.eigenvectors.length >= 2 && metadata) {
      const pcaBases = createPCAViews(
        pcaResult.eigenvectors,
        metadata.dimCount,
        pcaResult.numDims,
        previewCount,
      );
      scatter.setBases(pcaBases);
    } else if (views && views.length > 0) {
      scatter.setBases(views.map((b) => new Float32Array(b)));
    } else if (!views && embeddedViews) {
      scatter.setBases(embeddedViews.map((b) => new Float32Array(b)));
    } else if (metadata && metadata.dimCount >= 2 && activeIndices.length >= 2) {
      const defaultViews = createDefaultViews(metadata.dimCount, previewCount, activeIndices);
      scatter.setBases(defaultViews);
    }
    // Safety: explicitly request a full re-render after views are set,
    // ensuring all preview canvases get painted even if messages race.
    scatter.render();
  }, [scatter, views, embeddedViews, metadata, previewCount, activeIndices, tourBy, pcaResult]);

  const { animateTo, cancelAnimation } = useAnimatePosition();

  // Slider click → animated seek to the clicked position
  const handlePositionSeek = useCallback(
    (pos: number) => {
      setGuidedSuspended(false);
      setPlaying(false);
      animateTo(pos);
    },
    [setGuidedSuspended, setPlaying, animateTo],
  );

  // Slider drag start → cancel animation, switch to immediate updates
  const handleDragStart = useCallback(() => {
    cancelAnimation();
    setGuidedSuspended(false);
  }, [cancelAnimation, setGuidedSuspended]);

  // Slider drag move → send directly to GPU, update slider imperatively,
  // debounce atom write to minimize React re-renders during drag.
  const handlePositionChange = useCallback(
    (pos: number) => {
      setGuidedSuspended(false);
      scatterRef.current?.setTourPosition(pos);
      sliderRef.current?.setPosition(pos);
      positionRef.current = pos;
      schedulePositionFlush();
    },
    [setGuidedSuspended, schedulePositionFlush],
  );

  // Wheel → scrub tour position (guided mode) or zoom (Shift+wheel, all modes).
  // Imperative listener with { passive: false } so preventDefault() works.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handler = (e: WheelEvent) => {
      if (e.shiftKey) {
        // Shift+wheel → zoom (camera distance)
        e.preventDefault();
        store.set(cameraZoomAtom, (prev) => {
          // deltaY > 0 = scroll down = zoom out (smaller zoom value)
          const factor = 1 - (e.deltaX || e.deltaY) * 0.002;
          // Clamp to distance range [1x, 4x] → zoom range [0.25, 1]
          return Math.min(1, Math.max(0.25, prev * factor));
        });
        return;
      }
      if (store.get(viewModeAtom) !== 'guided') return;
      e.preventDefault();
      // Cancel any running position animation before wheel scrub
      store.set(animationGenAtom, (g) => g + 1);
      store.set(tourPlayingAtom, false);
      store.set(guidedSuspendedAtom, false);
      // Send directly to GPU + slider, debounce atom write
      let next = positionRef.current + e.deltaY * 0.002;
      next = next - Math.floor(next);
      positionRef.current = next;
      scatterRef.current?.setTourPosition(next);
      sliderRef.current?.setPosition(next);
      scheduleFlushRef.current();
    };
    container.addEventListener('wheel', handler, { passive: false });
    return () => container.removeEventListener('wheel', handler);
  }, [store]);

  const tickCount = views?.length ?? embeddedViews?.length ?? previewCount;
  const hasData = !!data && !!metadata;

  if (import.meta.env.DEV && views && views.length !== previewCount) {
    console.warn(
      `[dtour] views.length (${views.length}) differs from previewCount (${previewCount}). Selector ticks and radial bars reflect views count; preview gallery reflects previewCount. Set previewCount to match views.length for full alignment.`,
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-dtour-bg">
      {/* Overlay wrapper — translateY matches the shader inset so overlays
          stay visually centered in the area below the toolbar. */}
      <div className="absolute inset-0" style={{ transform: `translateY(${overlayOffsetY}px)` }}>
        {/* Preview gallery — only in guided mode */}
        {isGuidedMode &&
          hasData &&
          containerSize.width > 0 &&
          previewCanvasesRef.current.length > 0 && (
            <Gallery
              previewCanvases={previewCanvasesRef.current}
              containerWidth={containerSize.width}
              containerHeight={containerSize.height}
              isToolbarVisible={isToolbarVisible}
            />
          )}

        {/* Lasso selection overlay — available in all modes, below circular selector */}
        {hasData && containerSize.width > 0 && (
          <LassoOverlay
            scatter={scatter}
            width={containerSize.width}
            height={containerSize.height}
          />
        )}

        {/* Manual mode axis overlay — rendered after lasso so handles are on top */}
        {viewMode === 'manual' && hasData && containerSize.width > 0 && (
          <AxisOverlay
            scatter={scatter}
            width={containerSize.width}
            height={containerSize.height}
          />
        )}

        {/* Circular selector + radial chart overlay — only in guided mode, above lasso */}
        {isGuidedMode && hasData && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            {/* Radial chart — behind selector */}
            {coloredTracks.length > 0 && (
              <div className="absolute">
                <RadialChart
                  tracks={coloredTracks}
                  keyframeCount={tickCount}
                  position={position}
                  size={selectorSize}
                  innerRadius={selectorSize * 0.4}
                />
              </div>
            )}
            {/* Selector — on top for drag interaction */}
            <div className="pointer-events-none relative z-10">
              <CircularSlider
                ref={sliderRef}
                value={position}
                onChange={handlePositionChange}
                onSeek={handlePositionSeek}
                onDragStart={handleDragStart}
                tickCount={tickCount}
                size={selectorSize}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
