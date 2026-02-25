import { computeArcLengths, createScatter, interpolateAtPosition } from '@dtour/scatter';
import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AxisOverlay } from './components/AxisOverlay.tsx';
import { CircularSlider } from './components/CircularSlider.tsx';
import { Gallery } from './components/Gallery.tsx';
import { LassoOverlay } from './components/LassoOverlay.tsx';
import { useGrandTour } from './hooks/useGrandTour.ts';
import { useScatter } from './hooks/useScatter.ts';
import { computeGalleryPositions } from './layout/gallery-positions.ts';
import { RadialChart } from './radial-chart/RadialChart.tsx';
import { parseMetrics } from './radial-chart/parse-metrics.ts';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import {
  canvasSizeAtom,
  currentBasisAtom,
  guidedSuspendedAtom,
  metadataAtom,
  previewCountAtom,
  tourPositionAtom,
  viewModeAtom,
} from './state/atoms.ts';
import { createDefaultViews } from './views.ts';

export type DtourViewerProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer | undefined;
  /** Tour keyframe views (p×2 column-major). Auto-generated if omitted. */
  views?: Float32Array[] | undefined;
  /** Arrow IPC ArrayBuffer with per-view quality metrics. */
  metrics?: ArrayBuffer | undefined;
  /** Track configuration for radial bar charts. */
  metricTracks?: RadialTrackConfig[] | undefined;
  /** Called on every status event from the renderer. */
  onStatus?: ((status: ScatterStatus) => void) | undefined;
  /** Height in px of an overlay toolbar above the canvas. The shader shifts
   *  and scales content to center it in the visible area below the toolbar.
   *  Animates smoothly to 0 in zen mode. Default 0. */
  toolbarHeight?: number | undefined;
};

const MIN_SELECTOR_SIZE = 80;
const PREVIEW_PHYSICAL_SIZE = 200; // Physical pixels per preview canvas

const INSET_ANIMATION_MS = 300;

export const DtourViewer = ({
  data,
  views,
  metrics,
  metricTracks,
  onStatus,
  toolbarHeight = 0,
}: DtourViewerProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const scatterRef = useRef<ScatterInstance | null>(null);
  const previewCanvasesRef = useRef<HTMLCanvasElement[]>([]);
  const [position, setPosition] = useAtom(tourPositionAtom);
  const metadata = useAtomValue(metadataAtom);
  const setMetadata = useSetAtom(metadataAtom);
  const previewCount = useAtomValue(previewCountAtom);
  const viewMode = useAtomValue(viewModeAtom);
  const setGuidedSuspended = useSetAtom(guidedSuspendedAtom);
  const setCanvasSize = useSetAtom(canvasSizeAtom);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const lastDataRef = useRef<ArrayBuffer | undefined>(undefined);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  const setCurrentBasis = useSetAtom(currentBasisAtom);

  const isGuidedMode = viewMode === 'guided';

  // Resolve views (from props or auto-generated) and precompute arc lengths
  // so we can track the current tour basis on the main thread.
  const { resolvedViews, arcLengths } = useMemo(() => {
    if (!metadata || metadata.dimCount < 2) return { resolvedViews: null, arcLengths: null };
    const dims = metadata.dimCount;
    const rb =
      views && views.length > 0
        ? views.map((b) => new Float32Array(b))
        : createDefaultViews(dims, previewCount);
    return { resolvedViews: rb, arcLengths: computeArcLengths(rb, dims) };
  }, [views, metadata, previewCount]);

  // Keep currentBasisAtom in sync with the tour interpolation so other
  // modes (manual, zen) can initialize from the current projection.
  // Only update in guided mode — in manual/grand the atom is owned by
  // AxisOverlay / useGrandTour respectively.
  useEffect(() => {
    if (viewMode !== 'guided') return;
    if (!resolvedViews || !arcLengths || !metadata) return;
    const dims = metadata.dimCount;
    const out = new Float32Array(dims * 2);
    interpolateAtPosition(out, resolvedViews, arcLengths, dims, position);
    setCurrentBasis(out);
  }, [viewMode, resolvedViews, arcLengths, metadata, position, setCurrentBasis]);

  // Parse metrics Arrow IPC into renderable tracks
  const parsedTracks = useMemo(
    () => (metrics ? parseMetrics(metrics, metricTracks) : []),
    [metrics, metricTracks],
  );

  // Bridge Jotai atoms (style, camera) → scatter instance
  useScatter(scatterRef.current);

  // Animate camera inset when the toolbar appears/disappears (grand toggle).
  // The shader shifts + scales content to center it below the toolbar.
  // We also track the current pixel offset for positioning overlays.
  const [overlayOffsetY, setOverlayOffsetY] = useState(
    toolbarHeight > 0 && viewMode !== 'grand' ? toolbarHeight / 2 : 0,
  );
  const overlayOffsetRef = useRef(overlayOffsetY);
  overlayOffsetRef.current = overlayOffsetY;
  const insetAnimRef = useRef<number | null>(null);

  useEffect(() => {
    const scatter = scatterRef.current;
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
  }, [viewMode, toolbarHeight, containerSize.height]);

  // Grand mode: Givens-rotation grand tour
  useGrandTour(scatterRef.current, viewMode, metadata);

  // Compute gallery positions + padding for canvas insets
  const { previewSize, padX, padY } = useMemo(
    () => computeGalleryPositions(containerSize.width, containerSize.height, previewCount),
    [containerSize.width, containerSize.height, previewCount],
  );

  // Selector radius: min(canvas area) - X/2 where X = previewSize/2
  const canvasWidth = containerSize.width - 2 * padX;
  const canvasHeight = containerSize.height - 2 * padY;
  const selectorSize = Math.max(
    MIN_SELECTOR_SIZE,
    Math.min(canvasWidth, canvasHeight) - previewSize / 2,
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

    const scatter = createScatter({ canvases: [mainCanvas, ...previews] });
    scatterRef.current = scatter;

    scatter.subscribe((s: ScatterStatus) => {
      onStatusRef.current?.(s);
      if (s.type === 'metadata') {
        setMetadata(s.metadata);
      }
    });

    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const curDpr = window.devicePixelRatio || 1;
      scatter.resize(0, Math.round(width * curDpr), Math.round(height * curDpr));
      setContainerSize({ width, height });
      setCanvasSize({ width, height });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      scatter.destroy();
      scatterRef.current = null;
      mainCanvas.remove();
      for (const c of previews) c.remove();
      previewCanvasesRef.current = [];
    };
  }, [previewCount, setMetadata, setCanvasSize]);

  // Send data when it changes
  useEffect(() => {
    if (!data || !scatterRef.current || data === lastDataRef.current) return;
    lastDataRef.current = data;
    scatterRef.current.loadData(data.slice(0));
  }, [data]);

  // Set views when available (from props or auto-generated from metadata)
  useEffect(() => {
    const scatter = scatterRef.current;
    if (!scatter) return;
    if (views && views.length > 0) {
      scatter.setBases(views.map((b) => new Float32Array(b)));
    } else if (metadata && metadata.dimCount >= 2) {
      const defaultViews = createDefaultViews(metadata.dimCount, previewCount);
      scatter.setBases(defaultViews);
    }
    // Safety: explicitly request a full re-render after views are set,
    // ensuring all preview canvases get painted even if messages race.
    scatter.render();
  }, [views, metadata, previewCount]);

  const handlePositionChange = useCallback(
    (pos: number) => {
      setGuidedSuspended(false);
      setPosition(pos);
    },
    [setPosition, setGuidedSuspended],
  );

  const tickCount = views?.length ?? previewCount;
  const hasData = !!data;

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
            />
          )}

        {/* Lasso selection overlay — available in all modes, below circular selector */}
        {hasData && containerSize.width > 0 && (
          <LassoOverlay
            scatter={scatterRef.current}
            width={containerSize.width}
            height={containerSize.height}
          />
        )}

        {/* Manual mode axis overlay — rendered after lasso so handles are on top */}
        {viewMode === 'manual' && hasData && containerSize.width > 0 && (
          <AxisOverlay
            scatter={scatterRef.current}
            width={containerSize.width}
            height={containerSize.height}
          />
        )}

        {/* Circular selector + radial chart overlay — only in guided mode, above lasso */}
        {isGuidedMode && hasData && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
            {/* Radial chart — behind selector */}
            {parsedTracks.length > 0 && (
              <div className="absolute">
                <RadialChart
                  tracks={parsedTracks}
                  keyframeCount={tickCount}
                  position={position}
                  size={selectorSize}
                  innerRadius={selectorSize * 0.4}
                />
              </div>
            )}
            {/* Selector — on top for drag interaction */}
            <div className="pointer-events-none">
              <CircularSlider
                value={position}
                onChange={handlePositionChange}
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
