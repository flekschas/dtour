import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { bitPackIndices } from '@dtour/scatter';
import { QuestionMarkIcon } from '@phosphor-icons/react';
import { createStore, Provider, useAtomValue, useSetAtom, useStore } from 'jotai';
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ColorLegend } from './components/ColorLegend.tsx';
import { DtourToolbar } from './components/DtourToolbar.tsx';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './components/ui/tooltip.tsx';
import { DtourViewer } from './DtourViewer.tsx';
import { useIsTruncated } from './hooks/useIsTruncated.ts';
import { useModeCycling } from './hooks/useModeCycling.ts';
import { useSystemTheme } from './hooks/useSystemTheme.ts';
import { PortalContainerContext, usePortalContainer } from './portal-container.tsx';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import type { DtourSpec, KeyframeLoading } from './spec.ts';
import {
  activeColumnsAtom,
  backgroundColorAtom,
  betweenKeyframesAtom,
  colorMapAtom,
  embeddedConfigAtom,
  keyframeDescriptionsAtom,
  keyframeLoadingsAtom,
  legendSelectionAtom,
  legendVisibleAtom,
  metadataAtom,
  pointColorByAtom,
  resolvedThemeAtom,
  showTourDescriptionAtom,
  tourByAtom,
  tourDescriptionAtom,
  tourFamilyAtom,
  tourTraversalAtom,
} from './state/atoms.ts';
import { applySpecToStore, initStoreFromSpec, useSpecSync } from './state/spec-sync.ts';

export type DtourHandle = {
  /** Select points by index array or bit-packed mask. */
  select: (
    indicesOrMask: number[] | Int32Array | Uint32Array,
    opts?: { isBitPacked?: boolean },
  ) => void;
  /** Select by categorical label names. Resolves labels against the active color column. */
  selectByLabels: (labels: string[]) => void;
  /** Clear the current selection. */
  clearSelection: () => void;
};

export type DtourProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer;
  /** Tour keyframe bases (p×2 column-major). Auto-generated if omitted. */
  views?: Float32Array[];
  /** Arrow IPC ArrayBuffer with per-view quality metrics (columns = metrics, rows = views). */
  metrics?: ArrayBuffer;
  /** Track configuration for radial bar charts. When omitted, all metrics are shown with defaults. */
  metricTracks?: RadialTrackConfig[];
  /** Global bar width override for radial charts ('full' or px). */
  metricBarWidth?: 'full' | number;
  /** Partial spec controlling component state. Omitted fields use defaults. */
  spec?: DtourSpec;
  /** Fires when internal state changes (debounced ~250ms). Full resolved spec. */
  onSpecChange?: (spec: Required<DtourSpec>) => void;
  /** Called on every status event from the renderer. */
  onStatus?: (status: ScatterStatus) => void;
  /** Hide the toolbar. Default false. */
  hideToolbar?: boolean;
  /** Called when the user requests loading new data via the toolbar file picker. */
  onLoadData?: (data: ArrayBuffer, fileName: string) => void;
  /** Called when the user clicks the toolbar logo. */
  onLogoClick?: () => void;
  /** Fires when legend selection changes for a categorical color column. Reports selected label names or empty array when cleared. */
  onSelectionChange?: (labels: string[]) => void;
  /** Fires when lasso selection completes. Reports the bit-packed selection mask (1 bit per point, Uint32Array). */
  onPointSelectionChange?: (mask: Uint32Array) => void;
  /** Per-label color map. Values are hex strings or theme-aware {light, dark} objects. */
  colorMap?: Record<string, string | { light: string; dark: string }>;
  /** Element to portal Radix popups into (for Shadow DOM isolation). When omitted, portals render into document.body as usual. */
  portalContainer?: HTMLElement;
  /** Called when the viewer is ready with an API handle for programmatic control. */
  onReady?: (api: DtourHandle) => void;
  /** Rendering backend. Default 'auto' — probes for WebGPU (incl. the
   *  float32-blendable feature) and falls back to the WebGL2 backend otherwise. */
  backend?: 'webgpu' | 'webgl' | 'auto';
  /** Tour family: hyperdimensional (one high-D space) or sequential (multiple 2D embeddings). */
  tourFamily?: 'hyperdimensional' | 'sequential';
  /** Human-readable tour description shown in the description sub-bar. */
  tourDescription?: string | null;
  /** Per-keyframe descriptions: string[] of literals, or a template string
   *  with {primary}, {secondary}, {relation} placeholders. */
  keyframeDescriptions?: string | string[] | null;
  /** Per-keyframe feature loadings. Overrides embedded config. */
  keyframeLoadings?: KeyframeLoading[] | null;
};

// Minimal inline markdown: [text](url), **bold**/__bold__, *italic*/_italic_
const MD_INLINE = /\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|__(.+?)__|\*(.+?)\*|_(.+?)_/g;

/** Caveat surfaced via the golden question-mark icon while a sequential tour is between keyframes. */
const SEQUENTIAL_TOUR_CAVEAT =
  "Between keyframes, dtour interpolates only to help you follow points across embeddings. These intermediate frames are not meaningful embeddings—don't read structure into them.";

function InlineMarkdown({ text }: { text: string }) {
  const parts: ReactNode[] = [];
  let last = 0;
  let key = 0;
  for (const m of text.matchAll(MD_INLINE)) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    if (m[1] != null) {
      parts.push(
        <a key={key++} href={m[2]} target="_blank" rel="noopener noreferrer" className="underline">
          {m[1]}
        </a>,
      );
    } else if (m[3] != null || m[4] != null) {
      parts.push(<strong key={key++}>{m[3] ?? m[4]}</strong>);
    } else if (m[5] != null || m[6] != null) {
      parts.push(<em key={key++}>{m[5] ?? m[6]}</em>);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <>{parts}</>;
}

export const Dtour = ({
  data,
  views,
  metrics,
  metricTracks,
  metricBarWidth,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar = false,
  onLoadData,
  onLogoClick,
  onSelectionChange,
  onPointSelectionChange,
  colorMap,
  portalContainer,
  onReady,
  backend,
  tourFamily,
  tourDescription,
  keyframeDescriptions,
  keyframeLoadings,
}: DtourProps) => {
  // Each Dtour instance gets its own isolated jotai store.
  // Eagerly apply initial spec values so there's no flash of defaults.
  // biome-ignore lint/correctness/useExhaustiveDependencies: store created once on mount
  const store = useMemo(() => {
    const s = createStore();
    initStoreFromSpec(s, spec);
    return s;
  }, []);

  return (
    <PortalContainerContext.Provider value={portalContainer}>
      <Provider store={store}>
        <DtourInner
          data={data}
          views={views}
          metrics={metrics}
          metricTracks={metricTracks}
          metricBarWidth={metricBarWidth}
          spec={spec}
          onSpecChange={onSpecChange}
          onStatus={onStatus}
          hideToolbar={hideToolbar}
          onLoadData={onLoadData}
          onLogoClick={onLogoClick}
          onSelectionChange={onSelectionChange}
          onPointSelectionChange={onPointSelectionChange}
          colorMap={colorMap}
          onReady={onReady}
          backend={backend}
          tourFamily={tourFamily}
          tourDescription={tourDescription}
          keyframeDescriptions={keyframeDescriptions}
          keyframeLoadings={keyframeLoadings}
        />
      </Provider>
    </PortalContainerContext.Provider>
  );
};

/** Inner component that lives inside the Provider so hooks bind to the store. */
const DtourInner = ({
  data,
  views,
  metrics,
  metricTracks,
  metricBarWidth,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar,
  onLoadData,
  onLogoClick,
  onSelectionChange,
  onPointSelectionChange,
  colorMap,
  onReady,
  backend,
  tourFamily: tourFamilyProp,
  tourDescription: tourDescriptionProp,
  keyframeDescriptions: keyframeDescriptionsProp,
  keyframeLoadings: keyframeLoadingsProp,
}: {
  data: ArrayBuffer | undefined;
  views: Float32Array[] | undefined;
  metrics: ArrayBuffer | undefined;
  metricTracks: RadialTrackConfig[] | undefined;
  metricBarWidth: 'full' | number | undefined;
  spec: DtourSpec | undefined;
  onSpecChange: ((spec: Required<DtourSpec>) => void) | undefined;
  onStatus: ((status: ScatterStatus) => void) | undefined;
  hideToolbar: boolean;
  onLoadData: ((data: ArrayBuffer, fileName: string) => void) | undefined;
  onLogoClick: (() => void) | undefined;
  onSelectionChange: ((labels: string[]) => void) | undefined;
  onPointSelectionChange: ((mask: Uint32Array) => void) | undefined;
  colorMap: Record<string, string | { light: string; dark: string }> | undefined;
  onReady: ((api: DtourHandle) => void) | undefined;
  backend: 'webgpu' | 'webgl' | 'auto' | undefined;
  tourFamily: 'hyperdimensional' | 'sequential' | undefined;
  tourDescription: string | null | undefined;
  keyframeDescriptions: string | string[] | null | undefined;
  keyframeLoadings: KeyframeLoading[] | null | undefined;
}) => {
  useSpecSync(spec, onSpecChange);
  useModeCycling();
  useSystemTheme();

  // ── Apply theme class to portal container ──────────────────────────────
  const resolvedTheme = useAtomValue(resolvedThemeAtom);
  const portalContainer = usePortalContainer();
  useEffect(() => {
    const target = portalContainer ?? document.body;
    if (resolvedTheme === 'light') {
      target.classList.add('dtour-light');
    } else {
      target.classList.remove('dtour-light');
    }
    return () => {
      if (!portalContainer) {
        document.body.classList.remove('dtour-light');
      }
    };
  }, [portalContainer, resolvedTheme]);

  // ── Apply embedded config from Parquet metadata ──────────────────────
  const embeddedConfig = useAtomValue(embeddedConfigAtom);
  const store = useStore();
  const embeddedAppliedRef = useRef(false);

  // Reset when data changes so the next file's embedded config can apply
  // biome-ignore lint/correctness/useExhaustiveDependencies: data triggers reset
  useEffect(() => {
    embeddedAppliedRef.current = false;
  }, [data]);

  // Apply embedded spec fields that are NOT overridden by the prop spec
  useEffect(() => {
    if (!embeddedConfig || embeddedAppliedRef.current) return;
    embeddedAppliedRef.current = true;

    const fieldsToApply: DtourSpec = {};
    for (const [key, value] of Object.entries(embeddedConfig.spec)) {
      if (spec?.[key as keyof DtourSpec] == null) {
        (fieldsToApply as Record<string, unknown>)[key] = value;
      }
    }
    applySpecToStore(store, fieldsToApply);
  }, [embeddedConfig, spec, store]);

  // Sync colorMap prop → atom (embedded spec colorMap used as fallback)
  const setColorMap = useSetAtom(colorMapAtom);
  useEffect(() => {
    setColorMap(colorMap ?? embeddedConfig?.spec?.pointColorMap ?? null);
  }, [colorMap, embeddedConfig, setColorMap]);

  // Sync tour metadata: props take priority over embedded config
  const setKeyframeLoadings = useSetAtom(keyframeLoadingsAtom);
  const setKeyframeDescriptions = useSetAtom(keyframeDescriptionsAtom);
  const setTourFamily = useSetAtom(tourFamilyAtom);
  const setTourDescription = useSetAtom(tourDescriptionAtom);
  const setTourBy = useSetAtom(tourByAtom);
  useEffect(() => {
    setKeyframeLoadings(keyframeLoadingsProp ?? embeddedConfig?.tour?.keyframeLoadings ?? null);
    setKeyframeDescriptions(
      keyframeDescriptionsProp ?? embeddedConfig?.tour?.keyframeDescriptions ?? null,
    );
    setTourDescription(tourDescriptionProp ?? embeddedConfig?.tour?.description ?? null);

    // Resolve tourFamily and enforce tourBy consistency
    const resolvedKind = tourFamilyProp ?? embeddedConfig?.tour?.family ?? 'hyperdimensional';
    setTourFamily(resolvedKind);

    type TourBy = 'dimensions' | 'pca' | 'parameter';
    const specWillSetTourBy = embeddedConfig?.spec?.tourBy;
    if (resolvedKind === 'sequential') {
      setTourBy((prev: TourBy) => {
        if (prev !== 'parameter') {
          if (specWillSetTourBy !== 'parameter') {
            console.warn(
              `[dtour] tourFamily is 'sequential' but tourBy was '${prev}'; forcing tourBy to 'parameter'`,
            );
          }
          return 'parameter';
        }
        return prev;
      });
    } else {
      setTourBy((prev: TourBy) => {
        if (prev === 'parameter') {
          if (specWillSetTourBy === 'parameter') return prev; // spec will handle it
          console.warn(
            `[dtour] tourBy is 'parameter' but tourFamily is '${resolvedKind}'; falling back to 'dimensions'`,
          );
          return 'dimensions';
        }
        return prev;
      });
    }
  }, [
    embeddedConfig,
    keyframeLoadingsProp,
    keyframeDescriptionsProp,
    tourFamilyProp,
    tourDescriptionProp,
    setKeyframeLoadings,
    setKeyframeDescriptions,
    setTourFamily,
    setTourDescription,
    setTourBy,
  ]);

  // Sync resolved theme → background color + CSS class
  const setBackgroundColor = useSetAtom(backgroundColorAtom);
  useEffect(() => {
    setBackgroundColor(resolvedTheme === 'light' ? [1, 1, 1] : [0, 0, 0]);
  }, [resolvedTheme, setBackgroundColor]);

  // Forward legend selection changes to the parent as label name strings
  const legendSelection = useAtomValue(legendSelectionAtom);
  const pointColorBy = useAtomValue(pointColorByAtom);
  const metadata = useAtomValue(metadataAtom);

  // Apply tour.dimensions → activeColumnsAtom so the toolbar shows which
  // numeric columns participate in the predefined tour.
  const setActiveColumns = useSetAtom(activeColumnsAtom);
  useEffect(() => {
    if (!metadata) return;
    const tourDims = embeddedConfig?.tour?.dimensions;
    if (!tourDims || tourDims.length === 0) {
      setActiveColumns(null);
      return;
    }
    const indices = new Set<number>();
    for (const name of tourDims) {
      const idx = metadata.columnNames.indexOf(name);
      if (idx !== -1) indices.add(idx);
    }
    if (indices.size >= 2) {
      setActiveColumns(indices);
    }
  }, [metadata, embeddedConfig, setActiveColumns]);

  useEffect(() => {
    if (!onSelectionChange) return;

    if (!pointColorBy || !metadata) return;
    if (!metadata.categoricalColumnNames.includes(pointColorBy)) return;

    if (!legendSelection || legendSelection.size === 0) {
      onSelectionChange([]);
      return;
    }

    const allLabels = metadata.categoricalLabels[pointColorBy] ?? [];
    const selectedLabels = Array.from(legendSelection)
      .map((i) => allLabels[i])
      .filter((l): l is string => l !== undefined);

    onSelectionChange(selectedLabels.length > 0 ? selectedLabels : []);
  }, [legendSelection, pointColorBy, metadata, onSelectionChange]);

  // Track scatter instance for programmatic select API
  const [scatterInstance, setScatterInstance] = useState<ScatterInstance | null>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    if (!scatterInstance || !metadata) return;

    const handle: DtourHandle = {
      select: (indicesOrMask, opts) => {
        if (indicesOrMask.length === 0) {
          scatterInstance.clearSelection();
          return;
        }
        if (opts?.isBitPacked && indicesOrMask instanceof Uint32Array) {
          scatterInstance.setSelectionMask(new Uint32Array(indicesOrMask));
        } else {
          const packed = bitPackIndices(indicesOrMask, metadata.rowCount);
          scatterInstance.setSelectionMask(packed);
        }
      },
      selectByLabels: (labels) => {
        const colorByCol = store.get(pointColorByAtom);
        if (!colorByCol || !metadata.categoricalColumnNames.includes(colorByCol)) return;
        const allLabels = metadata.categoricalLabels[colorByCol] ?? [];
        const labelSet = new Set(labels);
        const indices = allLabels.map((l, i) => (labelSet.has(l) ? i : -1)).filter((i) => i >= 0);
        if (indices.length > 0) {
          scatterInstance.selectByColumn(colorByCol, { labelIndices: indices });
          store.set(legendSelectionAtom, new Set(indices));
        } else {
          scatterInstance.clearSelection();
          store.set(legendSelectionAtom, null);
        }
      },
      clearSelection: () => {
        scatterInstance.clearSelection();
      },
    };

    onReadyRef.current?.(handle);
  }, [scatterInstance, metadata, store]);

  // Intercept scatter status to extract point selection results
  const onPointSelectionRef = useRef(onPointSelectionChange);
  onPointSelectionRef.current = onPointSelectionChange;
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  const handleStatus = useCallback((status: ScatterStatus) => {
    if (status.type === 'selectionResult') {
      onPointSelectionRef.current?.(status.mask);
    }
    onStatusRef.current?.(status);
  }, []);

  const tourTraversal = useAtomValue(tourTraversalAtom);
  const isGrand = tourTraversal === 'grand';
  const legendVisible = useAtomValue(legendVisibleAtom);

  // Tour description sub-bar
  const showTourDescriptionPref = useAtomValue(showTourDescriptionAtom);
  const tourDescription = useAtomValue(tourDescriptionAtom);
  const tourFamily = useAtomValue(tourFamilyAtom);
  const isSequential = tourFamily === 'sequential';
  const betweenKeyframes = useAtomValue(betweenKeyframesAtom);
  const descriptionVisible =
    (showTourDescriptionPref ?? tourDescription !== null) &&
    tourTraversal === 'guided' &&
    tourDescription !== null;
  // Show a tooltip with the full text only when the description bar clips it.
  const [descriptionRef, descriptionTruncated] = useIsTruncated<HTMLSpanElement>(tourDescription);
  const effectiveToolbarHeight = hideToolbar ? 0 : descriptionVisible ? 72 : 40;

  // Sidebar width state — remembered across open/close cycles
  const [sidebarWidth, setSidebarWidth] = useState(200);
  const [dragging, setDragging] = useState(false);
  // Grand mode slides the legend out with the rest of the chrome; hovering the
  // right edge peeks it back in. Transient, so no persistent atom needed.
  const [legendPeek, setLegendPeek] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // In grand mode the legend collapses with the chrome unless the user is
  // peeking it via the right-edge hover strip. Resizing/peeking is disabled
  // in grand mode (the hover strip drives visibility instead).
  const legendInteractive = legendVisible && !isGrand;
  const legendShown = legendVisible && (!isGrand || legendPeek);
  const displayWidth = legendShown ? sidebarWidth : 0;

  // Below 720px the legend overlays the canvas instead of pushing the canvas +
  // toolbar narrower, so the toolbar (and its legend toggle) stay put. Measured
  // on the container so the threshold is stable whether or not the legend is open.
  const [containerWidth, setContainerWidth] = useState(0);
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      setContainerWidth(entries[0]?.contentRect.width ?? 0);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const legendOverlay = containerWidth > 0 && containerWidth < 720;

  // Reset any peek when leaving grand mode
  useEffect(() => {
    if (!isGrand) setLegendPeek(false);
  }, [isGrand]);

  // Drag-to-resize handler
  const onHandleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!legendInteractive) return;
      e.preventDefault();
      setDragging(true);

      const onMouseMove = (me: MouseEvent) => {
        const container = containerRef.current;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const maxWidth = rect.width * 0.4;
        const newWidth = Math.min(maxWidth, Math.max(64, rect.right - me.clientX));
        setSidebarWidth(newWidth);
      };

      const onMouseUp = () => {
        setDragging(false);
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    },
    [legendInteractive],
  );

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-full overflow-hidden flex ${resolvedTheme === 'light' ? 'dtour-light' : ''}`}
    >
      {/* Canvas panel — grows to fill remaining space */}
      <div className="relative flex-1 min-w-0">
        {/* Toolbar + optional description sub-bar */}
        <div
          className={`absolute inset-x-0 top-0 z-10 transition-[transform,opacity] duration-300 ease-out ${
            isGrand ? '-translate-y-full' : 'translate-y-0'
          } ${hideToolbar ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
        >
          <div className="h-10">
            <DtourToolbar onLoadData={onLoadData} onLogoClick={onLogoClick} />
          </div>
          {descriptionVisible && (
            <div className="h-8 flex items-center justify-center border-b border-dtour-surface bg-dtour-bg px-3">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span
                      ref={descriptionRef}
                      className={`max-w-full min-w-0 truncate text-[11px] text-dtour-text-muted ${
                        descriptionTruncated ? 'cursor-help' : ''
                      }`}
                    >
                      <strong>
                        {isSequential ? 'Sequential tour:' : 'Hyperdimensional tour:'}
                      </strong>{' '}
                      <InlineMarkdown text={tourDescription ?? ''} />
                    </span>
                  </TooltipTrigger>
                  {descriptionTruncated && (
                    <TooltipContent side="bottom" className="max-w-[360px]">
                      <InlineMarkdown text={tourDescription ?? ''} />
                    </TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>
            </div>
          )}
          {/* Sequential tours: a golden help icon straddles the bottom of the chrome
              (description bar if shown, otherwise the toolbar), fading in only while
              the tour is between keyframes. */}
          {isSequential && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label="About sequential tour transitions"
                    className={`absolute left-1/2 bottom-0 z-20 flex size-4 -translate-x-1/2 translate-y-1/2 items-center justify-center rounded-full bg-dtour-bg text-[#d4af37] ring-1 ring-[#d4af37] cursor-help transition-[opacity,transform] duration-200 ease-out ${
                      betweenKeyframes
                        ? 'opacity-100 scale-100'
                        : 'pointer-events-none opacity-0 scale-75'
                    }`}
                  >
                    <QuestionMarkIcon size={10} weight="bold" />
                  </button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[260px]">
                  {SEQUENTIAL_TOUR_CAVEAT}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <div className="absolute inset-0 overflow-hidden">
          <DtourViewer
            data={data}
            views={views}
            metrics={metrics}
            metricTracks={metricTracks}
            metricBarWidth={metricBarWidth}
            onStatus={handleStatus}
            toolbarHeight={effectiveToolbarHeight}
            onScatterReady={setScatterInstance}
            backend={backend}
          />
        </div>
      </div>
      {/* Drag handle — hidden in overlay mode (resize is a desktop affordance) */}
      {!legendOverlay && (
        <div
          className={`w-px shrink-0 transition-colors ${
            legendInteractive
              ? 'cursor-col-resize bg-dtour-surface hover:bg-dtour-text-muted active:bg-dtour-highlight'
              : 'pointer-events-none'
          }`}
          onMouseDown={onHandleMouseDown}
        />
      )}
      {/* Legend sidebar — flex column (pushes canvas) at ≥720px, absolute
          overlay over the canvas (below the toolbar) under 720px. */}
      <div
        className={
          legendOverlay
            ? 'absolute right-0 z-30 overflow-hidden border-l border-dtour-surface bg-dtour-bg'
            : 'shrink-0 overflow-hidden'
        }
        style={{
          width: displayWidth,
          transition: dragging ? 'none' : 'width 300ms cubic-bezier(.1,.1,0,1)',
          ...(legendOverlay ? { top: effectiveToolbarHeight, bottom: 0 } : {}),
        }}
        onMouseLeave={isGrand ? () => setLegendPeek(false) : undefined}
      >
        <div className="h-full" style={{ width: sidebarWidth }}>
          <ColorLegend />
        </div>
      </div>
      {/* Grand mode: hover the right edge to peek the collapsed legend back in */}
      {isGrand && legendVisible && !legendPeek && (
        <div
          className="absolute top-0 right-0 bottom-0 w-4 z-20"
          onMouseEnter={() => setLegendPeek(true)}
        />
      )}
    </div>
  );
};
