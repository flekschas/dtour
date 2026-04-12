import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { bitPackIndices } from '@dtour/scatter';
import { Provider, createStore, useAtomValue, useSetAtom, useStore } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DtourViewer } from './DtourViewer.tsx';
import { ColorLegend } from './components/ColorLegend.tsx';
import { DtourToolbar } from './components/DtourToolbar.tsx';
import { useModeCycling } from './hooks/useModeCycling.ts';
import { useSystemTheme } from './hooks/useSystemTheme.ts';
import { PortalContainerContext } from './portal-container.tsx';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import type { DtourSpec, FrameLoading } from './spec.ts';
import {
  backgroundColorAtom,
  colorMapAtom,
  embeddedConfigAtom,
  frameLoadingsAtom,
  frameSummariesAtom,
  legendSelectionAtom,
  legendVisibleAtom,
  metadataAtom,
  pointColorAtom,
  resolvedThemeAtom,
  showTourDescriptionAtom,
  tourByAtom,
  tourDescriptionAtom,
  tourFrameDescriptionAtom,
  tourModeAtom,
  viewModeAtom,
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
  /** Tour keyframe views (p×2 column-major). Auto-generated if omitted. */
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
  /** Per-label color map. Values are hex strings or theme-aware {light, dark} objects. */
  colorMap?: Record<string, string | { light: string; dark: string }>;
  /** Element to portal Radix popups into (for Shadow DOM isolation). When omitted, portals render into document.body as usual. */
  portalContainer?: HTMLElement;
  /** Called when the viewer is ready with an API handle for programmatic control. */
  onReady?: (api: DtourHandle) => void;
  /** Rendering backend. Default 'webgpu'. */
  backend?: 'webgpu' | 'webgl';
  /** Tour mode identifier. Overrides embedded config when provided. */
  tourMode?: 'signed' | 'discriminative' | 'parameter' | null;
  /** Human-readable tour description shown in the description sub-bar. */
  tourDescription?: string | null;
  /** Per-frame tooltip template with {dim1}, {dim2}, {relation} placeholders. */
  tourFrameDescription?: string | null;
  /** Per-frame text summaries shown below preview thumbnails. */
  frameSummaries?: string[] | null;
  /** Per-frame top-2 feature correlations. Overrides embedded config. */
  frameLoadings?: FrameLoading[][] | null;
};

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
  colorMap,
  portalContainer,
  onReady,
  backend,
  tourMode,
  tourDescription,
  tourFrameDescription,
  frameSummaries,
  frameLoadings,
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
          colorMap={colorMap}
          onReady={onReady}
          backend={backend}
          tourMode={tourMode}
          tourDescription={tourDescription}
          tourFrameDescription={tourFrameDescription}
          frameSummaries={frameSummaries}
          frameLoadings={frameLoadings}
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
  colorMap,
  onReady,
  backend,
  tourMode: tourModeProp,
  tourDescription: tourDescriptionProp,
  tourFrameDescription: tourFrameDescriptionProp,
  frameSummaries: frameSummariesProp,
  frameLoadings: frameLoadingsProp,
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
  colorMap: Record<string, string | { light: string; dark: string }> | undefined;
  onReady: ((api: DtourHandle) => void) | undefined;
  backend: 'webgpu' | 'webgl' | undefined;
  tourMode: 'signed' | 'discriminative' | 'parameter' | null | undefined;
  tourDescription: string | null | undefined;
  tourFrameDescription: string | null | undefined;
  frameSummaries: string[] | null | undefined;
  frameLoadings: FrameLoading[][] | null | undefined;
}) => {
  useSpecSync(spec, onSpecChange);
  useModeCycling();
  useSystemTheme();

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
      if (spec?.[key as keyof DtourSpec] === undefined) {
        (fieldsToApply as Record<string, unknown>)[key] = value;
      }
    }
    applySpecToStore(store, fieldsToApply);
  }, [embeddedConfig, spec, store]);

  // Sync colorMap prop → atom (embedded colorMap used as fallback)
  const setColorMap = useSetAtom(colorMapAtom);
  useEffect(() => {
    setColorMap(colorMap ?? embeddedConfig?.colorMap ?? null);
  }, [colorMap, embeddedConfig, setColorMap]);

  // Sync tour metadata: props take priority over embedded config
  const setFrameLoadings = useSetAtom(frameLoadingsAtom);
  const setFrameSummaries = useSetAtom(frameSummariesAtom);
  const setTourMode = useSetAtom(tourModeAtom);
  const setTourDescription = useSetAtom(tourDescriptionAtom);
  const setTourFrameDescription = useSetAtom(tourFrameDescriptionAtom);
  const setTourBy = useSetAtom(tourByAtom);
  useEffect(() => {
    setFrameLoadings(frameLoadingsProp ?? embeddedConfig?.tour?.frameLoadings ?? null);
    setFrameSummaries(frameSummariesProp ?? embeddedConfig?.tour?.frameSummaries ?? null);
    setTourDescription(tourDescriptionProp ?? embeddedConfig?.tour?.tourDescription ?? null);
    setTourFrameDescription(
      tourFrameDescriptionProp ?? embeddedConfig?.tour?.tourFrameDescription ?? null,
    );

    // Resolve tourMode and enforce tourBy consistency
    const resolvedTourMode = tourModeProp ?? embeddedConfig?.tour?.tourMode ?? null;
    setTourMode(resolvedTourMode);

    type TourBy = 'dimensions' | 'pca' | 'parameter';
    if (resolvedTourMode === 'parameter') {
      setTourBy((prev: TourBy) => {
        if (prev !== 'parameter') {
          console.warn(
            `[dtour] tourMode is 'parameter' but tourBy was '${prev}'; forcing tourBy to 'parameter'`,
          );
          return 'parameter';
        }
        return prev;
      });
    } else {
      setTourBy((prev: TourBy) => {
        if (prev === 'parameter') {
          console.warn(
            `[dtour] tourBy is 'parameter' but tourMode is '${resolvedTourMode}'; falling back to 'dimensions'`,
          );
          return 'dimensions';
        }
        return prev;
      });
    }
  }, [
    embeddedConfig,
    frameLoadingsProp,
    frameSummariesProp,
    tourModeProp,
    tourDescriptionProp,
    tourFrameDescriptionProp,
    setFrameLoadings,
    setFrameSummaries,
    setTourMode,
    setTourDescription,
    setTourFrameDescription,
    setTourBy,
  ]);

  // Sync resolved theme → background color + CSS class
  const resolvedTheme = useAtomValue(resolvedThemeAtom);
  const setBackgroundColor = useSetAtom(backgroundColorAtom);
  useEffect(() => {
    // sRGB values matching --color-dtour-bg: dark=#000000, light=#ffffff
    setBackgroundColor(resolvedTheme === 'light' ? [1, 1, 1] : [0, 0, 0]);
  }, [resolvedTheme, setBackgroundColor]);

  // Forward legend selection changes to the parent as label name strings
  const legendSelection = useAtomValue(legendSelectionAtom);
  const pointColor = useAtomValue(pointColorAtom);
  const metadata = useAtomValue(metadataAtom);

  useEffect(() => {
    if (!onSelectionChange) return;

    if (typeof pointColor !== 'string' || !metadata) return;
    if (!metadata.categoricalColumnNames.includes(pointColor)) return;

    if (!legendSelection || legendSelection.size === 0) {
      onSelectionChange([]);
      return;
    }

    const allLabels = metadata.categoricalLabels[pointColor] ?? [];
    const selectedLabels = Array.from(legendSelection)
      .map((i) => allLabels[i])
      .filter((l): l is string => l !== undefined);

    onSelectionChange(selectedLabels.length > 0 ? selectedLabels : []);
  }, [legendSelection, pointColor, metadata, onSelectionChange]);

  // Track scatter instance for programmatic select API
  const [scatterInstance, setScatterInstance] = useState<ScatterInstance | null>(null);
  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  useEffect(() => {
    if (!scatterInstance || !metadata) return;

    const handle: DtourHandle = {
      select: (indicesOrMask, opts) => {
        if (opts?.isBitPacked && indicesOrMask instanceof Uint32Array) {
          scatterInstance.setSelectionMask(new Uint32Array(indicesOrMask));
        } else {
          const packed = bitPackIndices(indicesOrMask, metadata.rowCount);
          scatterInstance.setSelectionMask(packed);
        }
      },
      selectByLabels: (labels) => {
        const color = store.get(pointColorAtom);
        if (typeof color !== 'string' || !metadata.categoricalColumnNames.includes(color)) return;
        const allLabels = metadata.categoricalLabels[color] ?? [];
        const labelSet = new Set(labels);
        const indices = allLabels.map((l, i) => (labelSet.has(l) ? i : -1)).filter((i) => i >= 0);
        if (indices.length > 0) {
          scatterInstance.selectByColumn(color, { labelIndices: indices });
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

  const viewMode = useAtomValue(viewModeAtom);
  const isGrand = viewMode === 'grand';
  const legendVisible = useAtomValue(legendVisibleAtom);

  // Tour description sub-bar
  const showTourDescription = useAtomValue(showTourDescriptionAtom);
  const tourDescription = useAtomValue(tourDescriptionAtom);
  const descriptionVisible =
    showTourDescription && viewMode === 'guided' && tourDescription !== null;
  const effectiveToolbarHeight = hideToolbar ? 0 : descriptionVisible ? 72 : 40;

  // Sidebar width state — remembered across open/close cycles
  const [sidebarWidth, setSidebarWidth] = useState(200);
  const [dragging, setDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Drag-to-resize handler
  const onHandleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!legendVisible) return;
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
    [legendVisible],
  );

  const displayWidth = legendVisible ? sidebarWidth : 0;

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
              <span className="text-[11px] text-dtour-text-muted italic">
                <strong>Tour:</strong> {tourDescription}
              </span>
            </div>
          )}
        </div>
        <div className="absolute inset-0 overflow-hidden">
          <DtourViewer
            data={data}
            views={views}
            metrics={metrics}
            metricTracks={metricTracks}
            metricBarWidth={metricBarWidth}
            onStatus={onStatus}
            toolbarHeight={effectiveToolbarHeight}
            onScatterReady={setScatterInstance}
            backend={backend}
          />
        </div>
      </div>
      {/* Drag handle */}
      <div
        className={`w-px shrink-0 transition-colors ${
          legendVisible
            ? 'cursor-col-resize bg-dtour-surface hover:bg-dtour-text-muted active:bg-dtour-highlight'
            : 'pointer-events-none'
        }`}
        onMouseDown={onHandleMouseDown}
      />
      {/* Legend sidebar */}
      <div
        className="shrink-0 overflow-hidden"
        style={{
          width: displayWidth,
          transition: dragging ? 'none' : 'width 300ms cubic-bezier(.1,.1,0,1)',
        }}
      >
        <div className="h-full" style={{ width: sidebarWidth }}>
          <ColorLegend />
        </div>
      </div>
    </div>
  );
};
