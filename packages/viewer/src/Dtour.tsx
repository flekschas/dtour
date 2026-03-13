import type { ScatterStatus } from '@dtour/scatter';
import { Provider, createStore, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { DtourViewer } from './DtourViewer.tsx';
import { ColorLegend } from './components/ColorLegend.tsx';
import { DtourToolbar } from './components/DtourToolbar.tsx';
import { useModeCycling } from './hooks/useModeCycling.ts';
import { useSettingsPersistence } from './hooks/useSettingsPersistence.ts';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import type { DtourSpec } from './spec.ts';
import { dataNameAtom, legendVisibleAtom, viewModeAtom } from './state/atoms.ts';
import { initStoreFromSpec, useSpecSync } from './state/spec-sync.ts';

export type DtourProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer;
  /** Tour keyframe views (p×2 column-major). Auto-generated if omitted. */
  views?: Float32Array[];
  /** Arrow IPC ArrayBuffer with per-view quality metrics (columns = metrics, rows = views). */
  metrics?: ArrayBuffer;
  /** Track configuration for radial bar charts. When omitted, all metrics are shown with defaults. */
  metricTracks?: RadialTrackConfig[];
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
  /** Name identifying the current dataset (e.g. filename). Used as localStorage key for settings persistence. */
  dataName?: string;
};

export const Dtour = ({
  data,
  views,
  metrics,
  metricTracks,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar = false,
  onLoadData,
  dataName,
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
    <Provider store={store}>
      <DtourInner
        data={data}
        views={views}
        metrics={metrics}
        metricTracks={metricTracks}
        spec={spec}
        onSpecChange={onSpecChange}
        onStatus={onStatus}
        hideToolbar={hideToolbar}
        onLoadData={onLoadData}
        dataName={dataName}
      />
    </Provider>
  );
};

/** Inner component that lives inside the Provider so hooks bind to the store. */
const DtourInner = ({
  data,
  views,
  metrics,
  metricTracks,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar,
  onLoadData,
  dataName,
}: {
  data: ArrayBuffer | undefined;
  views: Float32Array[] | undefined;
  metrics: ArrayBuffer | undefined;
  metricTracks: RadialTrackConfig[] | undefined;
  spec: DtourSpec | undefined;
  onSpecChange: ((spec: Required<DtourSpec>) => void) | undefined;
  onStatus: ((status: ScatterStatus) => void) | undefined;
  hideToolbar: boolean;
  onLoadData: ((data: ArrayBuffer, fileName: string) => void) | undefined;
  dataName: string | undefined;
}) => {
  useSpecSync(spec, onSpecChange);
  useModeCycling();
  useSettingsPersistence();

  const [mounted, setMounted] = useState(false);

  const viewerRef = useRef<HTMLDivElement>(null);

  // Sync dataName prop → atom for settings persistence
  const setDataName = useSetAtom(dataNameAtom);
  useEffect(() => {
    setDataName(dataName ?? null);
  }, [dataName, setDataName]);

  const viewMode = useAtomValue(viewModeAtom);
  const isGrand = viewMode === 'grand';
  const legendVisible = useAtomValue(legendVisibleAtom);

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

  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      console.log(width, height);
    });
    ro.observe(viewer);
    setMounted(true);
    return () => ro.disconnect();
  }, []);

  const displayWidth = legendVisible ? sidebarWidth : 0;

  return (
    <div ref={containerRef} className="relative w-full h-full overflow-hidden flex">
      {/* Canvas panel — grows to fill remaining space */}
      <div className="relative flex-1 min-w-0">
        {/* Toolbar — inside left panel so it shrinks with the legend */}
        <div
          className={`absolute inset-x-0 top-0 z-10 h-10 transition-[transform,opacity] duration-300 ease-out ${
            isGrand ? '-translate-y-full' : 'translate-y-0'
          } ${hideToolbar ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
        >
          <DtourToolbar onLoadData={onLoadData} />
        </div>
        <div ref={viewerRef} className="absolute inset-0 overflow-hidden">
          {mounted && (
            <DtourViewer
              data={data}
              views={views}
              metrics={metrics}
              metricTracks={metricTracks}
              onStatus={onStatus}
              toolbarHeight={hideToolbar ? 0 : 40}
            />
          )}
        </div>
      </div>
      {/* Drag handle */}
      <div
        className={`w-px shrink-0 transition-colors ${
          legendVisible
            ? 'cursor-col-resize bg-dtour-surface hover:bg-dtour-text-muted active:bg-white'
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
