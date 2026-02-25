import type { ScatterStatus } from '@dtour/scatter';
import { Provider, createStore, useAtomValue } from 'jotai';
import { useMemo } from 'react';
import { DtourViewer } from './DtourViewer.tsx';
import { DtourToolbar } from './components/DtourToolbar.tsx';
import { useModeCycling } from './hooks/useModeCycling.ts';
import type { RadialTrackConfig } from './radial-chart/types.ts';
import type { DtourSpec } from './spec.ts';
import { viewModeAtom } from './state/atoms.ts';
import { initStoreFromSpec, useSpecSync } from './state/spec-sync.ts';

export type DtourProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer;
  /** Tour keyframe bases (p×2 column-major). Auto-generated if omitted. */
  bases?: Float32Array[];
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
};

export const Dtour = ({
  data,
  bases,
  metrics,
  metricTracks,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar = false,
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
        bases={bases}
        metrics={metrics}
        metricTracks={metricTracks}
        spec={spec}
        onSpecChange={onSpecChange}
        onStatus={onStatus}
        hideToolbar={hideToolbar}
      />
    </Provider>
  );
};

/** Inner component that lives inside the Provider so hooks bind to the store. */
const DtourInner = ({
  data,
  bases,
  metrics,
  metricTracks,
  spec,
  onSpecChange,
  onStatus,
  hideToolbar,
}: {
  data: ArrayBuffer | undefined;
  bases: Float32Array[] | undefined;
  metrics: ArrayBuffer | undefined;
  metricTracks: RadialTrackConfig[] | undefined;
  spec: DtourSpec | undefined;
  onSpecChange: ((spec: Required<DtourSpec>) => void) | undefined;
  onStatus: ((status: ScatterStatus) => void) | undefined;
  hideToolbar: boolean;
}) => {
  useSpecSync(spec, onSpecChange);
  useModeCycling();

  const viewMode = useAtomValue(viewModeAtom);
  const isZen = viewMode === 'zen';

  return (
    <div className="relative w-full h-full overflow-hidden">
      <div className="absolute inset-0">
        <DtourViewer
          data={data}
          bases={bases}
          metrics={metrics}
          metricTracks={metricTracks}
          onStatus={onStatus}
          toolbarHeight={hideToolbar ? 0 : 40}
        />
      </div>
      {!hideToolbar && (
        <div
          className={`absolute inset-x-0 top-0 z-10 h-10 transition-transform duration-300 ease-in-out ${
            isZen ? '-translate-y-full' : 'translate-y-0'
          }`}
        >
          <DtourToolbar />
        </div>
      )}
    </div>
  );
};
