import { createRender, useModel } from '@anywidget/react';
import type { DtourHandle, DtourSpec, KeyframeLoading, RadialTrackConfig } from '@dtour/viewer';
import { Dtour } from '@dtour/viewer';
import viewerCss from '@dtour/viewer/dist/viewer.css?inline';
import { useCallback, useEffect, useRef, useState } from 'react';

type TourFamily = 'hyperdimensional' | 'sequential';

type TourMeta = {
  tourDescription?: string | null;
  keyframeDescriptions?: string | string[] | null;
  keyframeLoadings?: KeyframeLoading[] | null;
};

// Import CSS as strings so we can inject them into the Shadow DOM
import preflightCss from './preflight.css?inline';

// ---------------------------------------------------------------------------
// Traitlets (snake_case) ↔ DtourSpec (camelCase)
// ---------------------------------------------------------------------------

/** snake_case form of a camelCase string type, e.g. 'cameraPanX' → 'camera_pan_x'. */
type SnakeCase<S extends string> = S extends `${infer Head}${infer Tail}`
  ? `${Head extends Lowercase<Head> ? Head : `_${Lowercase<Head>}`}${SnakeCase<Tail>}`
  : S;

/** Traitlets that mirror DtourSpec fields. Each trait name is its spec key in snake_case. */
const SPEC_TRAITS: SnakeCase<Extract<keyof DtourSpec, string>>[] = [
  'tour_by',
  'tour_position',
  'tour_playing',
  'tour_speed',
  'tour_direction',
  'tour_slider_spacing',
  'tour_slider_visibility',
  'preview_count',
  'preview_size',
  'preview_padding',
  'preview_keyframe_numbers',
  'preview_label_content',
  'preview_label_visibility',
  'point_size',
  'point_opacity',
  'min_point_size',
  'point_color',
  'point_color_by',
  'camera_pan_x',
  'camera_pan_y',
  'camera_zoom',
  'tour_traversal',
  'show_legend',
  'show_axes',
  'show_tour_description',
  'theme_mode',
  'centering',
];

const toSpecKey = (trait: string) =>
  trait.replace(/_([a-z])/g, (_, char: string) => char.toUpperCase()) as keyof DtourSpec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert any binary buffer type into an ArrayBuffer. */
function toArrayBuffer(buf: DataView | ArrayBuffer | Uint8Array): ArrayBuffer {
  if (buf instanceof ArrayBuffer) return buf;
  if (buf instanceof DataView)
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  if (buf instanceof Uint8Array)
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return buf as ArrayBuffer;
}

function parseKeyframes(raw: DataView | ArrayBuffer | Uint8Array, nDims: number): Float32Array[] {
  const buf = toArrayBuffer(raw);
  const flat = new Float32Array(buf);
  const stride = nDims * 2;
  const nKeyframes = Math.floor(flat.length / stride);
  const keyframes: Float32Array[] = [];
  for (let i = 0; i < nKeyframes; i++) {
    keyframes.push(new Float32Array(flat.buffer, i * stride * 4, stride));
  }
  return keyframes;
}

const arraysEqual = (a: readonly unknown[], b: readonly unknown[]): boolean =>
  a.length === b.length && a.every((v, i) => v === b[i]);

// biome-ignore lint/suspicious/noExplicitAny: anywidget model is untyped
function readSpecFromModel(model: any): DtourSpec {
  const spec: Record<string, unknown> = {};
  for (const trait of SPEC_TRAITS) {
    spec[toSpecKey(trait)] = model.get(trait);
  }
  return spec as DtourSpec;
}

// ---------------------------------------------------------------------------
// Widget component
// ---------------------------------------------------------------------------

function Widget() {
  const model = useModel();
  const [data, setData] = useState<ArrayBuffer | undefined>();
  const [keyframes, setKeyframes] = useState<Float32Array[] | undefined>();
  const [metrics, setMetrics] = useState<ArrayBuffer | undefined>();
  const [tourMeta, setTourMeta] = useState<TourMeta>({});
  const [spec, setSpec] = useState<DtourSpec>(() => readSpecFromModel(model));
  const suppressRef = useRef(false);
  const dtourApiRef = useRef<DtourHandle | null>(null);

  // ── Bidirectional selection sync ──────────────────────────────────────
  // Guards to prevent infinite update loops: when JS pushes a selection to
  // Python, we record what we sent so the incoming `change:` echo is ignored.
  const lastSyncedLabels = useRef<string[]>([]);
  const lastSyncedIndices = useRef<number[]>([]);

  const handleReady = useCallback(
    (api: DtourHandle) => {
      dtourApiRef.current = api;
      // Apply any selection state that was set before the viewer became ready
      const labels: string[] = model.get('selected_labels') ?? [];
      const indices: number[] = model.get('selected_indices') ?? [];
      if (labels.length > 0) {
        lastSyncedLabels.current = labels;
        api.selectByLabels(labels);
      } else if (indices.length > 0) {
        lastSyncedIndices.current = indices;
        api.select(indices);
      }
    },
    [model],
  );

  // Detect Shadow DOM and create a dedicated portal container inside it so
  // Radix popovers/dropdowns/tooltips render within the shadow boundary and
  // inherit scoped styles instead of escaping to document.body.
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [portalContainer, setPortalContainer] = useState<HTMLElement | undefined>();

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const root = el.getRootNode();
    if (root instanceof ShadowRoot) {
      const portal = document.createElement('div');
      portal.setAttribute('data-dtour-portal', '');
      root.appendChild(portal);
      setPortalContainer(portal);
      return () => portal.remove();
    }
  }, []);

  // Custom messages → data / keyframes / metrics (binary buffers from Python)
  useEffect(() => {
    // biome-ignore lint/suspicious/noExplicitAny: anywidget buffer type varies by host
    function onMsg(msg: Record<string, any>, buffers: any[]) {
      console.log('[dtour] onMsg', msg.type, 'buffers:', buffers.length);
      if (msg.type === 'data' && buffers[0]) {
        setData(toArrayBuffer(buffers[0]));
      } else if (msg.type === 'keyframes' && buffers[0] && msg.n_dims) {
        setKeyframes(parseKeyframes(buffers[0], msg.n_dims));
        setTourMeta({
          tourDescription: msg.tour_description ?? null,
          keyframeDescriptions: msg.keyframe_descriptions ?? null,
          keyframeLoadings: msg.keyframe_loadings ?? null,
        });
      } else if (msg.type === 'metrics' && buffers[0]) {
        const ab = toArrayBuffer(buffers[0]);
        console.log('[dtour] metrics received, byteLength:', ab.byteLength);
        setMetrics(ab);
      } else if (msg.type === 'select' && buffers[0]) {
        const ab = toArrayBuffer(buffers[0]);
        const indices = new Int32Array(ab);
        dtourApiRef.current?.select(Array.from(indices));
      } else if (msg.type === 'clear_selection') {
        dtourApiRef.current?.clearSelection();
      }
    }
    model.on('msg:custom', onMsg);

    // Signal ready so Python (re-)sends binary data
    model.send({ type: 'ready' });

    return () => model.off('msg:custom', onMsg);
  }, [model]);

  // Traitlet changes → spec (inbound from Python)
  useEffect(() => {
    function onChange() {
      if (!suppressRef.current) {
        setSpec(readSpecFromModel(model));
      }
    }
    for (const trait of SPEC_TRAITS) {
      model.on(`change:${trait}`, onChange);
    }
    return () => {
      for (const trait of SPEC_TRAITS) {
        model.off(`change:${trait}`, onChange);
      }
    };
  }, [model]);

  // onSpecChange → traitlet sync (outbound from UI interaction)
  const handleSpecChange = useCallback(
    (newSpec: Required<DtourSpec>) => {
      suppressRef.current = true;
      for (const trait of SPEC_TRAITS) {
        const value = newSpec[toSpecKey(trait)];
        if (value !== undefined) {
          model.set(trait, value);
        }
      }
      model.save_changes();
      queueMicrotask(() => {
        suppressRef.current = false;
      });
    },
    [model],
  );

  // JS → Python: lasso point selection (bit-packed mask → index list)
  const handlePointSelectionChange = useCallback(
    (mask: Uint32Array) => {
      const indices: number[] = [];
      for (let w = 0; w < mask.length; w++) {
        let bits = mask[w]!;
        while (bits !== 0) {
          const bit = bits & -bits; // lowest set bit
          indices.push((w << 5) + (31 - Math.clz32(bit)));
          bits ^= bit;
        }
      }
      lastSyncedIndices.current = indices;
      // Clear label selection when point selection comes from lasso
      lastSyncedLabels.current = [];
      model.set('selected_indices', indices);
      model.set('selected_labels', []);
      model.save_changes();
    },
    [model],
  );

  // JS → Python: legend/lasso selection changed in the viewer
  const handleSelectionChange = useCallback(
    (labels: string[]) => {
      if (arraysEqual(labels, lastSyncedLabels.current)) return;
      lastSyncedLabels.current = labels;
      // Clear index selection when label selection changes from UI
      lastSyncedIndices.current = [];
      model.set('selected_labels', labels);
      model.set('selected_indices', []);
      model.save_changes();
    },
    [model],
  );

  // Python → JS: traitlet changed from Python side (or linked widget)
  useEffect(() => {
    function onLabelsChange() {
      const labels: string[] = model.get('selected_labels') ?? [];
      if (arraysEqual(labels, lastSyncedLabels.current)) return;
      lastSyncedLabels.current = labels;
      if (labels.length === 0) {
        dtourApiRef.current?.clearSelection();
      } else {
        dtourApiRef.current?.selectByLabels(labels);
      }
    }
    function onIndicesChange() {
      const indices: number[] = model.get('selected_indices') ?? [];
      if (arraysEqual(indices, lastSyncedIndices.current)) return;
      lastSyncedIndices.current = indices;
      if (indices.length === 0) {
        dtourApiRef.current?.clearSelection();
      } else {
        dtourApiRef.current?.select(indices);
      }
    }
    model.on('change:selected_labels', onLabelsChange);
    model.on('change:selected_indices', onIndicesChange);
    return () => {
      model.off('change:selected_labels', onLabelsChange);
      model.off('change:selected_indices', onIndicesChange);
    };
  }, [model]);

  const height: number = model.get('height') ?? 600;
  const [metricBarWidth, setMetricBarWidth] = useState<'full' | number>(
    () => model.get('metric_bar_width') ?? 'full',
  );
  const [metricTracks, setMetricTracks] = useState<RadialTrackConfig[]>(
    () => model.get('metric_tracks') ?? [],
  );
  const [tourFamily, setTourFamily] = useState<TourFamily | undefined>(
    () => model.get('_tour_family') ?? undefined,
  );
  const [tourDimensions, setTourDimensions] = useState<string[]>(
    () => model.get('tour_dimensions') ?? [],
  );
  const [colorMap, setColorMap] = useState<
    Record<string, string | { light: string; dark: string }> | undefined
  >(() => {
    const raw = model.get('color_map');
    return raw && Object.keys(raw).length > 0 ? raw : undefined;
  });

  useEffect(() => {
    function onBarWidth() {
      setMetricBarWidth(model.get('metric_bar_width') ?? 'full');
    }
    function onTracks() {
      setMetricTracks(model.get('metric_tracks') ?? []);
    }
    function onTourFamily() {
      setTourFamily(model.get('_tour_family') ?? undefined);
    }
    function onTourDimensions() {
      setTourDimensions(model.get('tour_dimensions') ?? []);
    }
    function onColorMap() {
      const raw = model.get('color_map');
      setColorMap(raw && Object.keys(raw).length > 0 ? raw : undefined);
    }
    model.on('change:metric_bar_width', onBarWidth);
    model.on('change:metric_tracks', onTracks);
    model.on('change:_tour_family', onTourFamily);
    model.on('change:tour_dimensions', onTourDimensions);
    model.on('change:color_map', onColorMap);
    return () => {
      model.off('change:metric_bar_width', onBarWidth);
      model.off('change:metric_tracks', onTracks);
      model.off('change:_tour_family', onTourFamily);
      model.off('change:tour_dimensions', onTourDimensions);
      model.off('change:color_map', onColorMap);
    };
  }, [model]);

  return (
    <div
      ref={wrapperRef}
      className="w-full"
      style={{ height: `${height}px`, position: 'relative' }}
    >
      <Dtour
        data={data}
        keyframes={keyframes}
        tourDimensions={tourDimensions.length > 0 ? tourDimensions : undefined}
        metrics={metrics}
        metricTracks={metricTracks.length > 0 ? metricTracks : undefined}
        metricBarWidth={metricBarWidth}
        colorMap={colorMap}
        tourFamily={tourFamily}
        tourDescription={tourMeta.tourDescription}
        keyframeDescriptions={tourMeta.keyframeDescriptions}
        keyframeLoadings={tourMeta.keyframeLoadings}
        spec={spec}
        onSpecChange={handleSpecChange}
        onSelectionChange={handleSelectionChange}
        onPointSelectionChange={handlePointSelectionChange}
        onReady={handleReady}
        portalContainer={portalContainer}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shadow DOM render wrapper
// ---------------------------------------------------------------------------
// anywidget's createRender handles the React root + model context.
// We wrap it to mount everything inside a Shadow DOM for style isolation.

const innerRender = createRender(Widget);

export default {
  // biome-ignore lint/suspicious/noExplicitAny: anywidget render protocol
  render(props: any) {
    const host = props.el as HTMLElement;
    let shadow = host.shadowRoot;

    if (!shadow) {
      shadow = host.attachShadow({ mode: 'open' });
    }

    // Clear previous content (styles, containers) on re-render
    shadow.innerHTML = '';

    // Inject scoped CSS into the shadow root (not <head>)
    const style = document.createElement('style');
    style.textContent = preflightCss + viewerCss;
    shadow.appendChild(style);

    // React mounts into this container
    const container = document.createElement('div');
    container.style.width = '100%';
    container.style.height = '100%';
    shadow.appendChild(container);

    // Forward all props (model, experimental, etc.) with el swapped
    return innerRender({ ...props, el: container });
  },
};
