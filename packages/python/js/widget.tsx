import { createRender, useModel } from '@anywidget/react';
import { Dtour } from '@dtour/viewer';
// Import CSS as strings so we can inject them into the Shadow DOM
import preflightCss from './preflight.css?inline';
import viewerCss from '@dtour/viewer/dist/viewer.css?inline';
import type { DtourSpec } from '@dtour/viewer';
import { useCallback, useEffect, useRef, useState } from 'react';

// ---------------------------------------------------------------------------
// Traitlet (snake_case) ↔ DtourSpec (camelCase) mapping
// ---------------------------------------------------------------------------

const TRAIT_TO_SPEC: Record<string, keyof DtourSpec> = {
  tour_position: 'tourPosition',
  tour_playing: 'tourPlaying',
  tour_speed: 'tourSpeed',
  tour_direction: 'tourDirection',
  preview_count: 'previewCount',
  preview_padding: 'previewPadding',
  point_size: 'pointSize',
  point_opacity: 'pointOpacity',
  point_color: 'pointColor',
  camera_pan_x: 'cameraPanX',
  camera_pan_y: 'cameraPanY',
  camera_zoom: 'cameraZoom',
  view_mode: 'viewMode',
};

const TRAIT_NAMES = Object.keys(TRAIT_TO_SPEC);

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

function parseViews(
  raw: DataView | ArrayBuffer | Uint8Array,
  nDims: number,
): Float32Array[] {
  const buf = toArrayBuffer(raw);
  const flat = new Float32Array(buf);
  const stride = nDims * 2;
  const nViews = Math.floor(flat.length / stride);
  const views: Float32Array[] = [];
  for (let i = 0; i < nViews; i++) {
    views.push(new Float32Array(flat.buffer, i * stride * 4, stride));
  }
  return views;
}

// biome-ignore lint/suspicious/noExplicitAny: anywidget model is untyped
function readSpecFromModel(model: any): DtourSpec {
  const spec: Record<string, unknown> = {};
  for (const [trait, specKey] of Object.entries(TRAIT_TO_SPEC)) {
    spec[specKey] = model.get(trait);
  }
  return spec as DtourSpec;
}

// ---------------------------------------------------------------------------
// Widget component
// ---------------------------------------------------------------------------

function Widget() {
  const model = useModel();
  const [data, setData] = useState<ArrayBuffer | undefined>();
  const [views, setViews] = useState<Float32Array[] | undefined>();
  const [metrics, setMetrics] = useState<ArrayBuffer | undefined>();
  const [spec, setSpec] = useState<DtourSpec>(() => readSpecFromModel(model));
  const suppressRef = useRef(false);

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

  // Custom messages → data / views / metrics (binary buffers from Python)
  useEffect(() => {
    // biome-ignore lint/suspicious/noExplicitAny: anywidget buffer type varies by host
    function onMsg(msg: { type: string; n_dims?: number }, buffers: any[]) {
      if (msg.type === 'data' && buffers[0]) {
        setData(toArrayBuffer(buffers[0]));
      } else if (msg.type === 'views' && buffers[0] && msg.n_dims) {
        setViews(parseViews(buffers[0], msg.n_dims));
      } else if (msg.type === 'metrics' && buffers[0]) {
        setMetrics(toArrayBuffer(buffers[0]));
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
    for (const trait of TRAIT_NAMES) {
      model.on(`change:${trait}`, onChange);
    }
    return () => {
      for (const trait of TRAIT_NAMES) {
        model.off(`change:${trait}`, onChange);
      }
    };
  }, [model]);

  // onSpecChange → traitlet sync (outbound from UI interaction)
  const handleSpecChange = useCallback(
    (newSpec: Required<DtourSpec>) => {
      suppressRef.current = true;
      for (const [trait, specKey] of Object.entries(TRAIT_TO_SPEC)) {
        model.set(trait, newSpec[specKey]);
      }
      model.save_changes();
      queueMicrotask(() => {
        suppressRef.current = false;
      });
    },
    [model],
  );

  // Selection → traitlet sync (JS → Python)
  const handleSelectionChange = useCallback(
    (labels: string[]) => {
      model.set('selected_labels', labels);
      model.save_changes();
    },
    [model],
  );

  const height: number = model.get('height') ?? 600;
  const [metricBarWidth, setMetricBarWidth] = useState<'full' | number>(
    () => model.get('metric_bar_width') ?? 'full',
  );

  useEffect(() => {
    function onChange() {
      setMetricBarWidth(model.get('metric_bar_width') ?? 'full');
    }
    model.on('change:metric_bar_width', onChange);
    return () => model.off('change:metric_bar_width', onChange);
  }, [model]);

  return (
    <div ref={wrapperRef} className="w-full" style={{ height: `${height}px`, position: 'relative' }}>
      <Dtour
        data={data}
        views={views}
        metrics={metrics}
        metricBarWidth={metricBarWidth}
        spec={spec}
        onSpecChange={handleSpecChange}
        onSelectionChange={handleSelectionChange}
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
    const shadow = (props.el as HTMLElement).attachShadow({ mode: 'open' });

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
