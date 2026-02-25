import { createRender, useModel } from '@anywidget/react';
import { Dtour } from '@dtour/viewer';
import '@dtour/viewer/dist/viewer.css';
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
};

const TRAIT_NAMES = Object.keys(TRAIT_TO_SPEC);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function dataViewToArrayBuffer(dv: DataView): ArrayBuffer {
  return dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength);
}

function parseBases(dv: DataView, nDims: number): Float32Array[] {
  const buf = dataViewToArrayBuffer(dv);
  const flat = new Float32Array(buf);
  const stride = nDims * 2;
  const nViews = Math.floor(flat.length / stride);
  const bases: Float32Array[] = [];
  for (let i = 0; i < nViews; i++) {
    bases.push(new Float32Array(flat.buffer, i * stride * 4, stride));
  }
  return bases;
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
  const [bases, setBases] = useState<Float32Array[] | undefined>();
  const [metrics, setMetrics] = useState<ArrayBuffer | undefined>();
  const [spec, setSpec] = useState<DtourSpec>(() => readSpecFromModel(model));
  const suppressRef = useRef(false);

  // Custom messages → data / bases / metrics (binary buffers from Python)
  useEffect(() => {
    function onMsg(msg: { type: string; n_dims?: number }, buffers: DataView[]) {
      if (msg.type === 'data' && buffers[0]) {
        setData(dataViewToArrayBuffer(buffers[0]));
      } else if (msg.type === 'bases' && buffers[0] && msg.n_dims) {
        setBases(parseBases(buffers[0], msg.n_dims));
      } else if (msg.type === 'metrics' && buffers[0]) {
        setMetrics(dataViewToArrayBuffer(buffers[0]));
      }
    }
    model.on('msg:custom', onMsg);
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

  const height: number = model.get('height') ?? 600;

  return (
    <div className="w-full" style={{ height: `${height}px` }}>
      <Dtour
        data={data}
        bases={bases}
        metrics={metrics}
        spec={spec}
        onSpecChange={handleSpecChange}
      />
    </div>
  );
}

export default { render: createRender(Widget) };
