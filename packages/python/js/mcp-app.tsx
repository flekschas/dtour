// The dtour viewer as an MCP App. The `visualize` tool's result names a Parquet
// file holding the data, tour, and settings. The viewer fetches it from the
// server's localhost HTTP server or, if the host blocks that, in chunks through
// the app-only `read_data` tool. Opened directly in a browser (`/view/<token>`),
// it fetches the file named by its URL. The viewer tells the model what the
// user sees and selects via `updateModelContext`.

import { Dtour, type DtourSpec } from '@dtour/viewer';
import viewerCss from '@dtour/viewer/dist/viewer.css?inline';
import { App, type McpUiHostContext } from '@modelcontextprotocol/ext-apps/app-with-deps';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import preflightCss from './preflight.css?inline';

type ToolData = { url: string; token: string };

type Status =
  | { type: 'waiting' }
  | { type: 'computing'; tour: string }
  | { type: 'error'; message: string }
  // Without `view`, the user loaded the data themselves
  | { type: 'ready'; data: ArrayBuffer; key: number; view?: ToolData };

/** What the user sees and selects, for the model. */
type Report = { spec?: Required<DtourSpec>; labels: string[]; selection?: string };

const INLINE_HEIGHT = 560;
const standalone = window.parent === window;
const app = standalone
  ? undefined
  : new App(
      { name: 'dtour', version: '1.0.0' },
      { availableDisplayModes: ['inline', 'fullscreen'] },
    );

const fetchData = async (url: string) => {
  const response = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.arrayBuffer();
};

const readDataInChunks = async (host: App, token: string) => {
  const chunks: Uint8Array[] = [];
  let offset = 0;
  let size = Number.POSITIVE_INFINITY;
  while (offset < size) {
    const result = await host.callServerTool({ name: 'read_data', arguments: { token, offset } });
    if (result.isError) throw new Error(textOf(result.content));
    const chunk = result.structuredContent as { base64: string; size: number };
    const bytes = Uint8Array.from(atob(chunk.base64), (c) => c.charCodeAt(0));
    chunks.push(bytes);
    offset += bytes.length;
    size = chunk.size;
  }
  const data = new Uint8Array(size);
  offset = 0;
  for (const bytes of chunks) {
    data.set(bytes, offset);
    offset += bytes.length;
  }
  return data.buffer;
};

const textOf = (content: unknown) =>
  Array.isArray(content)
    ? content.map((block) => (block?.type === 'text' ? block.text : '')).join('\n')
    : '';

const toBase64 = (bytes: Uint8Array) => {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
};

const countBits = (mask: Uint32Array) => {
  let count = 0;
  for (let word of mask) {
    while (word) {
      word &= word - 1;
      count++;
    }
  }
  return count;
};

const describe = ({ spec, labels, selection }: Report, view: ToolData | undefined) => {
  const lines = [
    view
      ? 'The user is looking at the dtour viewer from the visualize call.'
      : 'The user loaded their own file into the dtour viewer.',
  ];
  if (spec) {
    const color = spec.pointColorBy
      ? `colored by ${JSON.stringify(spec.pointColorBy)}`
      : 'uncolored';
    lines.push(
      `View: ${spec.tourTraversal} mode, ${spec.tourPlaying ? 'playing' : 'paused'}, points ${color}.`,
    );
  }
  if (labels.length > 0) lines.push(`Legend selection: ${labels.join(', ')}.`);
  if (selection) lines.push(`Point selection: ${selection}`);
  return lines.join('\n');
};

function Viewer() {
  const [status, setStatus] = useState<Status>({ type: 'waiting' });
  const [context, setContext] = useState<McpUiHostContext | undefined>();
  const [dragging, setDragging] = useState(false);

  const [report, setReport] = useState<Report>({ labels: [] });
  // Whether the visualize call set `theme_mode`, which then wins over the host's theme
  const [toolSetsTheme, setToolSetsTheme] = useState(false);
  // Counts selections and data loads, so that a late summary of an older selection is dropped
  const selectionRequest = useRef(0);

  const show = (data: ArrayBuffer, view?: ToolData) => {
    selectionRequest.current++;
    setStatus({ type: 'ready', data, key: performance.now(), view });
    setReport({ labels: [] });
  };
  const showFile = (data: ArrayBuffer) => show(data);
  const fail = (error: unknown) =>
    setStatus({ type: 'error', message: error instanceof Error ? error.message : String(error) });

  useEffect(() => {
    if (!app) {
      const token = location.pathname.split('/').pop();
      fetchData(`/data/${token}`).then(showFile, fail);
      return;
    }
    app.ontoolinput = ({ arguments: args }) => {
      setStatus({ type: 'computing', tour: String(args?.tour ?? 'pca') });
      setToolSetsTheme(
        Boolean((args?.settings as Record<string, unknown> | undefined)?.theme_mode),
      );
    };
    app.ontoolresult = async (result) => {
      if (result.isError) return fail(textOf(result.content));
      const view = result.structuredContent as ToolData;
      try {
        show(await fetchData(view.url), view);
      } catch {
        // The host blocks localhost
        await readDataInChunks(app, view.token).then((data) => show(data, view), fail);
      }
    };
    app.ontoolcancelled = () => fail('The visualization was cancelled.');
    app.onhostcontextchanged = () => setContext(app.getHostContext());
    app.connect().then(() => setContext(app.getHostContext()), fail);
  }, []);

  const view = status.type === 'ready' ? status.view : undefined;

  // Each update replaces the previous one, so always describe the whole state
  const description = status.type === 'ready' ? describe(report, view) : undefined;
  useEffect(() => {
    if (!app || !description) return;
    const timeout = setTimeout(() => {
      app.updateModelContext({ content: [{ type: 'text', text: description }] });
    }, 500);
    return () => clearTimeout(timeout);
  }, [description]);

  // Dtour reruns effects when these callbacks change, so keep them stable
  const onSpecChange = useCallback((spec: Required<DtourSpec>) => {
    setReport((r) => ({ ...r, spec }));
  }, []);
  const onSelectionChange = useCallback((labels: string[]) => {
    setReport((r) => (r.labels.join('\n') === labels.join('\n') ? r : { ...r, labels }));
  }, []);
  const onPointSelectionChange = useCallback(
    async (mask: Uint32Array) => {
      const request = ++selectionRequest.current;
      const count = countBits(mask);
      setReport((r) => ({ ...r, selection: count > 0 ? `${count} rows.` : undefined }));
      if (!app || !view || count === 0) return;
      const result = await app.callServerTool({
        name: 'describe_selection',
        arguments: { token: view.token, mask: toBase64(new Uint8Array(mask.buffer)) },
      });
      if (result.isError || request !== selectionRequest.current) return;
      setReport((r) => ({ ...r, selection: textOf(result.content) }));
    },
    [view],
  );

  const fullscreen = context?.displayMode === 'fullscreen';
  const canFullscreen = context?.availableDisplayModes?.includes('fullscreen');
  const followHostTheme = Boolean(context?.theme) && !(view && toolSetsTheme);
  const spec = useMemo<DtourSpec | undefined>(
    () => (followHostTheme ? { themeMode: context?.theme } : undefined),
    [followHostTheme, context?.theme],
  );

  return (
    <div
      style={{
        position: 'relative',
        height: standalone || fullscreen ? '100vh' : INLINE_HEIGHT,
        outline: dragging ? '2px dashed #888' : 'none',
        outlineOffset: -2,
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        e.dataTransfer.files[0]?.arrayBuffer().then(showFile, fail);
      }}
    >
      {status.type === 'ready' ? (
        <Dtour
          key={status.key}
          data={status.data}
          spec={spec}
          onLoadData={showFile}
          onSpecChange={onSpecChange}
          onSelectionChange={onSelectionChange}
          onPointSelectionChange={onPointSelectionChange}
        />
      ) : (
        <div className="dtour-mcp-message">
          {status.type === 'waiting' && 'Loading…'}
          {status.type === 'computing' && `Computing the ${status.tour} tour…`}
          {status.type === 'error' && status.message}
        </div>
      )}
      {app && canFullscreen && (
        <button
          type="button"
          className="dtour-mcp-fullscreen"
          title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          onClick={() => app.requestDisplayMode({ mode: fullscreen ? 'inline' : 'fullscreen' })}
        >
          {fullscreen ? '⤡' : '⤢'}
        </button>
      )}
    </div>
  );
}

const style = document.createElement('style');
style.textContent = `${preflightCss}${viewerCss}
html, body { margin: 0; background: transparent; }
.dtour-mcp-message {
  display: flex; align-items: center; justify-content: center; height: 100%;
  padding: 1rem; font: 14px system-ui, sans-serif; color: #888; text-align: center;
}
.dtour-mcp-fullscreen {
  position: absolute; right: 8px; bottom: 8px; z-index: 50; width: 28px; height: 28px;
  border: 0; border-radius: 6px; background: rgb(128 128 128 / 0.25); color: inherit;
  font-size: 16px; cursor: pointer;
}`;
document.head.appendChild(style);

const root = document.createElement('div');
document.body.appendChild(root);
createRoot(root).render(<Viewer />);
