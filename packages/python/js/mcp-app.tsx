// The dtour viewer as an MCP App. The `visualize` tool's result lists one or
// more views, each a Parquet file holding the data, tour, and settings. The
// viewer shows them side by side, with linked selections. It fetches the files
// from the server's localhost HTTP server or, if the host blocks that, in
// chunks through the app-only `read_data` tool. Opened directly in a browser
// (`/view/<token>`), it fetches the views named by its URL. The viewer tells
// the model what the user sees and selects via `updateModelContext`.

import { Dtour, type DtourSpec } from '@dtour/viewer';
import viewerCss from '@dtour/viewer/dist/viewer.css?inline';
import { App, type McpUiHostContext } from '@modelcontextprotocol/ext-apps/app-with-deps';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import preflightCss from './preflight.css?inline';

type ToolData = { token: string; views: { url: string; title: string }[] };

/** The `settings` argument of a `visualize` call or view. */
type Settings = Record<string, unknown> | undefined;

type Status =
  | { type: 'waiting' }
  | { type: 'computing'; label: string }
  | { type: 'error'; message: string }
  // Without `tool`, the user loaded the data themselves
  | { type: 'ready'; data: ArrayBuffer[]; key: number; tool?: ToolData };

/** What the user sees and selects in each view, and the selection they share, for the model. */
type Report = {
  specs: (Required<DtourSpec> | undefined)[];
  labels: string[][];
  selection?: string;
};

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

const readDataInChunks = async (host: App, token: string, view: number) => {
  const chunks: Uint8Array[] = [];
  let offset = 0;
  let size = Number.POSITIVE_INFINITY;
  while (offset < size) {
    const result = await host.callServerTool({
      name: 'read_data',
      arguments: { token, view, offset },
    });
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

const loadViews = async (tool: ToolData) => {
  try {
    return await Promise.all(tool.views.map((view) => fetchData(view.url)));
  } catch (error) {
    if (!app) throw error;
    // The host blocks localhost
    return Promise.all(tool.views.map((_, i) => readDataInChunks(app, tool.token, i)));
  }
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

const replaceAt = <T,>(items: T[], index: number, item: T) => {
  const copy = [...items];
  copy[index] = item;
  return copy;
};

const describe = ({ specs, labels, selection }: Report, tool: ToolData | undefined) => {
  const lines = [
    !tool
      ? 'The user loaded their own file into the dtour viewer.'
      : tool.views.length === 1
        ? 'The user is looking at the dtour viewer from the visualize call.'
        : `The user is looking at the ${tool.views.length} views from the visualize call side by side. Selections are linked across views.`,
  ];
  specs.forEach((spec, i) => {
    const parts = [];
    if (spec) {
      const color = spec.pointColorBy
        ? `colored by ${JSON.stringify(spec.pointColorBy)}`
        : 'uncolored';
      // Two tour dimensions show as a static scatter, without a tour to play
      const isStatic = spec.tourDimensions?.length === 2;
      parts.push(
        isStatic
          ? `Points ${color}.`
          : `${spec.tourTraversal} mode, ${spec.tourPlaying ? 'playing' : 'paused'}, points ${color}.`,
      );
    }
    if (labels[i]?.length) parts.push(`Legend selection: ${labels[i]!.join(', ')}.`);
    if (parts.length === 0) return;
    const title =
      tool && tool.views.length > 1 ? `View ${i + 1} (${tool.views[i]?.title})` : 'View';
    lines.push(`${title}: ${parts.join(' ')}`);
  });
  if (selection) lines.push(`Point selection: ${selection}`);
  return lines.join('\n');
};

function Viewer() {
  const [status, setStatus] = useState<Status>({ type: 'waiting' });
  const [context, setContext] = useState<McpUiHostContext | undefined>();
  const [dragging, setDragging] = useState(false);

  const [report, setReport] = useState<Report>({ specs: [], labels: [] });
  // For each view, whether the visualize call set its `theme_mode`, which then wins over the host's theme
  const [toolThemes, setToolThemes] = useState<boolean[]>([]);
  // Counts selections and data loads, so that a late summary of an older selection is dropped
  const selectionRequest = useRef(0);
  // Linked views all report a shared selection, which is described once
  const lastMask = useRef('');

  const show = (data: ArrayBuffer[], tool?: ToolData) => {
    selectionRequest.current++;
    lastMask.current = '';
    setStatus({ type: 'ready', data, key: performance.now(), tool });
    setReport({ specs: [], labels: [] });
  };
  const showFile = (data: ArrayBuffer) => show([data]);
  const fail = (error: unknown) =>
    setStatus({ type: 'error', message: error instanceof Error ? error.message : String(error) });

  useEffect(() => {
    if (!app) {
      const token = location.pathname.split('/').pop();
      fetch(`/data/${token}`)
        .then((response) => response.json() as Promise<ToolData>)
        .then(async (tool) => show(await loadViews(tool), tool))
        .catch(fail);
      return;
    }
    app.ontoolinput = ({ arguments: args }) => {
      const views = (args?.views ?? []) as { settings?: Settings }[];
      setStatus({
        type: 'computing',
        label:
          views.length > 1
            ? `Computing ${views.length} views…`
            : `Computing the ${String(args?.tour ?? 'pca')} tour…`,
      });
      // A view's settings replace the call's, as on the server, so `null` clears an inherited theme
      const callSettings = args?.settings as Settings;
      setToolThemes(
        (views.length === 0 ? [{}] : views).map((view) =>
          Boolean({ ...callSettings, ...view.settings }.theme_mode),
        ),
      );
    };
    app.ontoolresult = async (result) => {
      if (result.isError) return fail(textOf(result.content));
      const tool = result.structuredContent as ToolData;
      await loadViews(tool).then((data) => show(data, tool), fail);
    };
    app.ontoolcancelled = () => fail('The visualization was cancelled.');
    app.onhostcontextchanged = () => setContext(app.getHostContext());
    app.connect().then(() => setContext(app.getHostContext()), fail);
  }, []);

  const tool = status.type === 'ready' ? status.tool : undefined;
  const viewCount = status.type === 'ready' ? status.data.length : 0;

  // Each update replaces the previous one, so always describe the whole state
  const description = status.type === 'ready' ? describe(report, tool) : undefined;
  useEffect(() => {
    if (!app || !description) return;
    const timeout = setTimeout(() => {
      app.updateModelContext({ content: [{ type: 'text', text: description }] });
    }, 500);
    return () => clearTimeout(timeout);
  }, [description]);

  // Dtour reruns effects when these callbacks change, so keep them stable
  const viewCallbacks = useMemo(
    () =>
      Array.from({ length: viewCount }, (_, i) => ({
        onSpecChange: (spec: Required<DtourSpec>) => {
          setReport((r) => ({ ...r, specs: replaceAt(r.specs, i, spec) }));
        },
        onSelectionChange: (labels: string[]) => {
          setReport((r) =>
            (r.labels[i] ?? []).join('\n') === labels.join('\n')
              ? r
              : { ...r, labels: replaceAt(r.labels, i, labels) },
          );
        },
      })),
    [viewCount],
  );
  const onPointSelectionChange = useCallback(
    async (mask: Uint32Array) => {
      const encoded = toBase64(new Uint8Array(mask.buffer, mask.byteOffset, mask.byteLength));
      if (encoded === lastMask.current) return;
      lastMask.current = encoded;
      const request = ++selectionRequest.current;
      const count = countBits(mask);
      setReport((r) => ({ ...r, selection: count > 0 ? `${count} rows.` : undefined }));
      if (!app || !tool || count === 0) return;
      const result = await app.callServerTool({
        name: 'describe_selection',
        arguments: { token: tool.token, mask: encoded },
      });
      if (result.isError || request !== selectionRequest.current) return;
      setReport((r) => ({ ...r, selection: textOf(result.content) }));
    },
    [tool],
  );

  const fullscreen = context?.displayMode === 'fullscreen';
  const canFullscreen = context?.availableDisplayModes?.includes('fullscreen');
  const hostTheme = context?.theme;
  const specs = useMemo(
    () =>
      Array.from({ length: viewCount }, (_, i): DtourSpec | undefined => {
        const themeMode = hostTheme && !(tool && toolThemes[i]) ? hostTheme : undefined;
        // The views of one call share their selection through the call's token
        const link = tool?.token;
        return themeMode || link ? { themeMode, link } : undefined;
      }),
    [viewCount, hostTheme, tool, toolThemes],
  );

  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
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
        status.data.map((data, i) => (
          <div key={`${status.key}-${i}`} className="dtour-mcp-view">
            <Dtour
              data={data}
              spec={specs[i]}
              onLoadData={showFile}
              onSpecChange={viewCallbacks[i]?.onSpecChange}
              onSelectionChange={viewCallbacks[i]?.onSelectionChange}
              onPointSelectionChange={onPointSelectionChange}
            />
          </div>
        ))
      ) : (
        <div className="dtour-mcp-message">
          {status.type === 'waiting' && 'Loading…'}
          {status.type === 'computing' && status.label}
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
.dtour-mcp-view { position: relative; flex: 1 1 0; min-width: 0; }
/* On the viewer, so the line blends with the viewer's background, not the page's */
.dtour-mcp-view + .dtour-mcp-view > * { border-left: 1px solid rgb(128 128 128 / 0.3); }
.dtour-mcp-message {
  display: flex; align-items: center; justify-content: center; flex: 1; height: 100%;
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
