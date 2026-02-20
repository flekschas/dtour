import { createScatter } from '@dtour/scatter';
import type { ScatterInstance, ScatterStatus } from '@dtour/scatter';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createDefaultBases } from './bases.ts';
import { CircularSelector } from './circular-range-selector.tsx';
import { Gallery } from './components/Gallery.tsx';
import { useScatter } from './hooks/useScatter.ts';
import { computeGalleryPositions } from './layout/gallery-positions.ts';
import { metadataAtom, tourPositionAtom, viewCountAtom } from './state/atoms.ts';

export type DtourProps = {
  /** Arrow IPC or Parquet ArrayBuffer. Ownership is transferred on load. */
  data?: ArrayBuffer | undefined;
  /** Tour keyframe bases (p×2 column-major). Auto-generated if omitted. */
  bases?: Float32Array[];
  /** Number of dimensions — required if bases is provided. */
  dims?: number;
  /** Called on every status event from the renderer. */
  onStatus?: (status: ScatterStatus) => void;
};

const MIN_SELECTOR_SIZE = 80;
const PREVIEW_PHYSICAL_SIZE = 200; // Physical pixels per preview canvas

export const Dtour = ({ data, bases, dims, onStatus }: DtourProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const scatterRef = useRef<ScatterInstance | null>(null);
  const previewCanvasesRef = useRef<HTMLCanvasElement[]>([]);
  const [position, setPosition] = useAtom(tourPositionAtom);
  const metadata = useAtomValue(metadataAtom);
  const setMetadata = useSetAtom(metadataAtom);
  const viewCount = useAtomValue(viewCountAtom);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const lastDataRef = useRef<ArrayBuffer | undefined>(undefined);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;

  // Bridge Jotai atoms (style, camera) → scatter instance
  useScatter(scatterRef.current);

  // Compute gallery positions + padding for canvas insets
  const { previewSize, padX, padY } = useMemo(
    () => computeGalleryPositions(containerSize.width, containerSize.height, viewCount),
    [containerSize.width, containerSize.height, viewCount],
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
    for (let i = 0; i < viewCount; i++) {
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

    // Resize main canvas + update container size for gallery layout
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const { width, height } = entry.contentRect;
      const curDpr = window.devicePixelRatio || 1;
      scatter.resize(0, Math.round(width * curDpr), Math.round(height * curDpr));
      setContainerSize({ width, height });
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
  }, [setMetadata]);

  // Send data when it changes
  useEffect(() => {
    if (!data || !scatterRef.current || data === lastDataRef.current) return;
    lastDataRef.current = data;
    scatterRef.current.loadData(data.slice(0));
  }, [data]);

  // Set bases when available (from props or auto-generated from metadata)
  useEffect(() => {
    const scatter = scatterRef.current;
    if (!scatter) return;
    if (bases && dims) {
      scatter.setBases(
        bases.map((b) => new Float32Array(b)),
        dims,
      );
    } else if (metadata && metadata.dimCount >= 2) {
      const defaultBases = createDefaultBases(metadata.dimCount, viewCount);
      scatter.setBases(defaultBases, metadata.dimCount);
    }
    // Safety: explicitly request a full re-render after bases are set,
    // ensuring all preview canvases get painted even if messages race.
    scatter.render();
  }, [bases, dims, metadata, viewCount]);

  const handlePositionChange = useCallback(
    (pos: number) => {
      setPosition(pos);
    },
    [setPosition],
  );

  const tickCount = bases?.length ?? viewCount;
  const hasData = !!data;

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        background: '#0f0f14',
      }}
    >
      {/* Preview gallery — positioned in corners */}
      {hasData && containerSize.width > 0 && previewCanvasesRef.current.length > 0 && (
        <Gallery
          previewCanvases={previewCanvasesRef.current}
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
        />
      )}

      {/* Circular selector overlay — centered */}
      {hasData && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
          }}
        >
          <div style={{ pointerEvents: 'auto' }}>
            <CircularSelector
              value={position}
              onChange={handlePositionChange}
              tickCount={tickCount}
              size={selectorSize}
            />
          </div>
        </div>
      )}

      {/* Empty state — shown when no data is loaded */}
      {!hasData && containerSize.width > 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 12,
            color: '#888899',
            pointerEvents: 'none',
          }}
        >
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            role="img"
            aria-labelledby="upload-icon-title"
          >
            <title id="upload-icon-title">Upload file</title>
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <span style={{ fontSize: 14 }}>Drop a Parquet or Arrow file to start</span>
        </div>
      )}
    </div>
  );
};
