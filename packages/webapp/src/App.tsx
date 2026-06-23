import type { DtourSpec } from '@dtour/viewer';
import { Dtour } from '@dtour/viewer';
import { GithubLogo, PlayCircle, SpinnerIcon, UploadSimple, X } from '@phosphor-icons/react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import 'plyr-react/plyr.css';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatedLogo } from './components/AnimatedLogo.tsx';
import { Button } from './components/ui/button.tsx';
import { CONTENT_TOP_VH } from './constants.ts';
import CsvWorkerFactory from './workers/csv.worker.ts?worker&inline';

type LogoPhase = 'drawing' | 'moving' | 'done';
type ThemeMode = 'light' | 'dark' | 'system';

const ACCEPTED_EXTENSIONS = ['.parquet', '.pq', '.arrow', '.csv'];

const REMOTE = import.meta.env.DEV ? '/cloudflare' : 'https://data.dtour.dev';

type ExampleDataset = {
  label: string;
  fileName: string;
  numPoints: string;
  numDims: string;
  size?: string;
  tourDescription?: string;
  // Default color encoding for examples whose data can't embed a dtour spec
  // (e.g. generated Arrow tables). Applied on first load; overridden by any
  // persisted user choice.
  pointColorBy?: string;
} & (
  | { type: 'remote'; url: string }
  | { type: 'generate'; worker: 'lorenz' | 'gaussian-blobs' | 'linked-rings' }
);

const EXAMPLES: ExampleDataset[] = [
  {
    type: 'generate',
    worker: 'gaussian-blobs',
    label: 'Gaussian Blobs',
    fileName: 'gaussian-blobs-5d.arrow',
    numPoints: '500K',
    numDims: '5',
    pointColorBy: 'cluster',
    tourDescription:
      'Five Gaussian clusters in 5D. The tour rotates through projections that reveal how the groups separate and overlap from different angles.',
  },
  {
    type: 'generate',
    worker: 'linked-rings',
    label: 'Linked Rings',
    fileName: 'linked-rings-4d.arrow',
    numPoints: '500K',
    numDims: '4',
    pointColorBy: 'ring',
    tourDescription:
      'Two interlocking rings in 4D. The tour reveals the linked topology that is invisible in any single 2D projection.',
  },
  {
    type: 'generate',
    worker: 'lorenz',
    label: 'Lorenz Attractor',
    fileName: 'lorenz-stenflo-1m.arrow',
    numPoints: '1M',
    numDims: '4',
    tourDescription:
      'The 4D [Lorenz-Stenflo chaotic attractor](https://simple.wikipedia.org/wiki/Lorenz_attractor). The tour traces its butterfly-shaped structure from multiple projection angles.',
  },
  {
    type: 'remote',
    label: 'Fashion MNIST',
    fileName: 'fashion-mnist-attraction-repulsion-tour.pq',
    url: `${REMOTE}/fashion-mnist-attraction-repulsion-tour.pq`,
    numPoints: '70K',
    numDims: '8',
    size: '3MB',
    tourDescription:
      '70K fashion product images from [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist). Four 2D embeddings sweep the attraction-repulsion spectrum from LE-like (rho=100) through UMAP-like (rho=4) to t-SNE (rho=1), revealing how cluster structure emerges as repulsion increases.',
  },
  {
    type: 'remote',
    label: 'News Headlines',
    fileName: 'huffpost-news-embeddings-umap-dense-supervised-4d.pq',
    url: `${REMOTE}/huffpost-news-embeddings-umap-dense-supervised-4d.pq`,
    numPoints: '204K',
    numDims: '4',
    size: '5MB',
    tourDescription:
      '[200K+ HuffPost news headlines](https://www.kaggle.com/datasets/rmisra/news-category-dataset) in a 4D supervised UMAP embedding. The tour rotates through projections, revealing topic clusters and the continuum between news categories.',
  },
  {
    type: 'remote',
    label: 'Single Cell Proteomics',
    fileName: 'mair-2022-tumor-le-fisher-tour-markers.pq',
    url: `${REMOTE}/mair-2022-tumor-le-fisher-tour-markers.pq`,
    numPoints: '345K',
    numDims: '9',
    size: '34MB',
    tourDescription:
      '345K tumor immune cells across 9 protein markers from [Mair et al. (2022)](https://doi.org/10.1038/s41586-022-04718-w). A Fisher-discriminant tour through spectral LE bases reveals immunophenotypic subpopulations at increasing differentiation levels.',
  },
  {
    type: 'remote',
    label: 'Single Cell RNA-seq',
    fileName: 'lamanno2021-pca-tour.pq',
    url: `${REMOTE}/lamanno2021-pca-tour.pq`,
    numPoints: '276K',
    numDims: '8',
    size: '8MB',
    tourDescription:
      '276K developing mouse brain cells from [La Manno et al. (2021)](https://doi.org/10.1038/s41586-021-03775-x). A PCA tour through the first 8 principal components, showing the transcriptomic diversity of neural cell types.',
  },
  {
    type: 'remote',
    label: 'Image Caption CLIP',
    fileName: 'sharegpt4v-coco-clip-joint-embeddings-umap-dense-2d-all-alphas-tour.pq',
    url: `${REMOTE}/sharegpt4v-coco-clip-joint-embeddings-umap-dense-2d-all-alphas-tour.pq`,
    numPoints: '49K',
    numDims: '10',
    size: '5MB',
    tourDescription:
      '49K image-caption pairs from [ShareGPT4V](https://sharegpt4v.github.io/) as joint CLIP embeddings. Five 2D embeddings sweep from pure caption to pure pixel representation, revealing how visual and textual semantics align across modalities.',
  },
  {
    type: 'remote',
    label: 'arXiv papers',
    fileName: 'arxiv-sequential-embedding-model-tour.pq',
    url: `${REMOTE}/arxiv-sequential-embedding-model-tour.pq`,
    numPoints: '3M',
    numDims: '8',
    size: '115MB',
    tourDescription:
      '3M [arXiv](https://arxiv.org/) papers. Four 2D UMAP embeddings of the titles and abstracts from [SPECTER2](https://huggingface.co/allenai/specter2), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [Nomic v2](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe), and [F2LLM-v2 8B](https://huggingface.co/codefuse-ai/F2LLM-v2-8B) embeddingsreveal how different models organize the scientific research landscape.',
  },
];

const DATASET_SLUGS: Record<string, number> = {
  'fashion-mnist': 0,
  'news-headlines': 1,
  'single-cell': 2,
  lorenz: 3,
  'gaussian-blobs': 4,
  'linked-rings': 5,
};

const THEME_STORAGE_KEY = 'dtour-theme-mode';
const SPEC_STORAGE_PREFIX = 'dtour-spec:';

function readPersistedTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {}
  return 'dark';
}

// Default spec for examples whose data format can't embed a dtour config
// (generated Arrow tables). Matched by exact fileName or, for generated
// examples loaded with a custom point count, by the `<worker>-` prefix.
function exampleDefaultSpec(fileName: string | undefined): DtourSpec {
  if (!fileName) return {};
  for (const example of EXAMPLES) {
    if (!example.pointColorBy) continue;
    const matches =
      example.fileName === fileName ||
      (example.type === 'generate' && fileName.startsWith(`${example.worker}-`));
    if (matches) return { pointColorBy: example.pointColorBy };
  }
  return {};
}

function loadPersistedSpec(fileName: string): DtourSpec {
  try {
    const raw = localStorage.getItem(SPEC_STORAGE_PREFIX + fileName);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return {};
    // Migrate: old specs stored showTourDescription as false; now null means "auto"
    if (parsed.showTourDescription === false) parsed.showTourDescription = null;
    return parsed as DtourSpec;
  } catch {
    return {};
  }
}

function savePersistedSpec(fileName: string, spec: Required<DtourSpec>): void {
  try {
    localStorage.setItem(SPEC_STORAGE_PREFIX + fileName, JSON.stringify(spec));
  } catch {}
}

function csvToArrow(csvBuffer: ArrayBuffer): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const worker = new CsvWorkerFactory();
    worker.onmessage = (e: MessageEvent<ArrayBuffer | { error: string }>) => {
      if (e.data instanceof ArrayBuffer) {
        resolve(e.data);
      } else {
        reject(new Error(e.data.error));
      }
      worker.terminate();
    };
    worker.onerror = (e: ErrorEvent) => {
      reject(new Error(e.message));
      worker.terminate();
    };
    worker.postMessage(csvBuffer, [csvBuffer]);
  });
}

// URL parameters for benchmark automation
const urlParams = new URLSearchParams(globalThis.location?.search ?? '');
const benchmarkMode = urlParams.has('benchmark');
const datasetSlug = urlParams.get('dataset');
const urlParam = urlParams.get('url');
const rendererQuery = urlParams.get('renderer');
const rendererParam =
  rendererQuery === 'webgl' ? 'webgl' : rendererQuery === 'webgpu' ? 'webgpu' : 'auto';
const pointsParam = Number(urlParams.get('points')) || null;

// In benchmark mode, expose flag so DtourViewer conditionally sets window.scatter
if (benchmarkMode) {
  (globalThis as Record<string, unknown>).__dtourBenchmarkMode = true;
}

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);
  const [fileName, setFileName] = useState<string | undefined>(undefined);
  const [tourDescription, setTourDescription] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [homeOpen, setHomeOpen] = useState(false);
  const [videoOpen, setVideoOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const plyrRef = useRef<InstanceType<typeof import('plyr').default> | null>(null);

  const prefersReducedMotion = useReducedMotion();
  const [logoPhase, setLogoPhase] = useState<LogoPhase>(
    prefersReducedMotion || benchmarkMode ? 'done' : 'drawing',
  );
  const drawCompleteRef = useRef(false);
  const gpuReadyRef = useRef(false);
  const logoPhaseRef = useRef(logoPhase);
  logoPhaseRef.current = logoPhase;
  const loadIdRef = useRef(0);

  // Theme: persisted globally in localStorage, synced from Dtour via onSpecChange
  const [themeMode, setThemeMode] = useState<ThemeMode>(readPersistedTheme);
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  );

  // Derive spec from fileName — recomputed synchronously when fileName changes.
  // Combined with key={fileName} on <Dtour>, this guarantees initStoreFromSpec
  // runs with the persisted spec before the first render (no flash of defaults).
  const spec = useMemo<DtourSpec>(() => {
    const persisted = fileName ? loadPersistedSpec(fileName) : {};
    return { ...exampleDefaultSpec(fileName), ...persisted, themeMode: readPersistedTheme() };
  }, [fileName]);

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setSystemTheme(e.matches ? 'dark' : 'light');
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  const resolvedTheme = themeMode === 'system' ? systemTheme : themeMode;

  // Close home/video modal on Escape
  useEffect(() => {
    if (!homeOpen && !videoOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setHomeOpen(false);
        setVideoOpen(false);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [homeOpen, videoOpen]);

  const handleSpecChange = useCallback(
    (newSpec: Required<DtourSpec>) => {
      setThemeMode(newSpec.themeMode);
      try {
        localStorage.setItem(THEME_STORAGE_KEY, newSpec.themeMode);
      } catch {}
      if (fileName) {
        savePersistedSpec(fileName, newSpec);
      }
    },
    [fileName],
  );

  const loadFile = useCallback(async (file: File) => {
    setHomeOpen(false);
    const id = ++loadIdRef.current;
    const isCsv = file.name.toLowerCase().endsWith('.csv');
    if (isCsv) setLoading(true);

    try {
      let buffer = await file.arrayBuffer();
      if (id !== loadIdRef.current) return;

      if (isCsv) {
        buffer = await csvToArrow(buffer);
        if (id !== loadIdRef.current) return;
      }

      metadataReceivedRef.current = false;
      gpuReadyRef.current = false;
      setTourDescription(undefined);
      setFileName(file.name);
      setData(buffer);
      setParsing(true);
    } catch (err) {
      if (id !== loadIdRef.current) return;
      console.error('Failed to load file:', err);
    } finally {
      if (isCsv && id === loadIdRef.current) setLoading(false);
    }
  }, []);

  const handleLoadData = useCallback((buffer: ArrayBuffer, name: string) => {
    const applyData = (b: ArrayBuffer) => {
      metadataReceivedRef.current = false;
      gpuReadyRef.current = false;
      setTourDescription(undefined);
      setFileName(name);
      setData(b);
      setParsing(true);
    };

    if (name.toLowerCase().endsWith('.csv')) {
      setLoading(true);
      const id = ++loadIdRef.current;
      csvToArrow(buffer)
        .then((arrowBuffer) => {
          if (id !== loadIdRef.current) return;
          applyData(arrowBuffer);
        })
        .catch((err) => {
          if (id !== loadIdRef.current) return;
          console.error('Failed to parse CSV:', err);
        })
        .finally(() => {
          if (id === loadIdRef.current) setLoading(false);
        });
      return;
    }

    applyData(buffer);
  }, []);

  const loadExample = useCallback(
    async (example: ExampleDataset) => {
      if (loading) return;
      setLoading(true);
      const id = ++loadIdRef.current;

      try {
        let buffer: ArrayBuffer;

        if (example.type === 'remote') {
          const response = await fetch(example.url);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          buffer = await response.arrayBuffer();
        } else {
          const mod = await (example.worker === 'gaussian-blobs'
            ? import('./workers/gaussianBlobs.worker.ts?worker&inline')
            : example.worker === 'linked-rings'
              ? import('./workers/linkedRings.worker.ts?worker&inline')
              : import('./workers/lorenz.worker.ts?worker&inline'));
          const WorkerFactory = mod.default;

          buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
            const worker = new WorkerFactory();
            worker.onmessage = (e: MessageEvent<ArrayBuffer>) => {
              resolve(e.data);
              worker.terminate();
            };
            worker.onerror = (e: ErrorEvent) => {
              reject(new Error(e.message));
              worker.terminate();
            };
            worker.postMessage(pointsParam);
          });
        }

        if (id !== loadIdRef.current) return;

        // When generating with a custom point count, use a distinct filename
        // so the spec cache doesn't collide between sizes.
        const effectiveName =
          example.type === 'generate' && pointsParam
            ? `${example.worker}-${pointsParam}.arrow`
            : example.fileName;

        metadataReceivedRef.current = false;
        gpuReadyRef.current = false;
        setTourDescription(example.tourDescription);
        setFileName(effectiveName);
        setData(buffer);
        setParsing(true);
      } catch (err) {
        if (id !== loadIdRef.current) return;
        console.error('Failed to load example:', err);
      } finally {
        if (id === loadIdRef.current) {
          setLoading(false);
        }
      }
    },
    [loading],
  );

  // Auto-load dataset from URL parameter (for benchmark automation).
  // Deferred until the logo draw completes so the worker doesn't compete
  // with the animation for main-thread resources.
  const loadExampleRef = useRef(loadExample);
  loadExampleRef.current = loadExample;
  const handleLoadDataRef = useRef(handleLoadData);
  handleLoadDataRef.current = handleLoadData;
  const pendingAutoLoad = useRef(true);
  useEffect(() => {
    if (!pendingAutoLoad.current) return;
    if (logoPhase === 'drawing') return; // wait for draw to finish
    pendingAutoLoad.current = false;

    if (urlParam) {
      setLoading(true);
      fetch(urlParam)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.arrayBuffer();
        })
        .then((buffer) => {
          const name = urlParam.split('/').pop() || 'data.pq';
          handleLoadDataRef.current(buffer, name);
        })
        .catch((err) => console.error('Failed to load URL:', err))
        .finally(() => setLoading(false));
      return;
    }
    if (!datasetSlug) return;
    const index = DATASET_SLUGS[datasetSlug];
    if (index === undefined) {
      console.warn(
        `Unknown dataset slug: "${datasetSlug}". Valid: ${Object.keys(DATASET_SLUGS).join(', ')}`,
      );
      return;
    }
    loadExampleRef.current(EXAMPLES[index]!);
  }, [logoPhase]);

  // Expose readiness signal for Playwright.
  // We wait for the first 'rendered' event (not just 'metadata'), because bases
  // are installed in a later React effect and benchmark() requires state.tour.
  const metadataReceivedRef = useRef(false);
  const handleStatus = useCallback((status: { type: string }) => {
    if (status.type === 'metadata') {
      metadataReceivedRef.current = true;
    }
    if (status.type === 'error') {
      setParsing(false);
    }
    if (status.type === 'rendered' && metadataReceivedRef.current) {
      (globalThis as Record<string, unknown>).__dtourReady = true;
      gpuReadyRef.current = true;
      // If logo is still animating, trigger move once draw is also complete.
      // If logo is already done (subsequent loads), just clear parsing.
      if (logoPhaseRef.current === 'done') {
        setParsing(false);
      } else if (drawCompleteRef.current) {
        setLogoPhase('moving');
      }
    }
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (!file) return;
      const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
      if (!ACCEPTED_EXTENSIONS.includes(ext)) return;
      loadFile(file);
    },
    [loadFile],
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) loadFile(file);
      // Reset so re-selecting the same file triggers onChange again
      e.target.value = '';
    },
    [loadFile],
  );

  const handleDrawComplete = useCallback(() => {
    drawCompleteRef.current = true;
    if (gpuReadyRef.current) {
      setLogoPhase('moving');
    }
  }, []);

  const handleMoveComplete = useCallback(() => {
    // Step 2: logo has landed → show toolbar (fade in via hideToolbar=false)
    setLogoPhase('done');
    // Step 3+4: brief pause for browser to settle, then reveal scatter
    setTimeout(() => {
      setParsing(false);
    }, 350);
  }, []);

  return (
    <div
      className={`flex flex-col w-screen h-screen ${resolvedTheme === 'light' ? 'dtour-light' : ''}`}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".parquet,.pq,.arrow,.csv"
        className="hidden"
        onChange={handleFileSelect}
      />
      <Dtour
        key={fileName}
        data={data}
        spec={spec}
        tourDescription={tourDescription}
        onLoadData={handleLoadData}
        onLogoClick={() => setHomeOpen(true)}
        onSpecChange={handleSpecChange}
        onStatus={handleStatus}
        hideToolbar={logoPhase !== 'done'}
        backend={rendererParam}
      />
      {/* Solid background cover — stays during logo move to hide scatter */}
      {(!data || parsing) && (
        <div className="absolute inset-0 z-20 bg-dtour-bg pointer-events-none" />
      )}
      {/* Content overlay (upload button + examples) — hidden during logo move/settle */}
      {(!data || (parsing && logoPhase !== 'done')) && logoPhase !== 'moving' && (
        <motion.div
          className={`absolute inset-0 flex flex-col items-center px-4 z-20 pointer-events-none ${
            logoPhase === 'done' ? 'justify-center' : 'justify-start'
          }`}
          style={logoPhase !== 'done' ? { paddingTop: `${CONTENT_TOP_VH * 100}vh` } : undefined}
        >
          {loading || parsing ? (
            <div className="flex flex-col items-center gap-3 px-6 py-4">
              <SpinnerIcon size={32} className="animate-spin text-dtour-text-muted" />
            </div>
          ) : (
            <>
              {/* 1. Description */}
              <motion.p
                className="w-full max-w-lg text-sm text-dtour-text-muted/80 text-center mb-6 pointer-events-auto"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  delay: logoPhase === 'drawing' ? 1.2 : 0,
                  duration: 0.6,
                  ease: 'easeOut',
                }}
              >
                Visually explore high-dimensional data and embeddings through interactive, smooth
                tours of 2D projections to build intuition for the full data manifold.
              </motion.p>
              {/* 2. Drop button */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  delay: logoPhase === 'drawing' ? 1.45 : 0,
                  duration: 0.4,
                  ease: 'easeOut',
                }}
              >
                <Button
                  variant="ghost"
                  className="w-full max-w-lg cursor-pointer flex flex-col items-center gap-2 p-4 h-auto pointer-events-auto bg-dtour-surface/60 hover:bg-dtour-surface"
                  onClick={() => inputRef.current?.click()}
                >
                  <UploadSimple size={36} />
                  <span className="text-sm select-none">
                    Drop a Parquet, Arrow, or CSV file to start
                  </span>
                </Button>
              </motion.div>
              {/* 3. "or try" */}
              <motion.span
                className="text-xs text-dtour-text-muted/60 select-none mt-4"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  delay: logoPhase === 'drawing' ? 1.5 : 0,
                  duration: 0.4,
                  ease: 'easeOut',
                }}
              >
                or try
              </motion.span>
              {/* 4. Example grid */}
              <div className="grid grid-cols-3 gap-1.5 sm:gap-4 mt-3 w-full max-w-lg pointer-events-auto">
                {EXAMPLES.map((example, i) => (
                  <motion.button
                    key={example.fileName}
                    type="button"
                    className="w-full p-2 border border-dtour-surface rounded-md text-left cursor-pointer transition-colors bg-dtour-bg/50 hover:bg-dtour-surface select-none backdrop-blur-sm"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{
                      delay: (logoPhase === 'drawing' ? 1.55 : 0) + i * 0.05,
                      duration: 0.4,
                      ease: 'easeOut',
                    }}
                    onClick={() => loadExample(example)}
                  >
                    <span className="block text-xs text-dtour-text/70 truncate">
                      {example.label}
                    </span>
                    <span className="flex justify-between text-[10px] text-dtour-text-muted/50 mt-1">
                      <span>
                        {example.numPoints} &times; {example.numDims}D
                      </span>
                      {example.size && <span>{example.size}</span>}
                    </span>
                  </motion.button>
                ))}
              </div>
              {/* 5. "or watch" */}
              <motion.span
                className="text-xs text-dtour-text-muted/60 select-none mt-4 mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  delay: (logoPhase === 'drawing' ? 1.55 : 0) + EXAMPLES.length * 0.05 + 0.1,
                  duration: 0.4,
                  ease: 'easeOut',
                }}
              >
                or watch
              </motion.span>
              {/* 6. Video button */}
              <motion.button
                type="button"
                className="flex items-center gap-1.5 text-xs text-dtour-text-muted/80 hover:text-dtour-text border border-dtour-surface hover:bg-dtour-surface rounded-full px-3 py-1.5 transition-colors cursor-pointer select-none mt-2 pointer-events-auto"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  delay: (logoPhase === 'drawing' ? 1.55 : 0) + EXAMPLES.length * 0.05 + 0.15,
                  duration: 0.4,
                  ease: 'easeOut',
                }}
                onClick={() => setVideoOpen(true)}
              >
                <PlayCircle size={16} weight="fill" />
                <span>Intro Video</span>
              </motion.button>
            </>
          )}
          <p className="absolute bottom-6 text-xs text-dtour-text-muted/60 flex items-center gap-1">
            <span>
              Made by{' '}
              <a
                href="https://lekschas.de"
                target="_blank"
                rel="noopener noreferrer"
                className="pointer-events-auto hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
              >
                Fritz
              </a>
              {' and '}
              <a
                href="https://nvictus.me"
                target="_blank"
                rel="noopener noreferrer"
                className="pointer-events-auto hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
              >
                Nezar
              </a>
              .
            </span>
            <span className="bg-dtour-text-muted/30 w-px h-4 mx-1" />
            <a
              href="https://github.com/flekschas/dtour"
              target="_blank"
              rel="noopener noreferrer"
              className="pointer-events-auto flex items-center gap-1 hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
            >
              <GithubLogo size={14} weight="fill" />
              GitHub
            </a>
          </p>
        </motion.div>
      )}
      {logoPhase !== 'done' && (
        <AnimatedLogo
          phase={logoPhase}
          theme={resolvedTheme}
          onDrawComplete={handleDrawComplete}
          onMoveComplete={handleMoveComplete}
        />
      )}
      <AnimatePresence>
        {homeOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div
              role="presentation"
              className="absolute inset-0 bg-black/70 backdrop-blur-sm"
              onClick={() => setHomeOpen(false)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setHomeOpen(false);
              }}
            />
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center px-4 pointer-events-none">
              {loading ? (
                <div className="flex flex-col items-center gap-3 px-6 py-4">
                  <SpinnerIcon size={32} className="animate-spin text-dtour-text-muted" />
                </div>
              ) : (
                <>
                  <Button
                    variant="ghost"
                    className="w-full max-w-lg cursor-pointer flex flex-col items-center gap-2 p-4 h-auto pointer-events-auto bg-dtour-surface/60 hover:bg-dtour-surface backdrop-blur-sm"
                    onClick={() => inputRef.current?.click()}
                  >
                    <UploadSimple size={36} />
                    <span className="text-sm select-none">
                      Drop a Parquet, Arrow, or CSV file to start
                    </span>
                  </Button>
                  <span className="text-xs text-dtour-text-muted/60 select-none mt-4">or try</span>
                  <div className="grid grid-cols-3 gap-1.5 sm:gap-4 mt-3 w-full max-w-lg pointer-events-auto">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example.fileName}
                        type="button"
                        className="w-full p-2 border border-dtour-surface rounded-md text-left cursor-pointer transition-colors bg-dtour-surface/50 hover:bg-dtour-surface select-none backdrop-blur-sm"
                        onClick={() => {
                          setHomeOpen(false);
                          loadExample(example);
                        }}
                      >
                        <span className="block text-xs text-dtour-text/70 truncate">
                          {example.label}
                        </span>
                        <span className="flex justify-between text-[10px] text-dtour-text-muted/50 mt-1">
                          <span>
                            {example.numPoints} &times; {example.numDims}D
                          </span>
                          {example.size && <span>{example.size}</span>}
                        </span>
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center gap-2 mt-4 pointer-events-auto">
                    <span className="text-xs text-dtour-text-muted/60 select-none">or watch</span>
                  </div>
                  <div className="flex items-center gap-2 mt-4 pointer-events-auto">
                    <button
                      type="button"
                      className="flex items-center gap-1.5 text-xs text-dtour-text-muted/80 hover:text-dtour-text border border-dtour-surface hover:bg-dtour-surface rounded-full px-3 py-1.5 transition-colors cursor-pointer select-none"
                      onClick={() => {
                        setHomeOpen(false);
                        setVideoOpen(true);
                      }}
                    >
                      <PlayCircle size={16} weight="fill" />
                      <span>Intro Video</span>
                    </button>
                  </div>
                </>
              )}
              <p className="absolute bottom-6 text-xs text-dtour-text-muted/60 flex items-center gap-1">
                <span>
                  Made by{' '}
                  <a
                    href="https://lekschas.de"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="pointer-events-auto hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
                  >
                    Fritz
                  </a>
                  {' and '}
                  <a
                    href="https://nvictus.me"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="pointer-events-auto hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
                  >
                    Nezar
                  </a>
                  .
                </span>
                <span className="text-dtour-text-muted/30 mx-1">|</span>
                <a
                  href="https://github.com/flekschas/dtour"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="pointer-events-auto flex items-center gap-1 hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
                >
                  <GithubLogo size={14} weight="fill" />
                  GitHub
                </a>
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      {/* Video intro modal */}
      <AnimatePresence>
        {videoOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
          >
            <div
              role="presentation"
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
              onClick={() => setVideoOpen(false)}
              onKeyDown={(e) => {
                if (e.key === 'Escape') setVideoOpen(false);
              }}
            />
            <div className="relative z-10 w-full max-w-4xl mx-4">
              <button
                type="button"
                className="absolute -top-10 right-0 text-dtour-text-muted hover:text-dtour-text transition-colors cursor-pointer"
                onClick={() => setVideoOpen(false)}
              >
                <X size={24} />
              </button>
              <div
                className="rounded-lg overflow-hidden"
                ref={(el) => {
                  if (!el) {
                    plyrRef.current?.destroy();
                    plyrRef.current = null;
                    return;
                  }
                  const video = el.querySelector('video');
                  if (!video || plyrRef.current) return;
                  import('plyr').then(({ default: PlyrLib }) => {
                    plyrRef.current = new PlyrLib(video, {
                      markers: {
                        enabled: true,
                        points: [
                          { time: 51, label: 'Repulsion tour' },
                          { time: 113, label: 'Structure tour' },
                          { time: 153, label: 'PCA–UMAP tour' },
                          { time: 211, label: 'Embedding tour' },
                        ],
                      },
                    });
                  });
                }}
              >
                <video
                  ref={videoRef}
                  playsInline
                  poster="https://data.dtour.dev/dtour-teaser-poster.webp"
                >
                  <source src="https://data.dtour.dev/dtour-teaser.mp4" type="video/mp4" />
                  <track kind="captions" />
                </video>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
