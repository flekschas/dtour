import { Dtour } from 'dtour';
import type { DtourSpec } from 'dtour';
import { SpinnerIcon } from '@phosphor-icons/react';
import { motion, useReducedMotion } from 'motion/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatedLogo } from './components/AnimatedLogo.tsx';
import { Button } from './components/ui/button.tsx';
import LorenzWorkerFactory from './workers/lorenz.worker.ts?worker&inline';

type LogoPhase = 'drawing' | 'moving' | 'moved' | 'done';
type ThemeMode = 'light' | 'dark' | 'system';

const ACCEPTED_EXTENSIONS = ['.parquet', '.pq', '.arrow'];

const GCS_BASE = import.meta.env.DEV
  ? '/gcs/dtour'
  : 'https://storage.googleapis.com/dtour';

type ExampleDataset =
	| { label: string; fileName: string; type: 'remote'; url: string }
	| { label: string; fileName: string; type: 'generate' };

const EXAMPLES: ExampleDataset[] = [
	{
		type: 'remote',
		label: 'Fashion MNIST',
		fileName: 'fashion-mnist-embeddings-umap-dense-supervised-4d.pq',
		url: `${GCS_BASE}/fashion-mnist-embeddings-umap-dense-supervised-4d.pq`,
	},
	{
		type: 'remote',
		label: 'News Headlines',
		fileName: 'huffpost-news-embeddings-umap-dense-supervised-4d.pq',
		url: `${GCS_BASE}/huffpost-news-embeddings-umap-dense-supervised-4d.pq`,
	},
	{
		type: 'remote',
		label: 'Single Cell',
		fileName: 'mair-2022-tumor-006-ozette-umap-4d.pq',
		url: `${GCS_BASE}/mair-2022-tumor-006-ozette-umap-4d.pq`,
	},
	{
		type: 'generate',
		label: 'Lorenz Attractor',
		fileName: 'lorenz-stenflo-1m.arrow',
	},
];

const THEME_STORAGE_KEY = 'dtour-theme-mode';
const SPEC_STORAGE_PREFIX = 'dtour-spec:';

function readPersistedTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {}
  return 'dark';
}

function loadPersistedSpec(fileName: string): DtourSpec {
  try {
    const raw = localStorage.getItem(SPEC_STORAGE_PREFIX + fileName);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return {};
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

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);
  const [fileName, setFileName] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const prefersReducedMotion = useReducedMotion();
  const [logoPhase, setLogoPhase] = useState<LogoPhase>(prefersReducedMotion ? 'done' : 'drawing');
  const pendingDataRef = useRef<ArrayBuffer | null>(null);
  const pendingNameRef = useRef<string | null>(null);
  const drawCompleteRef = useRef(false);
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
    return { ...persisted, themeMode: readPersistedTheme() };
  }, [fileName]);

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setSystemTheme(e.matches ? 'dark' : 'light');
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  const resolvedTheme = themeMode === 'system' ? systemTheme : themeMode;

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

  const loadFile = useCallback(
    async (file: File) => {
      const id = ++loadIdRef.current;
      const buffer = await file.arrayBuffer();
      if (id !== loadIdRef.current) return;
      if (logoPhase === 'done') {
        setFileName(file.name);
        setData(buffer);
      } else {
        pendingDataRef.current = buffer;
        pendingNameRef.current = file.name;
        if (drawCompleteRef.current) {
          setLogoPhase('moving');
        }
      }
    },
    [logoPhase],
  );

  const handleLoadData = useCallback(
    (buffer: ArrayBuffer, name: string) => {
      if (logoPhase === 'done') {
        setFileName(name);
        setData(buffer);
      } else {
        pendingDataRef.current = buffer;
        pendingNameRef.current = name;
        if (drawCompleteRef.current) {
          setLogoPhase('moving');
        }
      }
    },
    [logoPhase],
  );

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
          buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
            const worker = new LorenzWorkerFactory();
            worker.onmessage = (e: MessageEvent<ArrayBuffer>) => {
              resolve(e.data);
              worker.terminate();
            };
            worker.onerror = (e: ErrorEvent) => {
              reject(new Error(e.message));
              worker.terminate();
            };
            worker.postMessage(null);
          });
        }

        if (id !== loadIdRef.current) return;

        if (logoPhase === 'done') {
          setFileName(example.fileName);
          setData(buffer);
        } else {
          pendingDataRef.current = buffer;
          pendingNameRef.current = example.fileName;
          if (drawCompleteRef.current) {
            setLogoPhase('moving');
          }
        }
      } catch (err) {
        if (id !== loadIdRef.current) return;
        console.error('Failed to load example:', err);
      } finally {
        if (id === loadIdRef.current) {
          setLoading(false);
        }
      }
    },
    [loading, logoPhase],
  );

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
    if (pendingDataRef.current) {
      setLogoPhase('moving');
    }
  }, []);

  const handleMoveComplete = useCallback(() => {
    setLogoPhase('moved');
    // Delay data loading so toolbar can fade in first
    setTimeout(() => {
      if (pendingDataRef.current) {
        setFileName(pendingNameRef.current ?? undefined);
        setData(pendingDataRef.current);
        pendingDataRef.current = null;
        pendingNameRef.current = null;
      }
      setLogoPhase('done');
    }, 300);
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
        accept=".parquet,.pq,.arrow"
        className="hidden"
        onChange={handleFileSelect}
      />
      <Dtour
        key={fileName}
        data={data}
        spec={spec}
        onLoadData={handleLoadData}
        onSpecChange={handleSpecChange}
        hideToolbar={logoPhase === 'drawing' || logoPhase === 'moving'}
      />
      {!data && logoPhase !== 'moving' && logoPhase !== 'moved' && (
        <motion.div
          className={`absolute inset-0 flex flex-col items-center z-20 pointer-events-none ${
            logoPhase !== 'done' ? 'justify-end pb-[40vh]' : 'justify-center'
          }`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            delay: logoPhase === 'drawing' ? 0.75 : 0,
            duration: 0.5,
            ease: 'easeOut',
          }}
        >
          {loading ? (
            <div className="flex flex-col items-center gap-3 px-6 py-4">
              <SpinnerIcon size={32} className="animate-spin text-dtour-text-muted" />
            </div>
          ) : (
            <>
              <Button
                variant="ghost"
                className="cursor-pointer flex flex-col items-center gap-3 px-6 py-4 h-auto pointer-events-auto bg-dtour-surface/60 hover:bg-dtour-surface"
                onClick={() => inputRef.current?.click()}
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
                <span className="text-sm select-none">Drop a Parquet or Arrow file to start</span>
              </Button>
              <span className="text-xs text-dtour-text-muted/60 select-none mt-4">or try</span>
              <div className="flex items-center gap-1 mt-3 pointer-events-auto">
                {EXAMPLES.map((example, i) => (
                  <motion.button
                    key={example.fileName}
                    type="button"
                    className="text-xs text-dtour-text-muted hover:text-dtour-highlight hover:underline underline-offset-2 cursor-pointer transition-colors px-1 py-0.5 select-none"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{
                      delay: (logoPhase === 'drawing' ? 0.75 : 0) + 0.15 + i * 0.05,
                      duration: 0.4,
                      ease: 'easeOut',
                    }}
                    onClick={() => loadExample(example)}
                  >
                    {example.label}
                  </motion.button>
                ))}
              </div>
            </>
          )}
          <p className="absolute bottom-6 text-xs text-dtour-text-muted/60">
            Explore high-dimensional data through guided, manual, and grand tours of 2D projections.
            <a
              href="https://github.com/flekschas/dtour"
              target="_blank"
              rel="noopener noreferrer"
              className="pointer-events-auto ml-1 hover:underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
            >
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
    </div>
  );
};

export default App;
