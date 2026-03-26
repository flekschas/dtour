import { Dtour } from '@dtour/viewer';
import type { DtourSpec } from '@dtour/viewer';
import { motion, useReducedMotion } from 'motion/react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatedLogo } from './components/AnimatedLogo.tsx';
import { Button } from './components/ui/button.tsx';

type LogoPhase = 'drawing' | 'moving' | 'moved' | 'done';
type ThemeMode = 'light' | 'dark' | 'system';

const ACCEPTED_EXTENSIONS = ['.parquet', '.pq', '.arrow'];

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
          <Button
            variant="ghost"
            className="cursor-pointer flex flex-col items-center gap-3 px-6 py-4 h-auto pointer-events-auto"
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
          <p className="absolute bottom-6 text-xs text-dtour-text-muted/60 select-none">
            Explore high-dimensional data through guided, manual, and grand tours of 2D projections.
            <a
              href="https://github.com/flekschas/dtour"
              target="_blank"
              rel="noopener noreferrer"
              className="pointer-events-auto ml-1 underline underline-offset-2 hover:text-dtour-text-muted transition-colors"
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
