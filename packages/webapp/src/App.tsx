import { Dtour } from '@dtour/viewer';
import type { DtourSpec } from '@dtour/viewer';
import { motion, useReducedMotion } from 'motion/react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatedLogo } from './components/AnimatedLogo.tsx';
import { Button } from './components/ui/button.tsx';

type LogoPhase = 'drawing' | 'moving' | 'moved' | 'done';
type ThemeMode = 'light' | 'dark' | 'system';

const THEME_STORAGE_KEY = 'dtour-theme-mode';

function readPersistedTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    if (v === 'light' || v === 'dark' || v === 'system') return v;
  } catch {}
  return 'dark';
}

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);
  const [fileName, setFileName] = useState<string | undefined>(undefined);
  const inputRef = useRef<HTMLInputElement>(null);

  const prefersReducedMotion = useReducedMotion();
  const [logoPhase, setLogoPhase] = useState<LogoPhase>(prefersReducedMotion ? 'done' : 'drawing');
  const pendingDataRef = useRef<{ buffer: ArrayBuffer; name: string } | null>(null);
  const drawCompleteRef = useRef(false);

  // Theme: persisted globally in localStorage, synced from Dtour via onSpecChange
  const [themeMode, setThemeMode] = useState<ThemeMode>(readPersistedTheme);
  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(() =>
    window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
  );
  // Stable initial spec — pass persisted theme to Dtour on mount
  const [initialSpec] = useState<DtourSpec>(() => ({ themeMode: readPersistedTheme() }));

  useEffect(() => {
    const mql = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setSystemTheme(e.matches ? 'dark' : 'light');
    mql.addEventListener('change', handler);
    return () => mql.removeEventListener('change', handler);
  }, []);

  const resolvedTheme = themeMode === 'system' ? systemTheme : themeMode;

  const handleSpecChange = useCallback((spec: Required<DtourSpec>) => {
    setThemeMode(spec.themeMode);
    try { localStorage.setItem(THEME_STORAGE_KEY, spec.themeMode); } catch {}
  }, []);

  const loadFile = useCallback(
    async (file: File) => {
      const buffer = await file.arrayBuffer();
      const name = file.name;
      if (logoPhase === 'done') {
        setData(buffer);
        setFileName(name);
      } else {
        pendingDataRef.current = { buffer, name };
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
        setData(buffer);
        setFileName(name);
      } else {
        pendingDataRef.current = { buffer, name };
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
      if (file) loadFile(file);
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
        setData(pendingDataRef.current.buffer);
        setFileName(pendingDataRef.current.name);
        pendingDataRef.current = null;
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
        data={data}
        dataName={fileName}
        spec={initialSpec}
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
