import { Dtour } from '@dtour/viewer';
import { useCallback, useRef, useState } from 'react';
import { Button } from './components/ui/button.tsx';

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);
  const [fileName, setFileName] = useState<string | undefined>(undefined);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = useCallback(async (file: File) => {
    setData(await file.arrayBuffer());
    setFileName(file.name);
  }, []);

  const handleLoadData = useCallback((buffer: ArrayBuffer, name: string) => {
    setData(buffer);
    setFileName(name);
  }, []);

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

  return (
    <div
      className="flex flex-col w-screen h-screen"
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
      <Dtour data={data} dataName={fileName} onLoadData={handleLoadData} />
      {!data && (
        <div className="absolute inset-0 flex items-center justify-center z-20 pointer-events-none">
          <Button
            variant="ghost"
            className="flex flex-col items-center gap-3 px-6 py-4 h-auto pointer-events-auto"
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
        </div>
      )}
    </div>
  );
};

export default App;
