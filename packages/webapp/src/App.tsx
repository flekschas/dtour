import { Dtour } from '@dtour/viewer';
import { useCallback, useRef, useState } from 'react';

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);
  const inputRef = useRef<HTMLInputElement>(null);

  const loadFile = useCallback(async (file: File) => {
    setData(await file.arrayBuffer());
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
      onClick={() => data === undefined && inputRef.current?.click()}
      onKeyDown={(e) => e.key === 'Enter' && data === undefined && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".parquet,.pq,.arrow"
        className="hidden"
        onChange={handleFileSelect}
      />
      <Dtour data={data} />
    </div>
  );
};

export default App;
