import { Dtour, DtourToolbar } from '@dtour/viewer';
import { Provider } from 'jotai';
import { useCallback, useState } from 'react';

const App = () => {
  const [data, setData] = useState<ArrayBuffer | undefined>(undefined);

  const handleDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    setData(await file.arrayBuffer());
  }, []);

  return (
    <Provider>
      <div
        className="flex flex-col w-screen h-screen"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <DtourToolbar />
        <div className="flex-1 min-h-0">
          <Dtour data={data} />
        </div>
      </div>
    </Provider>
  );
};

export default App;
