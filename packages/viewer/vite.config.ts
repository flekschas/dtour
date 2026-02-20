import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), react()],
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'DtourViewer',
      fileName: 'viewer',
      formats: ['es'],
      cssFileName: 'viewer',
    },
    rollupOptions: {
      // Peer dependencies — consumers provide these
      external: ['react', 'react-dom', 'react/jsx-runtime', 'jotai'],
    },
  },
});
