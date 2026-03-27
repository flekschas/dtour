import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), react()],
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  build: {
    outDir: resolve(__dirname, 'src/dtour/static'),
    emptyOutDir: true,
    lib: {
      entry: resolve(__dirname, 'js/widget.tsx'),
      formats: ['es'],
      fileName: 'widget',
    },
    rolldownOptions: {
      // Bundle everything — Jupyter has no module resolution.
      external: [],
    },
    // CSS is inlined into JS (injected into Shadow DOM at runtime)
    cssCodeSplit: false,
  },
  worker: {
    format: 'es',
  },
});
