import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// The MCP App: one script the server inlines into the HTML resource it serves
export default defineConfig({
  plugins: [tailwindcss(), react()],
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
  },
  build: {
    outDir: resolve(__dirname, 'src/dtour/static'),
    emptyOutDir: false,
    lib: {
      entry: resolve(__dirname, 'js/mcp-app.tsx'),
      formats: ['es'],
      fileName: 'mcp-app',
    },
    rolldownOptions: {
      external: [],
    },
    cssCodeSplit: false,
  },
  worker: {
    format: 'es',
  },
});
