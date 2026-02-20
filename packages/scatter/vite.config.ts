import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'DtourScatter',
      fileName: 'scatter',
      formats: ['es'],
    },
    rollupOptions: {
      // flechette and hyparquet are bundled in (needed inside workers)
      external: [],
    },
  },
  worker: {
    // Inline workers as Blob URLs — consumers get a single JS file,
    // no separate worker files to serve.
    format: 'es',
    rollupOptions: {
      // Workers are self-contained; bundle all deps
    },
  },
  // WGSL files are imported with the ?raw suffix (Vite built-in), no plugin needed.
});
