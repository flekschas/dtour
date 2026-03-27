import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), react()],
  // Treat .wgsl files as raw strings (re-used from scatter internals via import)
  assetsInclude: [],
  server: {
    proxy: {
      '/gcs': {
        target: 'https://storage.googleapis.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/gcs/, ''),
      },
    },
  },
});
