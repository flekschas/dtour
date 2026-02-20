import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), react()],
  // Treat .wgsl files as raw strings (re-used from scatter internals via import)
  assetsInclude: [],
});
