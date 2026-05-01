import { createReadStream, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { type Plugin, defineConfig } from 'vite';

/** Serve files from the monorepo `data/` directory at `/data/*` in dev. */
function serveDataDir(): Plugin {
  const dataDir = resolve(__dirname, '../../data');
  return {
    name: 'serve-data-dir',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/data/')) return next();
        const filePath = join(dataDir, req.url.slice(6));
        if (!filePath.startsWith(dataDir)) return next();
        try {
          const stat = statSync(filePath);
          res.writeHead(200, {
            'Content-Length': stat.size,
            'Content-Type': 'application/octet-stream',
          });
          createReadStream(filePath).pipe(res);
        } catch {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [tailwindcss(), react(), serveDataDir()],
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
