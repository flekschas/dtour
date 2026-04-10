import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './scenarios',
  testMatch: '**/*.bench.ts',
  // Generous timeout for large lorenz datasets (10M pts takes time to generate + benchmark)
  timeout: 300_000,
  retries: 0,
  // Serial execution — GPU contention skews results
  workers: 1,
  use: {
    baseURL: 'http://localhost:5173',
    // Must use headed mode — headless Chrome has no GPU acceleration,
    // making WebGPU fall back to software rendering (~100x slower).
    headless: false,
    launchOptions: {
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox'],
    },
  },
  webServer: {
    command: 'pnpm --filter webapp dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
    cwd: '..',
  },
  projects: [
    {
      name: 'webgpu',
      use: { browserName: 'chromium' },
    },
    {
      name: 'webgl',
      use: { browserName: 'chromium' },
    },
  ],
});
