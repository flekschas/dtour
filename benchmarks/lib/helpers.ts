import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { ScatterInstance } from '@dtour/scatter';
import type { Page } from '@playwright/test';

export type DatasetSlug = 'fashion-mnist' | 'news-headlines' | 'single-cell' | 'lorenz';

export type DatasetSpec = {
  slug: DatasetSlug;
  /** Display label used in test names and CSV output. */
  label: string;
  /** Override point count (lorenz only). */
  points?: number;
};

export const DATASETS: DatasetSpec[] = [
  { slug: 'fashion-mnist', label: 'fashion-mnist' },
  { slug: 'news-headlines', label: 'news-headlines' },
  { slug: 'single-cell', label: 'single-cell' },
  { slug: 'lorenz', label: 'lorenz-1m', points: 1_000_000 },
  { slug: 'lorenz', label: 'lorenz-2m', points: 2_000_000 },
  { slug: 'lorenz', label: 'lorenz-5m', points: 5_000_000 },
  { slug: 'lorenz', label: 'lorenz-10m', points: 10_000_000 },
  { slug: 'lorenz', label: 'lorenz-20m', points: 20_000_000 },
];

export type Renderer = 'webgpu' | 'webgl';

export type BenchmarkResult = {
  dataset: string;
  scenario: string;
  renderer: Renderer;
  timestamp: string;
  numPoints: number;
  numDims: number;
  gpuMemoryBytes: number;
  /** Main-thread JS heap measured on the page. */
  jsHeapUsedBytes: number | null;
  /** GPU-worker JS heap measured inside the render worker. */
  workerJsHeapUsedBytes: number | null;
  stats: {
    avgMs: number;
    fps: number;
    p50Ms: number;
    p95Ms: number;
    minMs: number;
    maxMs: number;
  };
};

/** Navigate to webapp with dataset + renderer pre-selected, wait for data load to complete. */
export const loadDataset = async (
  page: Page,
  dataset: DatasetSpec,
  renderer: Renderer = 'webgpu',
): Promise<void> => {
  const params = new URLSearchParams({
    dataset: dataset.slug,
    benchmark: 'true',
    renderer,
  });
  if (dataset.points) params.set('points', String(dataset.points));
  await page.goto(`/?${params}`);
  await page.waitForFunction(() => (globalThis as Record<string, unknown>).__dtourReady === true, {
    timeout: 120_000,
  });
};

/** Collect a metrics snapshot via scatter.getMetrics().
 *  GPU memory is computed from known texture/buffer sizes inside the worker.
 *  Worker memory is the sum of ArrayBuffer backing stores (dataBuffers, tour bases, etc.)
 *  since V8's heap counters exclude transferred ArrayBuffer storage. */
export const collectMetrics = async (
  page: Page,
): Promise<{
  gpuMemoryBytes: number;
  numPoints: number;
  numDims: number;
  jsHeapUsedBytes: number | null;
  workerJsHeapUsedBytes: number | null;
}> => {
  return page.evaluate(() =>
    // biome-ignore lint/suspicious/noExplicitAny: untyped page context
    ((globalThis as any).scatter as any).getMetrics(),
  );
};

/** Run a playback scenario: start playback for duration, return all frame times from worker. */
export const runPlaybackBenchmark = async (
  page: Page,
  durationMs: number,
  speed = 1,
): Promise<number[]> => {
  return page.evaluate(
    async ({ durationMs, speed }) => {
      const scatter = (globalThis as Record<string, unknown>).scatter as ScatterInstance;

      // Collect the playbackStopped event which contains ALL rAF frame times from the worker
      const stoppedPromise = new Promise<number[]>((resolve) => {
        const unsub = scatter.subscribe((s) => {
          if (s.type === 'playbackStopped') {
            unsub();
            resolve(Array.from(s.frameTimes as Float64Array));
          }
        });
      });

      scatter.startPlayback(speed, 1);
      await new Promise((r) => setTimeout(r, durationMs));
      scatter.stopPlayback();

      return stoppedPromise;
    },
    { durationMs, speed },
  );
};

/** Run a scroll/scrub scenario: programmatically set tour positions and measure latency. */
export const runScrollBenchmark = async (page: Page, numSteps: number): Promise<number[]> => {
  return page.evaluate(async (numSteps) => {
    const scatter = (globalThis as Record<string, unknown>).scatter as ScatterInstance;
    const frameTimes: number[] = [];

    for (let i = 0; i < numSteps; i++) {
      const position = i / numSteps;
      const t0 = performance.now();
      scatter.setTourPosition(position);
      // Wait for the rendered event for the main view
      await new Promise<void>((resolve) => {
        const unsub = scatter.subscribe((s) => {
          if (s.type === 'rendered' && s.viewIndex === 0) {
            unsub();
            resolve();
          }
        });
      });
      frameTimes.push(performance.now() - t0);
    }

    return frameTimes;
  }, numSteps);
};

/** Compute summary statistics from an array of frame times. */
export const computeStats = (frameTimes: number[]) => {
  if (frameTimes.length === 0) return { avgMs: 0, fps: 0, p50Ms: 0, p95Ms: 0, minMs: 0, maxMs: 0 };
  const sorted = [...frameTimes].sort((a, b) => a - b);
  const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
  return {
    avgMs: avg,
    fps: 1000 / avg,
    p50Ms: sorted[Math.floor(sorted.length * 0.5)]!,
    p95Ms: sorted[Math.floor(sorted.length * 0.95)]!,
    minMs: sorted[0]!,
    maxMs: sorted[sorted.length - 1]!,
  };
};

const CSV_COLUMNS = [
  'timestamp',
  'renderer',
  'dataset',
  'scenario',
  'numPoints',
  'numDims',
  'gpuMemoryBytes',
  'jsHeapUsedBytes',
  'workerJsHeapUsedBytes',
  'avgMs',
  'fps',
  'p50Ms',
  'p95Ms',
  'minMs',
  'maxMs',
] as const;

/** Append a benchmark result as a row to CSV. Defaults to results/benchmarks.csv; override with BENCH_OUT env var. */
export const writeResult = (result: BenchmarkResult): void => {
  const file = process.env.BENCH_OUT
    ? resolve(process.cwd(), process.env.BENCH_OUT)
    : resolve(import.meta.dirname, '..', 'results', 'benchmarks.csv');
  mkdirSync(resolve(file, '..'), { recursive: true });

  const header = CSV_COLUMNS.join(',');
  const row = [
    result.timestamp,
    result.renderer,
    result.dataset,
    result.scenario,
    result.numPoints,
    result.numDims,
    result.gpuMemoryBytes,
    result.jsHeapUsedBytes ?? '',
    result.workerJsHeapUsedBytes ?? '',
    result.stats.avgMs,
    result.stats.fps,
    result.stats.p50Ms,
    result.stats.p95Ms,
    result.stats.minMs,
    result.stats.maxMs,
  ].join(',');

  if (existsSync(file)) {
    const existingHeader = readFileSync(file, 'utf8').split('\n')[0]?.trim();
    if (existingHeader !== header) {
      throw new Error(
        `CSV schema mismatch in ${file}.\nExpected: ${header}\nFound:    ${existingHeader}\nDelete the file or set BENCH_OUT to a new path.`,
      );
    }
    appendFileSync(file, `${row}\n`);
  } else {
    appendFileSync(file, `${header}\n${row}\n`);
  }
};

/** Format bytes as a human-readable string. */
export const formatBytes = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

/** Format JS heap breakdown: "45.0 MB total (main: 39.8 MB + worker: 5.2 MB)". */
export const formatHeap = (main: number | null, worker: number | null): string => {
  // Treat undefined / NaN / Infinity as absent — these can arrive when the browser
  // exposes performance.memory but returns a non-finite value for a context.
  const m = typeof main === 'number' && Number.isFinite(main) ? main : null;
  const w = typeof worker === 'number' && Number.isFinite(worker) ? worker : null;
  if (m === null && w === null) return 'N/A';
  if (w === null) return formatBytes(m!);
  if (m === null) return `worker: ${formatBytes(w)}`;
  return `${formatBytes(m + w)} total (main: ${formatBytes(m)} + worker: ${formatBytes(w)})`;
};
