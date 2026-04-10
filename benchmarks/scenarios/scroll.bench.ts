import { test } from '@playwright/test';
import {
  DATASETS,
  type Renderer,
  collectMetrics,
  computeStats,
  formatBytes,
  formatHeap,
  loadDataset,
  runScrollBenchmark,
  writeResult,
} from '../lib/helpers.ts';

const NUM_STEPS = 200;

for (const dataset of DATASETS) {
  test(`scroll benchmark: ${dataset.label}`, async ({ page }, testInfo) => {
    const renderer = testInfo.project.name as Renderer;
    await loadDataset(page, dataset, renderer);
    // Warmup: scrub through a few positions so the render loop is hot.
    await runScrollBenchmark(page, 20);

    const metrics = await collectMetrics(page);
    const frameTimes = await runScrollBenchmark(page, NUM_STEPS);

    const stats = computeStats(frameTimes);

    console.log(
      `\n=== ${dataset.label} [${renderer}] — scroll ${NUM_STEPS} steps (${metrics.numPoints} pts × ${metrics.numDims} dims) ===`,
    );
    console.log(`  GPU memory : ${formatBytes(metrics.gpuMemoryBytes)}`);
    console.log(
      `  JS heap    : ${formatHeap(metrics.jsHeapUsedBytes, metrics.workerJsHeapUsedBytes)}`,
    );
    console.log(`  avg        : ${stats.avgMs.toFixed(2)} ms  (${stats.fps.toFixed(1)} fps)`);
    console.log(`  p50        : ${stats.p50Ms.toFixed(2)} ms`);
    console.log(`  p95        : ${stats.p95Ms.toFixed(2)} ms`);
    console.log(`  min/max    : ${stats.minMs.toFixed(2)} / ${stats.maxMs.toFixed(2)} ms`);

    writeResult({
      dataset: dataset.label,
      scenario: 'scroll',
      renderer,
      timestamp: new Date().toISOString(),
      numPoints: metrics.numPoints,
      numDims: metrics.numDims,
      gpuMemoryBytes: metrics.gpuMemoryBytes,
      jsHeapUsedBytes: metrics.jsHeapUsedBytes,
      workerJsHeapUsedBytes: metrics.workerJsHeapUsedBytes,
      stats,
    });
  });
}
