import { test } from '@playwright/test';
import {
  DATASETS,
  type Renderer,
  collectMetrics,
  computeStats,
  formatBytes,
  formatHeap,
  loadDataset,
  runPlaybackBenchmark,
  writeResult,
} from '../lib/helpers.ts';

const PLAYBACK_DURATION_MS = 10_000;
const PLAYBACK_SPEED = 1;

for (const dataset of DATASETS) {
  test(`playback benchmark: ${dataset.label}`, async ({ page }, testInfo) => {
    const renderer = testInfo.project.name as Renderer;
    await loadDataset(page, dataset, renderer);

    // Warmup: run a short playback cycle so the render loop is hot.
    // A plain sleep doesn't help — Chrome can deprioritize idle tabs/workers,
    // and deferred shader compilation only triggers on actual draw calls.
    await runPlaybackBenchmark(page, 2000, PLAYBACK_SPEED);

    const metricsBefore = await collectMetrics(page);
    const frameTimes = await runPlaybackBenchmark(page, PLAYBACK_DURATION_MS, PLAYBACK_SPEED);
    const metricsAfter = await collectMetrics(page);

    const stats = computeStats(frameTimes);
    const memDelta = metricsAfter.gpuMemoryBytes - metricsBefore.gpuMemoryBytes;

    console.log(
      `\n=== ${dataset.label} [${renderer}] — playback ${PLAYBACK_DURATION_MS / 1000}s (${metricsBefore.numPoints} pts × ${metricsBefore.numDims} dims) ===`,
    );
    console.log(`  frames     : ${frameTimes.length}`);
    console.log(
      `  GPU memory : ${formatBytes(metricsBefore.gpuMemoryBytes)} → ${formatBytes(metricsAfter.gpuMemoryBytes)} (Δ ${memDelta >= 0 ? '+' : ''}${formatBytes(Math.abs(memDelta))})`,
    );
    console.log(
      `  JS heap    : ${formatHeap(metricsAfter.jsHeapUsedBytes, metricsAfter.workerJsHeapUsedBytes)}`,
    );
    console.log(`  avg        : ${stats.avgMs.toFixed(2)} ms  (${stats.fps.toFixed(1)} fps)`);
    console.log(`  p50        : ${stats.p50Ms.toFixed(2)} ms`);
    console.log(`  p95        : ${stats.p95Ms.toFixed(2)} ms`);
    console.log(`  min/max    : ${stats.minMs.toFixed(2)} / ${stats.maxMs.toFixed(2)} ms`);

    writeResult({
      dataset: dataset.label,
      scenario: 'playback',
      renderer,
      timestamp: new Date().toISOString(),
      numPoints: metricsBefore.numPoints,
      numDims: metricsBefore.numDims,
      gpuMemoryBytes: metricsAfter.gpuMemoryBytes,
      jsHeapUsedBytes: metricsAfter.jsHeapUsedBytes,
      workerJsHeapUsedBytes: metricsAfter.workerJsHeapUsedBytes,
      stats,
    });
  });
}
