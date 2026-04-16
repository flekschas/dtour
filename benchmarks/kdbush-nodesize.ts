/**
 * Benchmark kdbush construction + query times across point counts and nodeSizes.
 *
 * Generates 2D projected positions from a Lorenz-Stenflo attractor (simple
 * random 2D projection of the 4D system) and measures:
 *   - index construction time
 *   - memory overhead (approximate)
 *   - single-point nearest-neighbor query time (median of 1000 queries)
 *
 * Usage:
 *   npx tsx benchmarks/kdbush-nodesize.ts
 */

import KDBush from 'kdbush';

// --- Lorenz-Stenflo attractor (same params as generate-attractor.ts) ---
const a = 2;
const b = 0.7;
const c = 26.5;
const s = 1.5;
type State = [number, number, number, number];

const derivatives = ([x, y, z, w]: State): State => [
  a * (y - x) + s * w,
  c * x - x * z - y,
  x * y - b * z,
  -x - a * w,
];

const rk4Step = (state: State, dt: number): State => {
  const k1 = derivatives(state);
  const h = 0.5 * dt;
  const s1: State = [
    state[0] + h * k1[0],
    state[1] + h * k1[1],
    state[2] + h * k1[2],
    state[3] + h * k1[3],
  ];
  const k2 = derivatives(s1);
  const s2: State = [
    state[0] + h * k2[0],
    state[1] + h * k2[1],
    state[2] + h * k2[2],
    state[3] + h * k2[3],
  ];
  const k3 = derivatives(s2);
  const s3: State = [
    state[0] + dt * k3[0],
    state[1] + dt * k3[1],
    state[2] + dt * k3[2],
    state[3] + dt * k3[3],
  ];
  const k4 = derivatives(s3);
  const d = dt / 6;
  return [
    state[0] + d * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    state[1] + d * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    state[2] + d * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    state[3] + d * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
  ];
};

/** Generate N 2D points by projecting the 4D attractor onto dims 0,1. */
const generatePoints = (n: number): Float32Array => {
  const coords = new Float32Array(n * 2);
  let state: State = [1, 1, 1, 1];
  // Discard transient
  for (let i = 0; i < 2000; i++) state = rk4Step(state, 0.005);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 5; j++) state = rk4Step(state, 0.005);
    coords[i * 2] = state[0];
    coords[i * 2 + 1] = state[1];
  }
  return coords;
};

// --- Benchmark config ---
const POINT_COUNTS = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000];
const NODE_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
const NUM_QUERIES = 1000;

/** Proposed dynamic nodeSize mapping. */
const dynamicNodeSize = (n: number): number => {
  if (n <= 100_000) return 64;
  if (n <= 1_000_000) return 128;
  if (n <= 5_000_000) return 256;
  if (n <= 10_000_000) return 512;
  return 1024;
};

type Result = {
  points: number;
  nodeSize: number;
  buildMs: number;
  queryMedianUs: number;
  queryP95Us: number;
  memoryMB: number;
  isDynamic: boolean;
};

const results: Result[] = [];

console.log('Generating points and benchmarking kdbush...\n');
console.log(
  'points'.padStart(12),
  'nodeSize'.padStart(10),
  'build(ms)'.padStart(11),
  'query-med(µs)'.padStart(15),
  'query-p95(µs)'.padStart(15),
  'mem(MB)'.padStart(9),
  'dynamic'.padStart(9),
);
console.log('-'.repeat(82));

for (const n of POINT_COUNTS) {
  process.stdout.write(`Generating ${(n / 1e6).toFixed(1)}M points... `);
  const coords = generatePoints(n);
  console.log('done');

  // Compute data bounds for query point generation
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) {
    const x = coords[i * 2]!;
    const y = coords[i * 2 + 1]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;

  // Search radius ~ 1% of data extent (typical hover precision)
  const radius = Math.max(rangeX, rangeY) * 0.01;

  for (const nodeSize of NODE_SIZES) {
    // Force GC if available
    if (globalThis.gc) globalThis.gc();
    const heapBefore = process.memoryUsage().heapUsed;

    const t0 = performance.now();
    const index = new KDBush(n, nodeSize);
    for (let i = 0; i < n; i++) {
      index.add(coords[i * 2]!, coords[i * 2 + 1]!);
    }
    index.finish();
    const buildMs = performance.now() - t0;

    const heapAfter = process.memoryUsage().heapUsed;
    const memoryMB = Math.max(0, (heapAfter - heapBefore) / 1024 / 1024);

    // Query benchmark: random points within data bounds
    const queryTimes: number[] = [];
    for (let q = 0; q < NUM_QUERIES; q++) {
      const qx = minX + Math.random() * rangeX;
      const qy = minY + Math.random() * rangeY;
      const qt0 = performance.now();
      index.within(qx, qy, radius);
      queryTimes.push((performance.now() - qt0) * 1000); // µs
    }
    queryTimes.sort((a, b) => a - b);
    const queryMedianUs = queryTimes[Math.floor(NUM_QUERIES * 0.5)]!;
    const queryP95Us = queryTimes[Math.floor(NUM_QUERIES * 0.95)]!;

    const isDynamic = nodeSize === dynamicNodeSize(n);

    const row: Result = {
      points: n,
      nodeSize,
      buildMs,
      queryMedianUs,
      queryP95Us,
      memoryMB,
      isDynamic,
    };
    results.push(row);

    console.log(
      n.toLocaleString().padStart(12),
      String(nodeSize).padStart(10),
      buildMs.toFixed(1).padStart(11),
      queryMedianUs.toFixed(1).padStart(15),
      queryP95Us.toFixed(1).padStart(15),
      memoryMB.toFixed(1).padStart(9),
      (isDynamic ? '  <<<' : '').padStart(9),
    );
  }
  console.log();
}

// Summary: show the dynamic nodeSize picks
console.log('\n=== Dynamic nodeSize summary ===\n');
console.log(
  'points'.padStart(12),
  'nodeSize'.padStart(10),
  'build(ms)'.padStart(11),
  'query-med(µs)'.padStart(15),
  'query-p95(µs)'.padStart(15),
);
console.log('-'.repeat(64));
for (const r of results.filter((r) => r.isDynamic)) {
  console.log(
    r.points.toLocaleString().padStart(12),
    String(r.nodeSize).padStart(10),
    r.buildMs.toFixed(1).padStart(11),
    r.queryMedianUs.toFixed(1).padStart(15),
    r.queryP95Us.toFixed(1).padStart(15),
  );
}
