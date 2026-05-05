/**
 * Inline Web Worker that generates multivariate Gaussian blobs in 5D space
 * and returns the result as an Arrow IPC ArrayBuffer.
 *
 * Produces K clusters with distinct centers and varying covariance, designed
 * to demonstrate how grand tours reveal cluster separation that may be hidden
 * in any single 2D projection.
 */

/// <reference lib="webworker" />
import { tableFromArrays, tableToIPC } from '@uwdata/flechette';

// Mulberry32 PRNG for reproducibility
function mulberry32(initialSeed: number) {
  let s = initialSeed;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform: two uniform → two normal
function boxMuller(rand: () => number): [number, number] {
  const u1 = rand();
  const u2 = rand();
  const r = Math.sqrt(-2 * Math.log(u1 || 1e-10));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

const K = 6; // number of clusters

const CLUSTER_LABELS = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta'];

const DEFAULT_NUM_POINTS = 500_000;

self.onmessage = (e: MessageEvent<number | null>) => {
  const numPoints = e.data && Number.isFinite(e.data) && e.data > 0 ? e.data : DEFAULT_NUM_POINTS;

  const rand = mulberry32(42);

  type Vec5 = [number, number, number, number, number];

  // Generate cluster centers spread out in 5D
  const centers: Vec5[] = [];
  const spreads: number[] = [];
  const axisScales: Vec5[] = [];

  for (let k = 0; k < K; k++) {
    centers.push([
      (rand() - 0.5) * 10,
      (rand() - 0.5) * 10,
      (rand() - 0.5) * 10,
      (rand() - 0.5) * 10,
      (rand() - 0.5) * 10,
    ]);
    spreads.push(0.4 + rand() * 0.8);
    axisScales.push([
      0.5 + rand() * 1.5,
      0.5 + rand() * 1.5,
      0.5 + rand() * 1.5,
      0.5 + rand() * 1.5,
      0.5 + rand() * 1.5,
    ]);
  }

  // Distribute points evenly, giving remainder to the first clusters
  const base = Math.floor(numPoints / K);
  const remainder = numPoints - base * K;

  const xArr = new Float32Array(numPoints);
  const yArr = new Float32Array(numPoints);
  const zArr = new Float32Array(numPoints);
  const wArr = new Float32Array(numPoints);
  const vArr = new Float32Array(numPoints);
  const clusterArr: string[] = new Array(numPoints);

  let idx = 0;
  for (let k = 0; k < K; k++) {
    const [cx, cy, cz, cw, cv] = centers[k]!;
    const sk = spreads[k]!;
    const [ax, ay, az, aw, av] = axisScales[k]!;
    const count = base + (k < remainder ? 1 : 0);

    for (let i = 0; i < count; i++) {
      const [n1, n2] = boxMuller(rand);
      const [n3, n4] = boxMuller(rand);
      const [n5] = boxMuller(rand);

      xArr[idx] = cx + n1 * sk * ax;
      yArr[idx] = cy + n2 * sk * ay;
      zArr[idx] = cz + n3 * sk * az;
      wArr[idx] = cw + n4 * sk * aw;
      vArr[idx] = cv + n5 * sk * av;
      clusterArr[idx] = CLUSTER_LABELS[k]!;
      idx++;
    }
  }

  const table = tableFromArrays({
    x: xArr,
    y: yArr,
    z: zArr,
    w: wArr,
    v: vArr,
    cluster: clusterArr,
  });
  const bytes = tableToIPC(table, {});
  const buffer = bytes!.buffer as ArrayBuffer;
  self.postMessage(buffer, [buffer]);
};
