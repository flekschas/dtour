/**
 * Inline Web Worker that generates two linked rings (tori) in 4D space
 * and returns the result as an Arrow IPC ArrayBuffer.
 *
 * The two rings are interlocked in 4D — they cannot be separated in any
 * single 3D projection, but touring through 4D reveals the linking topology.
 * This demonstrates how tours can uncover structure invisible in lower
 * dimensions.
 *
 * Ring A lies in the (x, y) plane at z=0, w=0.
 * Ring B lies in the (z, w) plane at x=0, y=0, offset so it threads through A.
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

function boxMuller(rand: () => number): [number, number] {
  const u1 = rand();
  const u2 = rand();
  const r = Math.sqrt(-2 * Math.log(u1 || 1e-10));
  const theta = 2 * Math.PI * u2;
  return [r * Math.cos(theta), r * Math.sin(theta)];
}

const R = 3; // major radius of each ring
const TUBE_R = 0.3; // tube thickness (noise around the ring)

const DEFAULT_NUM_POINTS = 500_000;

self.onmessage = (e: MessageEvent<number | null>) => {
  const numPoints = e.data && Number.isFinite(e.data) && e.data > 0 ? e.data : DEFAULT_NUM_POINTS;

  const rand = mulberry32(7);
  const halfA = Math.ceil(numPoints / 2);
  const halfB = numPoints - halfA;

  const x = new Float32Array(numPoints);
  const y = new Float32Array(numPoints);
  const z = new Float32Array(numPoints);
  const w = new Float32Array(numPoints);
  const ring: string[] = new Array(numPoints);

  // Ring A: circle in the (x, y) plane, centered at origin
  // Parametrized as (R*cos(θ), R*sin(θ), 0, 0) + noise in all 4D
  for (let i = 0; i < halfA; i++) {
    const theta = (2 * Math.PI * i) / halfA;
    const [n1, n2] = boxMuller(rand);
    const [n3, n4] = boxMuller(rand);

    x[i] = R * Math.cos(theta) + TUBE_R * n1;
    y[i] = R * Math.sin(theta) + TUBE_R * n2;
    z[i] = TUBE_R * n3;
    w[i] = TUBE_R * n4;
    ring[i] = 'Ring A';
  }

  // Ring B: circle in the (z, w) plane, offset by (R, 0, 0, 0) so it
  // threads through Ring A.
  // Parametrized as (R, 0, R*cos(θ), R*sin(θ)) + noise
  for (let i = 0; i < halfB; i++) {
    const theta = (2 * Math.PI * i) / halfB;
    const [n1, n2] = boxMuller(rand);
    const [n3, n4] = boxMuller(rand);

    const idx = halfA + i;
    x[idx] = R + TUBE_R * n1;
    y[idx] = TUBE_R * n2;
    z[idx] = R * Math.cos(theta) + TUBE_R * n3;
    w[idx] = R * Math.sin(theta) + TUBE_R * n4;
    ring[idx] = 'Ring B';
  }

  const table = tableFromArrays({ x, y, z, w, ring });
  const bytes = tableToIPC(table, {});
  const buffer = bytes!.buffer as ArrayBuffer;
  self.postMessage(buffer, [buffer]);
};
