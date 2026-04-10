/**
 * Inline Web Worker that generates a 1M-point Lorenz-Stenflo hyperchaotic
 * attractor and returns the result as an Arrow IPC ArrayBuffer.
 *
 * The Lorenz-Stenflo system (Stenflo 1996) extends the Lorenz equations with
 * a 4th variable representing atmospheric rotation, producing genuine 4D
 * hyperchaotic behavior with two positive Lyapunov exponents.
 */

/// <reference lib="webworker" />
import { tableFromArrays, tableToIPC } from '@uwdata/flechette';

// Lorenz-Stenflo parameters (hyperchaotic regime)
const a = 2;
const b = 0.7;
const c = 26.5;
const s = 1.5;

type State = [number, number, number, number];

function derivatives([x, y, z, w]: State): State {
  return [a * (y - x) + s * w, c * x - x * z - y, x * y - b * z, -x - a * w];
}

function rk4Step(state: State, dt: number): State {
  const k1 = derivatives(state);
  const s1: State = [
    state[0] + 0.5 * dt * k1[0],
    state[1] + 0.5 * dt * k1[1],
    state[2] + 0.5 * dt * k1[2],
    state[3] + 0.5 * dt * k1[3],
  ];
  const k2 = derivatives(s1);
  const s2: State = [
    state[0] + 0.5 * dt * k2[0],
    state[1] + 0.5 * dt * k2[1],
    state[2] + 0.5 * dt * k2[2],
    state[3] + 0.5 * dt * k2[3],
  ];
  const k3 = derivatives(s2);
  const s3: State = [
    state[0] + dt * k3[0],
    state[1] + dt * k3[1],
    state[2] + dt * k3[2],
    state[3] + dt * k3[3],
  ];
  const k4 = derivatives(s3);
  return [
    state[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    state[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    state[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    state[3] + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
  ];
}

const DEFAULT_NUM_POINTS = 1_000_000;
const dt = 0.005;
const transient = 2000;
const subsample = 5;

self.onmessage = (e: MessageEvent<number | null>) => {
  const numPoints = e.data && Number.isFinite(e.data) && e.data > 0 ? e.data : DEFAULT_NUM_POINTS;
  let state: State = [1, 1, 1, 1];

  // Discard transient
  for (let i = 0; i < transient; i++) {
    state = rk4Step(state, dt);
  }

  const x = new Float32Array(numPoints);
  const y = new Float32Array(numPoints);
  const z = new Float32Array(numPoints);
  const w = new Float32Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    for (let j = 0; j < subsample; j++) {
      state = rk4Step(state, dt);
    }
    x[i] = state[0];
    y[i] = state[1];
    z[i] = state[2];
    w[i] = state[3];
  }

  const table = tableFromArrays({ x, y, z, w });
  const bytes = tableToIPC(table);
  const buffer = bytes.buffer as ArrayBuffer;
  self.postMessage(buffer, [buffer]);
};
