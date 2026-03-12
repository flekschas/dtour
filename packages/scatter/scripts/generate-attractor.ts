/**
 * Generate a 4D Lorenz-Stenflo hyperchaotic attractor as an Arrow IPC file.
 *
 * Usage:
 *   pnpm --filter @dtour/scatter generate-attractor [numPoints] [output]
 *
 * Defaults: 500_000 points → data/lorenz-stenflo-500k.arrow
 *
 * The Lorenz-Stenflo system (Stenflo 1996) extends the Lorenz equations with
 * a 4th variable representing atmospheric rotation. It produces genuine 4D
 * hyperchaotic behavior with two positive Lyapunov exponents.
 *
 *   dx/dt = a(y − x) + s·w
 *   dy/dt = c·x − x·z − y
 *   dz/dt = x·y − b·z
 *   dw/dt = −x − a·w
 */

import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { tableFromArrays, tableToIPC } from '@uwdata/flechette'

// --- Lorenz-Stenflo parameters (hyperchaotic regime) ---
const a = 2
const b = 0.7
const c = 26.5
const s = 1.5

type State = [number, number, number, number]

function derivatives([x, y, z, w]: State): State {
  return [a * (y - x) + s * w, c * x - x * z - y, x * y - b * z, -x - a * w]
}

function rk4Step(state: State, dt: number): State {
  const k1 = derivatives(state)
  const s1: State = [
    state[0] + 0.5 * dt * k1[0],
    state[1] + 0.5 * dt * k1[1],
    state[2] + 0.5 * dt * k1[2],
    state[3] + 0.5 * dt * k1[3],
  ]
  const k2 = derivatives(s1)
  const s2: State = [
    state[0] + 0.5 * dt * k2[0],
    state[1] + 0.5 * dt * k2[1],
    state[2] + 0.5 * dt * k2[2],
    state[3] + 0.5 * dt * k2[3],
  ]
  const k3 = derivatives(s2)
  const s3: State = [
    state[0] + dt * k3[0],
    state[1] + dt * k3[1],
    state[2] + dt * k3[2],
    state[3] + dt * k3[3],
  ]
  const k4 = derivatives(s3)
  return [
    state[0] + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    state[1] + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
    state[2] + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
    state[3] + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
  ]
}

// --- Config ---
const numPoints = Number(process.argv[2]) || 500_000
const rootDir = resolve(import.meta.dirname, '..', '..', '..')
const output =
  process.argv[3] ||
  resolve(rootDir, 'data', `lorenz-stenflo-${(numPoints / 1000) | 0}k.arrow`)
const dt = 0.005
const transient = 2000
const subsample = 5 // record every Nth step for more uniform coverage

console.log(`Generating ${numPoints.toLocaleString()} points...`)

// --- Integrate ---
let state: State = [1, 1, 1, 1]

// Discard transient
for (let i = 0; i < transient; i++) {
  state = rk4Step(state, dt)
}

const x = new Float32Array(numPoints)
const y = new Float32Array(numPoints)
const z = new Float32Array(numPoints)
const w = new Float32Array(numPoints)

for (let i = 0; i < numPoints; i++) {
  for (let j = 0; j < subsample; j++) {
    state = rk4Step(state, dt)
  }
  x[i] = state[0]
  y[i] = state[1]
  z[i] = state[2]
  w[i] = state[3]
}

function fmin(arr: Float32Array): number {
  let m = Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i]
  return m
}
function fmax(arr: Float32Array): number {
  let m = -Infinity
  for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i]
  return m
}

console.log(`  x range: [${fmin(x).toFixed(2)}, ${fmax(x).toFixed(2)}]`)
console.log(`  y range: [${fmin(y).toFixed(2)}, ${fmax(y).toFixed(2)}]`)
console.log(`  z range: [${fmin(z).toFixed(2)}, ${fmax(z).toFixed(2)}]`)
console.log(`  w range: [${fmin(w).toFixed(2)}, ${fmax(w).toFixed(2)}]`)

// --- Write Arrow IPC ---
const table = tableFromArrays({ x, y, z, w })
const bytes = tableToIPC(table)
writeFileSync(output, bytes)

const sizeMB = (bytes.byteLength / 1024 / 1024).toFixed(1)
console.log(`Wrote ${output} (${sizeMB} MB)`)
