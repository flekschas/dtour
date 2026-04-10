#!/usr/bin/env node

// Thin wrapper that translates --renderer and --out into env vars + Playwright flags.
// Usage: node bench.js [--renderer webgl|webgpu] [--out path.csv] [-- ...playwright args]

import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const playwrightArgs = [];

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--renderer' && args[i + 1]) {
    process.env.BENCH_RENDERER = args[++i];
  } else if (args[i] === '--out' && args[i + 1]) {
    process.env.BENCH_OUT = args[++i];
  } else if (args[i] === '--') {
    playwrightArgs.push(...args.slice(i + 1));
    break;
  } else {
    playwrightArgs.push(args[i]);
  }
}

// Map --renderer to Playwright's --project flag
if (process.env.BENCH_RENDERER) {
  playwrightArgs.unshift(`--project=${process.env.BENCH_RENDERER}`);
}

try {
  execFileSync('pnpm', ['exec', 'playwright', 'test', ...playwrightArgs], {
    stdio: 'inherit',
    env: process.env,
  });
} catch {
  process.exit(1);
}
