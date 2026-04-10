# Benchmarks

Automated performance benchmarks using Playwright with a real Chrome browser and GPU.

## Results

**2026-04-09** · Apple M1 Max (32GB) MacBook Pro · Chrome 136

### Playback (10s continuous tour, avg FPS)

| Dataset | Points | Dims | WebGPU | WebGL |
|---------|-------:|-----:|-------:|------:|
| fashion-mnist | 69K | 4 | 119 | 119 |
| news-headlines | 205K | 4 | 119 | 118 |
| single-cell | 346K | 9 | 112 | 109 |
| lorenz-1m | 1M | 4 | 104 | 111 |
| lorenz-2m | 2M | 4 | 67 | 96 |
| lorenz-5m | 5M | 4 | 38 | 60 |
| lorenz-10m | 10M | 4 | 25 | 40 |
| lorenz-20m | 20M | 4 | 13 | 24 |

### Scroll (200 setTourPosition steps, avg FPS)

| Dataset | Points | Dims | WebGPU | WebGL |
|---------|-------:|-----:|-------:|------:|
| fashion-mnist | 69K | 4 | 598 | 252 |
| news-headlines | 205K | 4 | 525 | 201 |
| single-cell | 346K | 9 | 216 | 139 |
| lorenz-1m | 1M | 4 | 205 | 145 |
| lorenz-2m | 2M | 4 | 110 | 125 |
| lorenz-5m | 5M | 4 | 49 | 80 |
| lorenz-10m | 10M | 4 | 25 | 42 |
| lorenz-20m | 20M | 4 | 13 | 22 |

### Memory (lorenz-20m)

| | WebGPU | WebGL |
|---|------:|------:|
| GPU | 473 MB | 321 MB |
| Main heap | 46 MB | 47 MB |
| Worker | 0 MB | 305 MB |
| **Total** | **519 MB** | **673 MB** |

WebGL is faster at scale (vertex projection on CPU is cheaper than compute-shader dispatch) but uses ~30% more total memory since it retains raw data on the CPU for lasso selection.

---

## Run

### Prerequisites

```bash
pnpm install
pnpm --filter benchmarks exec playwright install chromium
```

Build scatter and viewer before running:

```bash
pnpm --filter @dtour/scatter build && pnpm --filter @dtour/viewer build
```

### Running

```bash
# All benchmarks, all datasets, both renderers
pnpm bench

# Single renderer
pnpm bench --renderer webgl
pnpm bench --renderer webgpu

# Single dataset
pnpm bench --grep "fashion-mnist"

# Single scenario
pnpm bench scenarios/batch.bench.ts

# Combine freely
pnpm bench --renderer webgl --grep "batch benchmark: lorenz"

# Custom output path (default: benchmarks/results/benchmarks.csv)
pnpm bench --out results/my-run.csv
```

#### Options

| Flag | Description |
|------|-------------|
| `--renderer webgl\|webgpu` | Run only the specified renderer (default: both) |
| `--out path.csv` | Write CSV results to a custom path |
| `--grep "pattern"` | Filter tests by name |

Any other flags are forwarded to Playwright (e.g. `--timeout`, `--retries`).

### Scenarios

| Scenario | What it measures |
|----------|-----------------|
| **batch** | 120 GPU-fenced render frames sweeping the full tour. Pure render throughput. |
| **playback** | 10s continuous tour playback. Rendered frame throughput (GPU-submission cadence) + memory delta. |
| **scroll** | 200 `setTourPosition()` steps with render-complete fencing. Scrub latency. |

### Datasets

| Label | Description |
|-------|-------------|
| `fashion-mnist` | 69K points, 4 dims |
| `news-headlines` | 200K points, 4 dims |
| `single-cell` | 340K points, 18 dims |
| `lorenz-1m` | 1M points, 4 dims (generated) |
| `lorenz-2m` | 2M points, 4 dims (generated) |
| `lorenz-5m` | 5M points, 4 dims (generated) |
| `lorenz-10m` | 10M points, 4 dims (generated) |
| `lorenz-20m` | 20M points, 4 dims (generated) |

The Lorenz attractor is generated on the fly in a Web Worker. You can also load any size manually via `?dataset=lorenz&points=3000000`.

### Output

Results are appended as rows to a single CSV file (default `benchmarks/results/benchmarks.csv`). Columns:

`timestamp`, `renderer`, `dataset`, `scenario`, `numPoints`, `numDims`, `gpuMemoryBytes`, `jsHeapUsedBytes`, `workerJsHeapUsedBytes`, `avgMs`, `fps`, `p50Ms`, `p95Ms`, `minMs`, `maxMs`

### Notes

- Benchmarks run in **headed** mode (not headless). Headless Chrome lacks GPU acceleration, which makes WebGPU fall back to software rendering and produces meaningless numbers.
- Tests run serially (`workers: 1`) to avoid GPU contention.
- The webapp dev server starts automatically if not already running. Set `reuseExistingServer: true` (the default for local) to reuse an existing `pnpm dev`.
