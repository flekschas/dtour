# Review of `5633ecd98014b234aa50a0a3f031762186434db1`

## Findings

### 1. High: synthetic benchmarking mutates renderer state and leaks WebGPU buffers

`packages/scatter/src/gpu/worker.ts` `handleBenchmark()` replaces `state.projectionResources`, `state.numPoints`, `state.numDims`, `state.normMins`, `state.normRanges`, and `state.adjBasisBuffer`, but unlike the normal `data` load path it does not destroy the old projection buffers before overwriting them. Re-running the synthetic benchmark will therefore leak GPU memory.

More importantly, the benchmark path is destructive to the current session state. After the benchmark completes, the previous dataset/tour is not restored. `renderMainView()` still uses `state.tour` if one exists, while `renderView()` projects using `state.numDims`; after benchmarking on top of a loaded session those can diverge, so later renders can use a stale tour against the synthetic benchmark dimensions. The WebGL path in `packages/scatter/src/webgl/worker.ts` has the same "overwrite live state and do not restore it" problem, even though it does at least delete the previous texture.

I would treat the synthetic benchmark as an isolated operation: either snapshot and fully restore renderer state afterwards, or run it in separate temporary resources that never replace the live scene.

### 2. Medium: `benchmark()` can hang and cannot distinguish concurrent requests

Both `packages/scatter/src/gpu/client.ts` and `packages/scatter/src/webgl/client.ts` implement `benchmark()` by subscribing to the shared status stream and resolving on the next `benchmarkResult`. That has two problems:

- There is no rejection path. If the worker responds with `{ type: 'error' }` instead of `benchmarkResult` (for example when `benchmarkExisting` is called before data is loaded), the promise never settles.
- There is no request identifier. If two benchmarks are started close together, both subscribers resolve from the first `benchmarkResult`, so callers can receive the wrong run's result.

This should be converted into a request/response flow with a benchmark request id, plus rejection on worker error or teardown.

### 3. Low: `DtourViewer` ships an unguarded global debug hook

`packages/viewer/src/DtourViewer.tsx` assigns `globalThis.scatter = instance` during normal setup and never clears it during cleanup. That leaks an internal object into global scope and can leave stale references to destroyed workers behind. It also makes the public surface area harder to reason about because the component now has an undocumented side channel.

If this is only for debugging, it should be behind a development-only guard or removed before merge.

## Clarity And Maintainability

The commit message says this is about "better color encoding", but the diff also introduces a full WebGL backend, benchmark plumbing, viewer/backend props, and a global debug hook. That makes the change much harder to review, test, and bisect than it needs to be.

I would strongly prefer this split into separate commits:

- color encoding and shader-path changes
- WebGL backend introduction
- benchmark API / viewer plumbing

## Testing Gaps

I did not find targeted automated coverage for the new benchmark lifecycle or the color-mode changes. At minimum, I would want:

- a regression test that synthetic benchmarking is non-destructive to a loaded session
- a client-side test that `benchmark()` rejects on worker errors instead of hanging
- a test that concurrent benchmark calls cannot cross-resolve
- a small backend-parity smoke test for switching between uniform, continuous, and categorical color modes
