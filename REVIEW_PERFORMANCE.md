# Review: `32c0065` `chore: improve rendering performance`

Build sanity check passed for both `@dtour/scatter` and `@dtour/viewer`.

## Findings

### 1. Medium: worker playback is wired through a mutable ref, so playback can miss the newly created scatter instance

Affected code:
- `packages/viewer/src/DtourViewer.tsx`
- `packages/viewer/src/hooks/usePlayback.ts`

`usePlayback(scatterRef.current)` is evaluated during render, but `scatterRef.current` is only assigned later inside the scatter-creation effect. Because mutating a ref does not trigger a re-render, the new `usePlayback()` effect can remain subscribed to `null` until some unrelated state change happens. In practice that means `tourPlayingAtom` can flip to `true` while the hook still has no live `ScatterInstance`, so `startPlayback()` is skipped and playback begins only after a later render.

This makes the control flow harder to reason about than the previous main-thread rAF loop, because the playback lifecycle now depends on incidental re-renders instead of explicit instance ownership.

Recommendation: promote the scatter instance to React state or trigger playback subscription from the same effect that creates the instance.

### 2. Medium: projection logic now exists in two separate implementations that must stay mathematically identical

Affected code:
- `packages/scatter/src/gpu/worker.ts`
- `packages/scatter/src/shaders/point.wgsl`
- `packages/scatter/src/gpu/projection.ts`
- `packages/scatter/src/gpu/compute-projection.wgsl`

The commit removes projection from the render hot path by folding normalization into `computeAdjustedBasis()` and redoing projection inline in the vertex shader, but lasso selection still uses the old compute projection pipeline. That means the project now has two projection implementations:

- render path: CPU-adjusted basis + inline vertex-shader projection
- lasso path: compute shader using raw basis + normalization buffer

Today the formulas appear intended to match, but future changes to normalization, viewport scaling, precision handling, or basis layout now have to be updated in two places. That is a maintainability regression because the renderer and lasso behavior can silently drift apart even when each path still looks locally correct.

Recommendation: centralize the invariant in one place if possible, or at least document the exact equivalence and add a regression test around lasso/render parity.

### 3. Low: the new "faster" path hides a non-obvious 4x projection loop in the vertex stage

Affected code:
- `packages/scatter/src/shaders/point.wgsl`
- `packages/scatter/src/renderer.ts`

The new renderer draws a 4-vertex triangle strip per point and performs the full `num_dims` projection loop inside `vs_main()`. Because the projected point center does not depend on `vertex_index`, the ND-to-2D work is repeated four times per point. The old design projected once per point in compute and reused the result across all vertices.

This may still be a net win in practice because it removes a compute pass, a barrier, and a projected-buffer read, but the trade-off is not obvious from the code and is easy for future maintainers to misjudge. For higher-dimensional datasets especially, the current implementation could move the bottleneck rather than remove it.

Recommendation: keep a short benchmark note near this change or document the intended crossover point so the next optimization pass has context.

## Overall

The direction is good: removing ephemeral preview projection buffers, throttling UI updates, and moving playback off the main thread all simplify some expensive paths. The main concern is that the commit also introduces more hidden coupling:

- playback depends on ref timing rather than explicit state
- projection correctness now depends on two implementations staying in sync
- the key rendering trade-off is harder to infer from the code than before

I would address item 1 before relying on this behavior broadly, and I would at least document item 2 if you want to keep the split projection design.
