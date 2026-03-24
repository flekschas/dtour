// PCA compute shader — two-pass reduction to compute covariance matrix.
//
// Pass 1 (reduce_partial): Each workgroup reduces its chunk of points into
// per-workgroup partial sums for d means + d*(d+1)/2 cross-product terms.
//
// Pass 2 (final_reduce): A single workgroup reduces all partial sums into
// the final means and covariance matrix.
//
// Data is normalized identically to the projection shader:
//   v = (raw - min) / range - 0.5

struct PcaParams {
  num_points: u32,
  num_dims: u32,
  num_workgroups: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: PcaParams;
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<storage, read> norm_params: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
// result layout: [mean_0 .. mean_{d-1}, cov_00, cov_01, ..., cov_{d-1,d-1}]
// upper-triangle covariance packed as: d1*d - d1*(d1-1)/2 + (d2-d1)
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

const WG: u32 = 256u;
const MAX_DIMS: u32 = 32u;

var<workgroup> wg_reduce: array<f32, 256>;

fn normalize(dim: u32, point_idx: u32, N: u32) -> f32 {
  let raw = data[dim * N + point_idx];
  let np = norm_params[dim];
  let range = max(np.y, 1e-6);
  return (raw - np.x) / range - 0.5;
}

fn cov_idx(d: u32, d1: u32, d2: u32) -> u32 {
  return d1 * d - d1 * (d1 - 1u) / 2u + (d2 - d1);
}

// ─── Pass 1: per-workgroup partial sums ─────────────────────────────────────

@compute @workgroup_size(256)
fn reduce_partial(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let N = params.num_points;
  let d = min(params.num_dims, MAX_DIMS);
  let num_cov = d * (d + 1u) / 2u;
  let num_terms = d + num_cov;
  let i = gid.x;
  let valid = i < N;

  // Load normalized values for this point into registers
  var vals: array<f32, 32>; // MAX_DIMS
  if (valid) {
    for (var dim = 0u; dim < d; dim++) {
      vals[dim] = normalize(dim, i, N);
    }
  }

  // Reduce mean terms
  for (var dim = 0u; dim < d; dim++) {
    wg_reduce[lid.x] = select(0.0, vals[dim], valid);
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (lid.x < s) {
        wg_reduce[lid.x] += wg_reduce[lid.x + s];
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      partials[wid.x * num_terms + dim] = wg_reduce[0];
    }
  }

  // Reduce upper-triangle covariance terms (including diagonal)
  for (var d1 = 0u; d1 < d; d1++) {
    for (var d2 = d1; d2 < d; d2++) {
      wg_reduce[lid.x] = select(0.0, vals[d1] * vals[d2], valid);
      workgroupBarrier();
      for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
          wg_reduce[lid.x] += wg_reduce[lid.x + s];
        }
        workgroupBarrier();
      }
      if (lid.x == 0u) {
        let ci = cov_idx(d, d1, d2);
        partials[wid.x * num_terms + d + ci] = wg_reduce[0];
      }
    }
  }
}

// ─── Pass 2: reduce across workgroups → final means + covariance ────────────

@compute @workgroup_size(256)
fn final_reduce(
  @builtin(local_invocation_id) lid: vec3u,
) {
  let d = min(params.num_dims, MAX_DIMS);
  let num_cov = d * (d + 1u) / 2u;
  let num_terms = d + num_cov;
  let nwg = params.num_workgroups;
  let N_f32 = f32(params.num_points);

  // For each term, reduce across all workgroups
  for (var t = 0u; t < num_terms; t++) {
    // Each thread accumulates a stride of workgroups
    var sum = 0.0;
    var w = lid.x;
    while (w < nwg) {
      sum += partials[w * num_terms + t];
      w += WG;
    }
    wg_reduce[lid.x] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (lid.x < s) {
        wg_reduce[lid.x] += wg_reduce[lid.x + s];
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      // Store as E[X] or E[XY]
      result[t] = wg_reduce[0] / N_f32;
    }
    workgroupBarrier();
  }

  // Correct covariance: cov(X,Y) = E[XY] - E[X]*E[Y]
  if (lid.x == 0u) {
    for (var d1 = 0u; d1 < d; d1++) {
      for (var d2 = d1; d2 < d; d2++) {
        let ci = cov_idx(d, d1, d2);
        result[d + ci] -= result[d1] * result[d2];
      }
    }
  }
}
