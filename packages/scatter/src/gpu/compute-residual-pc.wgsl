// Residual-PC compute shader — two-pass reduction.
//
// Computes the covariance matrix of the residual after projecting out
// the current p×2 tour basis. The top eigenvector of this covariance
// is the "most informative direction not currently visible."
//
// Pass 1 (reduce_partial): each workgroup reduces its chunk of N points
//   into partial sums for p means + p*(p+1)/2 upper-triangle cross-products
//   of the per-point residuals.
//
// Pass 2 (final_reduce): one workgroup sums all partial sums → final
//   mean-corrected covariance. Result layout:
//     [mean_0 .. mean_{p-1}, cov_00, cov_01, ..., cov_{p-1,p-1}]
//   where the covariance entries are the upper triangle packed as:
//     index(d1,d2) = d1*p - d1*(d1-1)/2 + (d2-d1)  (d2 >= d1)
//
// Power iteration (15 steps) is performed on the CPU after readback;
// p is always small (≤64) so this is negligible.

struct Params {
  num_points:    u32,
  num_dims:      u32,
  num_workgroups: u32,
  _pad:          u32,
}

@group(0) @binding(0) var<uniform>            params:     Params;
@group(0) @binding(1) var<storage, read>      data:       array<f32>;
@group(0) @binding(2) var<storage, read>      norm_params: array<vec2f>;
// basis: column-major p×2 — first p floats = x-weights, next p = y-weights
@group(0) @binding(3) var<storage, read>      basis:      array<f32>;
@group(0) @binding(4) var<storage, read_write> partials:  array<f32>;
@group(0) @binding(5) var<storage, read_write> result:    array<f32>;

const WG: u32 = 256u;
const MAX_DIMS: u32 = 64u;

var<workgroup> wg_reduce: array<f32, 256>;

fn cov_idx(d: u32, d1: u32, d2: u32) -> u32 {
  return d1 * d - d1 * (d1 - 1u) / 2u + (d2 - d1);
}

// ─── Pass 1: per-workgroup partial sums ─────────────────────────────────────

@compute @workgroup_size(256)
fn reduce_partial(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id)         wid: vec3u,
) {
  let N         = params.num_points;
  let d         = min(params.num_dims, MAX_DIMS);
  let num_cov   = d * (d + 1u) / 2u;
  let num_terms = d + num_cov;
  let i         = gid.x;
  let valid     = i < N;

  // Load normalized values into registers, then project out the 2D basis.
  var vals: array<f32, 64>;
  if (valid) {
    var dot_x = 0.0;
    var dot_y = 0.0;
    for (var dim = 0u; dim < d; dim++) {
      let raw   = data[dim * N + i];
      let np    = norm_params[dim];
      let range = max(np.y, 1e-6);
      let norm  = (raw - np.x) / range - 0.5;
      vals[dim] = norm;
      dot_x    += norm * basis[dim];
      dot_y    += norm * basis[d + dim];
    }
    // Subtract projection onto span{bx, by} → residual is orthogonal to basis
    for (var dim = 0u; dim < d; dim++) {
      vals[dim] -= dot_x * basis[dim] + dot_y * basis[d + dim];
    }
  }

  // Reduce mean terms (used for mean-correction in pass 2)
  for (var dim = 0u; dim < d; dim++) {
    wg_reduce[lid.x] = select(0.0, vals[dim], valid);
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (lid.x < s) { wg_reduce[lid.x] += wg_reduce[lid.x + s]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      partials[wid.x * num_terms + dim] = wg_reduce[0];
    }
  }

  // Reduce upper-triangle cross-products
  for (var d1 = 0u; d1 < d; d1++) {
    for (var d2 = d1; d2 < d; d2++) {
      wg_reduce[lid.x] = select(0.0, vals[d1] * vals[d2], valid);
      workgroupBarrier();
      for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (lid.x < s) { wg_reduce[lid.x] += wg_reduce[lid.x + s]; }
        workgroupBarrier();
      }
      if (lid.x == 0u) {
        partials[wid.x * num_terms + d + cov_idx(d, d1, d2)] = wg_reduce[0];
      }
    }
  }
}

// ─── Pass 2: reduce across workgroups → final mean-corrected covariance ─────

@compute @workgroup_size(256)
fn final_reduce(@builtin(local_invocation_id) lid: vec3u) {
  let d         = min(params.num_dims, MAX_DIMS);
  let num_cov   = d * (d + 1u) / 2u;
  let num_terms = d + num_cov;
  let nwg       = params.num_workgroups;
  let N_f32     = f32(params.num_points);

  for (var t = 0u; t < num_terms; t++) {
    var sum = 0.0;
    var w   = lid.x;
    while (w < nwg) {
      sum += partials[w * num_terms + t];
      w   += WG;
    }
    wg_reduce[lid.x] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
      if (lid.x < s) { wg_reduce[lid.x] += wg_reduce[lid.x + s]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) { result[t] = wg_reduce[0] / N_f32; }
    workgroupBarrier();
  }

  // Mean-corrected covariance: cov(X,Y) = E[XY] - E[X]*E[Y]
  if (lid.x == 0u) {
    for (var d1 = 0u; d1 < d; d1++) {
      for (var d2 = d1; d2 < d; d2++) {
        let ci = cov_idx(d, d1, d2);
        result[d + ci] -= result[d1] * result[d2];
      }
    }
  }
}
