// ND → 2D projection compute shader
//
// Reads p-dimensional point data from a single concatenated storage buffer
// (column-major: dim0[0..N], dim1[0..N], ..., dimP[0..N]),
// normalizes each dimension using per-dim [min, range] params,
// and projects to 2D using the supplied p×2 basis matrix.
//
// Output: N interleaved [x, y] pairs in the projected buffer.

struct Params {
  num_points: u32,
  num_dims: u32,
  viewport_scale: f32,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<f32>;
// norm_params[d] = vec2f(min, range) for dimension d
@group(0) @binding(2) var<storage, read> norm_params: array<vec2f>;
// basis: column-major p×2 — first p floats = x-weights, next p = y-weights
@group(0) @binding(3) var<storage, read> basis: array<f32>;
@group(0) @binding(4) var<storage, read_write> projected: array<f32>;

@compute @workgroup_size(256)
fn project(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_points) { return; }

  let N = params.num_points;
  let p = params.num_dims;

  var x = 0.0;
  var y = 0.0;

  for (var d = 0u; d < p; d++) {
    let raw = data[d * N + i];
    let np = norm_params[d];
    let range = max(np.y, 1e-6);
    // Center to [-0.5, 0.5]
    let v = (raw - np.x) / range - 0.5;

    x += v * basis[d];
    y += v * basis[p + d];
  }

  projected[i * 2u] = x * params.viewport_scale;
  projected[i * 2u + 1u] = y * params.viewport_scale;
}
