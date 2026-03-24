// GPU categorical color encoding
//
// Reads per-point category index, looks up packed RGBA from palette.

struct Params {
  num_points: u32,
  palette_size: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> palette: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_points) { return; }
  output[i] = palette[indices[i] % params.palette_size];
}
