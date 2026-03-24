// GPU continuous selection (bit-packed output)
//
// For each point, checks if the column value falls within any [lo, hi] range.
// Output mask must be pre-cleared; bits are set via atomicOr.

struct Params {
  num_points: u32,
  column_offset: u32,   // columnIndex * numPoints
  num_ranges: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> ranges: array<f32>;  // [lo0, hi0, lo1, hi1, ...]
@group(0) @binding(2) var<storage, read_write> mask: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_points) { return; }

  let v = data[params.column_offset + i];
  for (var r = 0u; r < params.num_ranges; r++) {
    if (v >= ranges[r * 2u] && v <= ranges[r * 2u + 1u]) {
      atomicOr(&mask[i / 32u], 1u << (i % 32u));
      break;
    }
  }
}
