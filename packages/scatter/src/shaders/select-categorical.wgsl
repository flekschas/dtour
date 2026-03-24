// GPU categorical selection (bit-packed output)
//
// For each point, checks if its category index is in the selected set.
// Output mask must be pre-cleared; bits are set via atomicOr.

struct Params {
  num_points: u32,
  num_labels: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> selected: array<u32>;  // 1 per label: selected or not
@group(0) @binding(2) var<storage, read_write> mask: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_points) { return; }
  let idx = indices[i];
  if (idx < params.num_labels && selected[idx] > 0u) {
    atomicOr(&mask[i / 32u], 1u << (i % 32u));
  }
}
