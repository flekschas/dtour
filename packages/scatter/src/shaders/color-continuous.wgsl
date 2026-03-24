// GPU continuous color encoding
//
// Reads raw values from the column-major data buffer, normalizes to [0,1],
// and interpolates between colormap LUT stops to produce packed RGBA u32.

struct Params {
  num_points: u32,
  column_offset: u32,   // columnIndex * numPoints
  min_val: f32,
  range_val: f32,
  lut_size: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> colormap: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn unpack_rgb(c: u32) -> vec3f {
  return vec3f(
    f32(c & 0xFFu),
    f32((c >> 8u) & 0xFFu),
    f32((c >> 16u) & 0xFFu),
  );
}

fn pack_rgba(r: f32, g: f32, b: f32) -> u32 {
  return (255u << 24u) | (u32(b) << 16u) | (u32(g) << 8u) | u32(r);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.num_points) { return; }

  let raw = data[params.column_offset + i];
  let inv_range = select(0.0, 1.0 / params.range_val, params.range_val > 0.0);
  let t = clamp((raw - params.min_val) * inv_range, 0.0, 1.0);
  let stops = f32(params.lut_size - 1u);
  let pos = t * stops;
  let idx = min(u32(floor(pos)), params.lut_size - 2u);
  let frac = pos - f32(idx);

  let c0 = unpack_rgb(colormap[idx]);
  let c1 = unpack_rgb(colormap[idx + 1u]);
  let mixed = c0 + frac * (c1 - c0);

  output[i] = pack_rgba(round(mixed.x), round(mixed.y), round(mixed.z));
}
