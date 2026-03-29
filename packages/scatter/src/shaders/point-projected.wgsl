// Benchmark variant: reads pre-projected 2D positions from a storage buffer
// instead of doing inline ND→2D projection in the vertex stage.
//
// Same bind group layout as point.wgsl (6 bindings) so it can share the
// pipeline layout. Binding 1 is reinterpreted as N*2 interleaved [x,y]
// floats; binding 5 (adj_basis) is declared but unused.

struct Uniforms {
  point_size: f32,
  opacity: f32,
  usePerPointColor: f32,
  useSelectionMask: f32,
  color: vec4f,
  useSubtractive: f32,
  num_points: u32,
  num_dims: u32,
  _pad: u32,
  bias: vec2f,
}

struct Camera {
  pan_x: f32,
  pan_y: f32,
  zoom: f32,
  aspect: f32,
  viewport_height: f32,
  inset_offset_y: f32,
  inset_zoom: f32,
}

@group(0) @binding(0) var<uniform> uni: Uniforms;
// Pre-projected 2D positions: [x0, y0, x1, y1, ...]
@group(0) @binding(1) var<storage, read> projected: array<f32>;
@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var<storage, read> pointColors: array<u32>;
@group(0) @binding(4) var<storage, read> selectionMask: array<u32>;
@group(0) @binding(5) var<storage, read> _adj_basis: array<f32>;

struct VertOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) point_color: vec4f,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) @interpolate(flat) effective_opacity: f32,
}

fn quad_vertex(vi: u32) -> vec2f {
  switch vi {
    case 0u: { return vec2f(-1.0, -1.0); }
    case 1u: { return vec2f( 1.0, -1.0); }
    case 2u: { return vec2f(-1.0,  1.0); }
    default: { return vec2f( 1.0,  1.0); }
  }
}

fn unpackColor(packed: u32) -> vec4f {
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, a);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertOut {
  // Read pre-projected 2D center — no ND→2D loop.
  let x = projected[ii * 2u];
  let y = projected[ii * 2u + 1u];

  let iz = camera.inset_zoom;
  let center = vec2f(
    (x + camera.pan_x) * camera.zoom * iz / camera.aspect,
    (y + camera.pan_y) * camera.zoom * iz + camera.inset_offset_y,
  );

  let q = quad_vertex(vi);
  let min_ndc = 4.0 / camera.viewport_height;
  let max_ndc = 64.0 / camera.viewport_height;
  let point_size = clamp(uni.point_size, min_ndc, max_ndc);
  let offset = vec2f(q.x / camera.aspect, q.y) * (point_size * 0.5);

  let z = camera.zoom * camera.inset_zoom;
  let eff_opacity = uni.opacity * z * z;

  var col: vec4f;
  if uni.usePerPointColor > 0.5 {
    col = unpackColor(pointColors[ii]);
  } else {
    col = uni.color;
  }

  var sel: u32 = 1u;
  if uni.useSelectionMask > 0.5 {
    sel = (selectionMask[ii / 32u] >> (ii % 32u)) & 1u;
  }

  var out: VertOut;
  out.clip_pos = vec4f(center + offset, 0.0, 1.0);
  out.uv = q;
  out.point_color = col;
  out.selected = sel;
  out.effective_opacity = eff_opacity;
  return out;
}

@fragment
fn fs_main(
  @location(0) uv: vec2f,
  @location(1) point_color: vec4f,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) @interpolate(flat) effective_opacity: f32,
) -> @location(0) vec4f {
  let dist = length(uv);
  if dist > 1.0 {
    discard;
  }
  let edge = 1.0 - smoothstep(0.75, 1.0, dist);

  var sel_factor = 1.0;
  if uni.useSelectionMask > 0.5 {
    if selected == 0u {
      sel_factor = 0.1;
    } else {
      sel_factor = 1.0 / max(effective_opacity, 0.01);
    }
  }

  let intensity = edge * effective_opacity * point_color.a * sel_factor;
  let out_alpha = select(0.0, intensity, uni.usePerPointColor > 0.5);
  let rgb = select(point_color.rgb, vec3f(1.0) - point_color.rgb, uni.useSubtractive > 0.5);
  return vec4f(rgb * intensity, out_alpha);
}
