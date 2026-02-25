// Point renderer — instanced quads with SDF circle discard
//
// Reads pre-projected 2D positions from a storage buffer (output of the
// compute-projection shader). Applies 2D camera transform (pan/zoom/aspect).
// Each point is rendered as a unit quad scaled by point_size (NDC units).

struct Uniforms {
  point_size: f32,
  opacity: f32,
  usePerPointColor: f32,
  useSelectionMask: f32,
  color: vec4f,
}

struct Camera {
  pan_x: f32,
  pan_y: f32,
  zoom: f32,
  aspect: f32,
  viewport_height: f32,
  // NDC-space Y offset applied after zoom — used to shift the viewport
  // down when a toolbar covers the top of the canvas.
  inset_offset_y: f32,
  // Zoom multiplier for the inset — scales content to fit in the visible
  // area below the toolbar (typically (H - toolbar) / H).
  inset_zoom: f32,
}

@group(0) @binding(0) var<uniform> uni: Uniforms;
// projected[i*2] = x, projected[i*2+1] = y — output from compute shader
@group(0) @binding(1) var<storage, read> projected: array<f32>;
@group(0) @binding(2) var<uniform> camera: Camera;
// Per-point packed RGBA8 colors (u32: 0xAABBGGRR)
@group(0) @binding(3) var<storage, read> pointColors: array<u32>;
// Per-point selection mask (1 = selected/visible, 0 = dimmed)
@group(0) @binding(4) var<storage, read> selectionMask: array<u32>;

struct VertOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) point_color: vec4f,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) @interpolate(flat) effective_opacity: f32,
}

// Two triangles making a unit square [-1, 1]^2
fn quad_vertex(vi: u32) -> vec2f {
  switch vi {
    case 0u: { return vec2f(-1.0, -1.0); }
    case 1u: { return vec2f( 1.0, -1.0); }
    case 2u: { return vec2f(-1.0,  1.0); }
    case 3u: { return vec2f(-1.0,  1.0); }
    case 4u: { return vec2f( 1.0, -1.0); }
    default: { return vec2f( 1.0,  1.0); }
  }
}

// Unpack RGBA8 from u32 (0xAABBGGRR little-endian)
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
  let px = projected[ii * 2u];
  let py = projected[ii * 2u + 1u];

  // Apply camera: translate then scale, with aspect correction on x.
  // inset_zoom shrinks the scene to fit below the toolbar;
  // inset_offset_y shifts it down so it's centered in the visible area.
  let iz = camera.inset_zoom;
  let center = vec2f(
    (px + camera.pan_x) * camera.zoom * iz / camera.aspect,
    (py + camera.pan_y) * camera.zoom * iz + camera.inset_offset_y,
  );

  let q = quad_vertex(vi);
  // Clamp point size to [2, 32] pixels (physical)
  let min_ndc = 4.0 / camera.viewport_height;
  let max_ndc = 64.0 / camera.viewport_height;
  let point_size = clamp(uni.point_size, min_ndc, max_ndc);
  // Aspect-correct the offset so circles stay circular on non-square canvases
  let offset = vec2f(q.x / camera.aspect, q.y) * (point_size * 0.5);

  // When the clamp enlarged points beyond the intended size, reduce opacity
  // proportionally to the area increase to preserve the paint budget.
  let enlarge = max(1.0, point_size / max(uni.point_size, 0.0001));
  let eff_opacity = uni.opacity / (enlarge * enlarge);

  // Resolve per-point color
  var col: vec4f;
  if uni.usePerPointColor > 0.5 {
    col = unpackColor(pointColors[ii]);
  } else {
    col = uni.color;
  }

  // Read selection
  var sel: u32 = 1u;
  if uni.useSelectionMask > 0.5 {
    sel = selectionMask[ii];
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
  // Smooth anti-aliased edge
  let edge = 1.0 - smoothstep(0.75, 1.0, dist);

  // Selection: boost selected, dim unselected
  var sel_factor = 1.0;
  if uni.useSelectionMask > 0.5 {
    if selected == 0u {
      sel_factor = 0.1;
    } else {
      // Boost selected points to full brightness
      sel_factor = 1.0 / max(effective_opacity, 0.01);
    }
  }

  // Additive intensity — accumulates light on the dark background
  let intensity = edge * effective_opacity * point_color.a * sel_factor;
  return vec4f(point_color.rgb * intensity, 0.0);
}
